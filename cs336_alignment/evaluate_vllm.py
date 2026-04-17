import json
import os
from typing import Callable, Dict, Any
from collections import Counter

import torch
from tqdm import tqdm

from drgrpo_grader import r1_zero_reward_fn

# ---------------------------------------------------------------------------
# HuggingFace Transformers + MPS backend
# Drop-in replacements for vllm.LLM and vllm.SamplingParams so that
# evaluate_vllm() works unchanged on Apple Silicon (or any non-CUDA device).
# ---------------------------------------------------------------------------

class HFSamplingParams:
    """Mimics vllm.SamplingParams."""

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
        include_stop_str_in_output: bool = False,
    ):
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop or []
        self.include_stop_str_in_output = include_stop_str_in_output


class _HFCompletionOutput:
    """Mimics a single vllm CompletionOutput (output.outputs[0])."""

    def __init__(self, text: str):
        self.text = text


class _HFRequestOutput:
    """Mimics a single vllm RequestOutput."""

    def __init__(self, prompt: str, generated_text: str):
        self.prompt = prompt
        self.outputs = [_HFCompletionOutput(generated_text)]


class HFLLM:
    """
    Drop-in replacement for vllm.LLM using HuggingFace Transformers.

    Supports MPS (Apple Silicon), CUDA, and CPU.
    The public interface mirrors vllm.LLM: only .generate() is needed by
    evaluate_vllm().
    """

    def __init__(
        self,
        model: str,
        dtype: torch.dtype | None = None,
        device: str | None = None,
        batch_size: int = 8,
        **kwargs,  # absorb unused vllm kwargs (tensor_parallel_size, etc.)
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # MPS has incomplete float16/bfloat16 support (matmul shape errors, logit
        # overflow during sampling). float32 is the only reliable choice on MPS.
        if dtype is None:
            if device == "mps":
                dtype = torch.float32
            else:
                dtype = torch.bfloat16

        print(f"[HFLLM] Loading tokenizer from {model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[HFLLM] Loading model onto {device} with dtype={dtype} ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            dtype=dtype,
            attn_implementation="eager",  # MPS doesn't support optimized attention kernels
        ).to(device)
        self.model.eval()
        self.batch_size = batch_size
        # Left-pad so all prompts in a batch end at the same position,
        # which is required for correct slicing of generated tokens.
        self.tokenizer.padding_side = "left"

    def generate(
        self,
        prompts: list[str],
        sampling_params: HFSamplingParams,
    ) -> list[_HFRequestOutput]:
        """Generate responses in batches."""
        outputs = []
        for i in tqdm(range(0, len(prompts), self.batch_size), desc="Generating"):
            batch = prompts[i : i + self.batch_size]
            texts = self._generate_batch(batch, sampling_params)
            for prompt, text in zip(batch, texts):
                outputs.append(_HFRequestOutput(prompt, text))
        return outputs

    def _generate_batch(self, prompts: list[str], sp: HFSamplingParams) -> list[str]:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.device)
        padded_input_len = inputs["input_ids"].shape[1]

        greedy = sp.temperature == 0.0
        gen_kwargs: dict = dict(
            max_new_tokens=sp.max_tokens,
            do_sample=not greedy,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if not greedy:
            gen_kwargs["temperature"] = sp.temperature
            gen_kwargs["top_p"] = sp.top_p

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for seq in output_ids:
            # Slice off the (padded) input prefix; only keep generated tokens.
            new_tokens = seq[padded_input_len:]
            generated = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            # Apply stop strings (post-process, matching vllm semantics).
            for stop_str in sp.stop:
                idx = generated.find(stop_str)
                if idx != -1:
                    end = idx + len(stop_str) if sp.include_stop_str_in_output else idx
                    generated = generated[:end]
                    break
            results.append(generated)
        return results


# ---------------------------------------------------------------------------
# Conditional vllm import — only loaded when actually used
# ---------------------------------------------------------------------------

def _load_vllm():
    from vllm import LLM, SamplingParams  # noqa: F401 — re-exported for callers
    return LLM, SamplingParams

# Define the model path and prompt path
MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
PROMPT_PATH = "cs336_alignment/prompts/r1_zero.prompt"
MATH_VALIDATION_PATH = "data/gsm8k/test.jsonl"
OUTPUT_DIR = "results/base"

def load_r1_zero_prompt(prompt_file_path: str) -> str:
    """Loads the r1_zero prompt template from a file."""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def format_prompt(question: str, prompt_template: str) -> str:
    """Formats the question into the r1_zero prompt template."""
    return prompt_template.format(question=question)

def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str, bool], Dict[str, float]],
    dataset_path: str,
    prompt_template: str,
    eval_sampling_params: Any,
    output_filepath: str,
    fast: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    if not os.path.exists(os.path.dirname(output_filepath)):
        os.makedirs(os.path.dirname(output_filepath))

    questions = []
    ground_truths = []
    prompts = []
    
    # Handle both JSONL and JSON array formats
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            # JSON array format (MATH data)
            data = json.load(f)
        else:
            # JSONL format (GSM8K data)
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    for example in data:
        # Handle different field names
        question = example.get('question', example.get('problem', ''))
        if 'expected_answer' in example:
            # MATH format: direct answer
            answer = example['expected_answer']
        elif 'answer' in example:
            # GSM8K format: extract final answer after ####
            answer_text = example['answer']
            import re
            m = re.search(r"####\s*(.+)\s*$", answer_text.strip())
            answer = m.group(1).strip() if m else answer_text.strip()
        else:
            answer = ''

        questions.append(question)
        ground_truths.append(answer)
        prompts.append(format_prompt(question, prompt_template))

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_answer_reward = 0.0
    total_format_reward = 0.0
    total_reward = 0.0
    combo_counts = Counter()

    for i, output in enumerate(outputs):
        # Fix for potential empty outputs
        generated_text = output.outputs[0].text if output.outputs else ""
        
        # Use r1_zero_reward_fn which expects `response` and `ground_truth`
        # The `response` for r1_zero_reward_fn is the raw model generated text
        # and it handles the extraction of the answer part itself.
        rewards = reward_fn(generated_text, ground_truths[i], fast=fast)

        fr = int(rewards.get("format_reward", 0.0) >= 0.5)
        ar = int(rewards.get("answer_reward", 0.0) >= 0.5)
        combo_counts[(fr, ar)] += 1

        results.append({
            "question": questions[i],
            "ground_truth": ground_truths[i],
            "prompt": output.prompt, # Use output.prompt for exact prompt sent to vLLM
            "model_response": generated_text,
            "rewards": rewards
        })
        total_answer_reward += rewards["answer_reward"]
        total_format_reward += rewards["format_reward"]
        total_reward += rewards["reward"]

    num_examples = len(results)
    avg_answer_reward = (total_answer_reward / num_examples) if num_examples else 0.0
    avg_format_reward = (total_format_reward / num_examples) if num_examples else 0.0
    avg_reward = (total_reward / num_examples) if num_examples else 0.0

    print(f"Evaluation Results:")
    print(f"  Average Answer Reward: {avg_answer_reward:.4f}")
    print(f"  Average Format Reward: {avg_format_reward:.4f}")
    print(f"  Average Total Reward: {avg_reward:.4f}")
    print(f"  Combo Counts: {combo_counts}")

    with open(output_filepath, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_filepath}")

    combo_table = {
        "format=1 answer=1": combo_counts[(1, 1)],
        "format=1 answer=0": combo_counts[(1, 0)],
        "format=0 answer=0": combo_counts[(0, 0)],
        "format=0 answer=1": combo_counts[(0, 1)], # Should ideally be 0
    }

    metrics = {
        "n": num_examples,
        "format_rate": avg_format_reward,
        "answer_accuracy": avg_answer_reward,
        "reward_mean": avg_reward,
        "counts": combo_table,
    }
    return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="hf",
        help="Inference backend: 'vllm' (Linux/CUDA) or 'hf' (HuggingFace, MPS/CPU/CUDA)",
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--dataset-path", default=MATH_VALIDATION_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for HF backend (ignored for vllm)",
    )
    args = parser.parse_args()

    r1_zero_template = load_r1_zero_prompt(PROMPT_PATH)

    sampling_kwargs = dict(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    if args.backend == "vllm":
        LLM, SamplingParams = _load_vllm()
        print(f"Initializing vLLM model from {args.model_path}...")
        llm = LLM(model=args.model_path, dtype=torch.bfloat16)
        sampling_params = SamplingParams(**sampling_kwargs)
    else:
        print(f"Initializing HF model from {args.model_path}...")
        llm = HFLLM(model=args.model_path, batch_size=args.batch_size)
        sampling_params = HFSamplingParams(**sampling_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "zero_shot_math_evaluation.jsonl")

    evaluation_metrics = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        dataset_path=args.dataset_path,
        prompt_template=r1_zero_template,
        eval_sampling_params=sampling_params,
        output_filepath=output_file,
        fast=True,
    )

    print("\nFinal Evaluation Metrics:")
    print(json.dumps(evaluation_metrics, indent=4, ensure_ascii=False))