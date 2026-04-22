from typing import Any, Dict, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase
import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(
        prompt_strs: List[str],
        output_strs: List[str],
        tokenizer: PreTrainedTokenizerBase,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    assert len(prompt_strs) == len(output_strs)
    bs = len(prompt_strs)

    prompt_tok = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    output_tok = tokenizer(
        output_strs,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )

    prompt_ids_list = prompt_tok["input_ids"]
    output_ids_list = output_tok["input_ids"]

    # 拼接 full_ids = prompt + output
    full_ids_list = [p + o for p, o in zip(prompt_ids_list, output_ids_list)]
    prompt_lens = [len(p) for p in prompt_ids_list]
    output_lens = [len(o) for o in output_ids_list]

    # pad_id 选择：Qwen2 通常 pad_token_id=None，所以用 eos 作为 pad（常见做法）
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # ========= 关键：先 pad，再 shift =========
    # 先统一时间轴，再做自回归对齐，避免错位
    # 1. 统一长度（pad）,让所有序列在同一个“时间轴”; 2. 做 shift（构造预测任务）; 3. 只在 output 上计算loss
    max_full_len = max(len(x) for x in full_ids_list) if bs > 0 else 0
    full_padded = torch.full((bs, max_full_len), pad_id, dtype=torch.long)

    for i, ids in enumerate(full_ids_list):
        if len(ids) > 0:
            full_padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    # shift
    input_ids = full_padded[:, :-1].contiguous()
    labels = full_padded[:, 1:].contiguous()

    # response_mask 对齐 labels（只覆盖 output 部分，不包括 padding）
    response_mask = torch.zeros_like(labels, dtype=torch.long)
    for i in range(bs):
        p_len = prompt_lens[i]
        o_len = output_lens[i]
        if o_len == 0:
            continue
        start = max(p_len - 1, 0)
        end = p_len + o_len - 1  # exclusive in labels coords
        end = min(end, labels.size(1))
        if end > start:
            response_mask[i, start:end] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-token entropy of next-token predictions.
    把模型在每个位置输出的 logits 转成“下一个 token 的预测熵”，也就是衡量模型在该位置有多不确定。
    LLM在每个 token 位置上的预测不确定性（entropy，熵）越大，说明模型越不确定，越难预测。

    Args:
        logits: Tensor of shape (batch_size, sequence_length, vocab_size)

    Returns:
        # 每个位置对应一个熵值标量，值越大表示分布越分散、模型越拿不准；值越小表示分布越尖锐、模型越确定。
        Tensor of shape (batch_size, sequence_length)
    """
    # 把最后一维 vocab_size 上的原始分数(unnormalized logits)归一化成对数概率 log p。
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # H(P) = - Σ p(x) log p(x)
    # 把 vocab_size 那一维求和，得到每个 batch、每个时间步的熵
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities log p_theta(x_t | x_<t)
    for a causal LM, and optionally per-token entropy.

    注意: padding 位置的 log_probs 和 entropy 也计算了. 哪些位置该参与 loss，通常要靠外面的 mask 决定。  
    Args:
        model: HF causal LM (on correct device; set eval/no_grad outside if desired)
        input_ids: (batch_size, sequence_length)
        labels: (batch_size, sequence_length)
        return_token_entropy: if True, also return token_entropy (batch_size, sequence_length)

    Returns:
        dict with:
          - "log_probs": (batch_size, sequence_length), 整个序列的 log_probs 越大，说明模型对这个序列的预测越准确。
          - "token_entropy": (batch_size, sequence_length) if requested, 每个 token 的 entropy 越大，说明模型对这个 token 的预测越不确定。
    """
    # Forward: logits (batch_size, sequence_length, vocabulary_size)
    logits = model(input_ids).logits
    # log-probs over vocab: (batch_size, sequence_length, vocabulary_size)
    log_probs_vocab = F.log_softmax(logits, dim=-1)
    # Select log-prob of the label token at each position.
    # labels: (batch_size, sequence_length) -> (batch_size, sequence_length, 1) for gather
    log_probs = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    out: Dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        out["token_entropy"] = compute_entropy(logits)

    return out


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        normalize_constant: float,
        dim: int | None = None,
) -> torch.Tensor:
    """
    Sum tensor elements where mask == 1 and normalize by a constant.
    """
    # 只保留 mask == 1 的位置，其余置 0
    masked_tensor = tensor * mask
    # 求和
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
    # 归一化
    return summed / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    One SFT microbatch step: masked NLL, batch-mean, normalize, grad-acc scaling, backward.
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log probabilities of the policy.
        response_mask: (batch_size, sequence_length), the mask of the response tokens.
        gradient_accumulation_steps: int, the number of microbatches per optimizer step.
        normalize_constant: float, the constant to normalize the loss by.

    Returns:
        tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - loss: (scalar), the microbatch loss.
            - metadata: dict[str, torch.Tensor], the metadata.
    """
    # mask -> same dtype as log probs
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    # per-token NLL
    per_token_nll = -policy_log_probs  # (batch_size, sequence_length)
    # sum over sequence per example (mask out prompt/pad)
    per_example_nll = (per_token_nll * mask).sum(dim=1)  # (batch_size,)
    # normalize by constant (as assignment says)
    per_example_nll = per_example_nll / float(normalize_constant)
    # batch mean
    microbatch_loss = per_example_nll.mean()  # scalar
    # scale for gradient accumulation
    loss = microbatch_loss / float(gradient_accumulation_steps)
    # backward
    loss.backward()
    metadata = {
        "microbatch_loss": microbatch_loss.detach(),
    }
    return loss, metadata