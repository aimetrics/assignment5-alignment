# CS336 Spring 2025 Assignment 5: Alignment

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Author & acknowledgments

**Author (this edition):** aimetrics

**Acknowledgment:** This version’s documentation and experiment setup lean heavily on the community course reproduction in [DataWhale diy-llm — `coursework/assignment5-alignment`](https://github.com/datawhalechina/diy-llm/tree/main/coursework/assignment5-alignment). Thank you to the DataWhale contributors for the detailed walkthrough and runnable layout.

**说明（中文）：** 本次仓库版本由 **aimetrics** 维护整理；大量参考了上述 DataWhale 课程仓库中的 assignment5-alignment 材料。

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

