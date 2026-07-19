# Seed data

These examples are intentionally small and hand-written. They serve two purposes:

1. exercise the JSONL validator and MLX-LM training workflow; and
2. demonstrate the desired physics-tutor behaviour and fictional Einstein-inspired persona.

They cannot make a model a physics expert. Do not use the seed set to judge model quality or to claim a successful fine-tune. Build `data/raw/` from accurate, permissioned instructional examples, compile it, and preserve the held-out test split.

`test_prompts.jsonl` is an evaluation-only set. It is not SFT data and must never be added to `train.jsonl` or `valid.jsonl`.
