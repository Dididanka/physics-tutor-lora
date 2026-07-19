# Physics Persona LoRA

A reproducible, local fine-tuning project for a physics tutor with a warm, curious, Einstein-inspired teaching manner. It uses **LoRA adapters** with [MLX-LM](https://github.com/ml-explore/mlx-lm), which is the most appropriate training stack for this Apple Silicon Mac.

The project intentionally teaches a *persona and teaching method*, not impersonation. The assistant is a fictional physics educator: it must not claim to be Albert Einstein, invent biographical experiences, or present fabricated quotations as his.

## What this project does

- Validates chat-format JSONL examples before training.
- Starts with a compact, hand-reviewed physics seed dataset.
- Uses a deterministic compiler to create training and validation splits from additional curated source data.
- Trains a small LoRA adapter against a 4-bit MLX base model.
- Runs the same held-out prompts against any Ollama model for before/after comparison.
- Keeps models, adapters, source data, and evaluations out of version control by default.

It does **not** train a model from scratch or full-fine-tune all model weights. On this 24 GB unified-memory Mac, adapter tuning is the correct trade-off.

## Recommended baseline

Use a text-only, dense 4B instruct model first:

```text
mlx-community/Qwen3-4B-4bit
```

The exact model checkpoint is configured in [`configs/qwen3-4b-lora.env`](configs/qwen3-4b-lora.env). The project uses the local `configs/` model directory and does not download a base model during training. The checkpoint and tokenizer files are intentionally not committed to Git because model artifacts are large. Before training after a fresh clone, download the MLX files for mlx-community/Qwen3-4B-4bit into `configs/`. The existing `qwen3.5:2b` Ollama model is useful as an inference baseline, but its Q8 GGUF file is not a training checkpoint.

## Setup

Python 3.12 is deliberately used instead of the globally installed Python 3.14 because it is the safer compatibility choice for ML packages.

```zsh
uv venv --python 3.12
source .venv/bin/activate
uv sync --active
```

Validate the included data before any training:

```zsh
python scripts/validate_dataset.py data/seed/train.jsonl --require-canonical-persona
python scripts/validate_dataset.py data/seed/valid.jsonl --require-canonical-persona
python -m unittest discover -s tests -v
```

## Data contract

SFT input is one JSON object per line, using `messages` in ordinary chat order:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"Why does ...?"},{"role":"assistant","content":"..."}]}
```

Training examples must satisfy these rules:

- Roles are `system`, `user`, or `assistant`.
- Each message has a non-empty string `content`.
- The final message is an `assistant` answer.
- A `system` message is present and equals the canonical persona in [`prompts/physics_einstein_inspired_system.txt`](prompts/physics_einstein_inspired_system.txt).
- Examples must be accurate, self-contained, licensed for training, and scrubbed of private or secret information.

The seed data proves the plumbing; it is not enough to make a real physics expert. For a meaningful result, add several hundred carefully verified examples across mechanics, electromagnetism, thermodynamics, quantum theory, relativity, and problem solving. Preserve a test set that is never given to the trainer.

This repository now includes Physics SFT v1, a 3,000-record original synthetic curriculum in data/raw/physics_sft_v1.jsonl. It is compiled into the processed dataset used by the training launcher; see data/raw/physics_sft_v1.md for its composition and review limits. It is intentionally pure SFT: no retrieval system is part of this project.

## Add and compile curated data

Place one or more source JSONL files in `data/raw/`. Source files use the same message format. The compiler canonicalizes the system message, removes exact duplicate user prompts, and makes deterministic splits:

```zsh
python scripts/compile_dataset.py \
  --input data/raw \
  --output data/processed \
  --seed 42 \
  --valid-fraction 0.10 \
  --test-fraction 0.10

python scripts/validate_dataset.py data/processed/train.jsonl --require-canonical-persona
python scripts/validate_dataset.py data/processed/valid.jsonl --require-canonical-persona
```

`data/processed/test.jsonl` is deliberately for evaluation only. Do not pass it to MLX-LM.

## Train a LoRA adapter

The training script reads conservative settings from the config file. It defaults to `data/processed`; point it at the seed dataset only to verify the workflow, never to claim a useful model.

```zsh
# First real run, after compiling your curated data
zsh scripts/train_lora.sh configs/qwen3-4b-lora.env

# Plumbing-only smoke run with the provided seed examples
DATA_DIR=data/seed ITERS=20 zsh scripts/train_lora.sh configs/qwen3-4b-lora.env
```

Expected first-run defaults: 4-bit base model, batch size 1, 1,024-token limit, 16 adapted layers, and 600 iterations. Start there. If macOS reports memory pressure, reduce `MAX_SEQ_LENGTH` to 512 before changing anything else.

With the current 2,459-record processed SFT set, 600 iterations is only a smoke run of roughly one quarter of an epoch at batch size 1. Run one epoch first with ITERS=2500, review its held-out results, then consider a longer run rather than blindly increasing epochs.

The command loads the local MLX model. Stop large Ollama processes and close memory-intensive apps beforehand.

## Evaluate before and after

The test prompts contain rubrics, not target answers. Run them against your base and adapted/deployed model and compare the two JSONL result files. They are designed to catch incorrect physics, missing units, overconfident claims, and persona violations.

```zsh
# Baseline; uses the already-installed local model by default
python scripts/evaluate_ollama.py \
  --model qwen3.5:2b \
  --output evaluations/baseline-qwen35-2b.jsonl

# Repeat after packaging a compatible adapter/model for Ollama
python scripts/evaluate_ollama.py \
  --model physics-tutor:latest \
  --output evaluations/physics-tutor.jsonl
```

Review the outputs manually against the rubrics. Training loss is a diagnostic, not evidence that the tutor became more correct.

Evaluation turns off model thinking mode by default so timing and output are comparable. Add `--think` when the ability to use a reasoning mode is itself what you want to evaluate. Add `--max-cases 1` for a one-prompt smoke check.

## Serving the adapter

Keep the LoRA adapter separate while iterating. Use the current MLX-LM generation command with the same base model and adapter to test it. Only fuse or convert an adapter for Ollama after verifying the current MLX-LM and Ollama instructions for the exact model architecture. An MLX adapter is not automatically an Ollama-compatible adapter.

## Persona and safety decisions

The persona is intentionally inspired by Einstein's public reputation for thought experiments, humility, and conceptual clarity. It must:

- Say it is a fictional physics tutor if directly asked whether it is Einstein.
- Separate established physics from speculation and state assumptions.
- Show a short derivation or reasoning path when it helps, including units for numerical work.
- Correct itself rather than bluffing; recommend a qualified professional for dangerous experiments or high-stakes decisions.
- Decline to provide dangerous experimental procedures, weapons guidance, or claims of professional certification.

Avoid training on scraped answer dumps, dubious social-media explanations, copyrighted textbook solutions without permission, or personally identifiable student conversations.

## Repository map

```text
configs/             Training defaults
data/seed/           Small reviewed example dataset and held-out prompts
data/raw/            Private, curated source examples (ignored by Git)
data/processed/      Generated train/valid/test splits (ignored by Git)
prompts/             Canonical system persona
scripts/             Validation, compilation, training, and evaluation tools
tests/               Standard-library tests for the data pipeline
adapters/             LoRA outputs (ignored by Git)
evaluations/         Model responses and review notes (ignored by Git)
```

## Success criteria for the first real adapter

Keep an adapter only if it improves at least 80% of a held-out, representative test set *without* making baseline physics answers worse. In review, prioritize: correct principle, valid derivation, units and estimates where relevant, uncertainty disclosure, concise tutoring style, and adherence to the fictional-persona boundary.
