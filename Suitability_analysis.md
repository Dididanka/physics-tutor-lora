# Local LLM fine-tuning: suitability analysis

Assessed: 18 July 2026

## Verdict

This MacBook Pro is **very suitable for learning supervised fine-tuning (SFT) with LoRA/QLoRA adapters** on current small and medium local language models. It is not a sensible machine for training a model from scratch or full-parameter fine-tuning. The practical sweet spot is a **2B–4B parameter instruct model**, with an **8B model** reserved for careful, slower experiments.

**Recommended learning stack:** `MLX-LM` + LoRA, in an isolated Python 3.12 environment.

**Recommended first model:** a 4-bit MLX version of **Qwen3-4B-Instruct** (or its current direct successor once its MLX-LM compatibility is confirmed). It is the best balance of modern capability, well-understood tooling, experiment speed, and memory headroom on this machine.

The newest model already installed locally is **`qwen3.5:2b`**. It is an excellent *inference baseline* and a good low-cost training candidate once its `qwen35` architecture is confirmed supported by the installed/current MLX-LM release. Do not try to train the Ollama file itself: it is a Q8 GGUF runtime artifact rather than a source training checkpoint.

## Measured machine profile

| Resource | Observation | Fine-tuning implication |
|---|---|---|
| Hardware | MacBook Pro, Apple M5 | Native Apple Silicon training is available through MLX/Metal. |
| Compute | 10-core CPU; 10-core integrated GPU; Metal 4 | Good interactive LoRA training, not workstation-class throughput. |
| Memory | 24 GB unified memory | The GPU and macOS share this memory. Keep the model, training activations, and other apps within this one budget. |
| Storage | 627 GiB free | More than enough for checkpoints, datasets, adapters, and several quantized models. |
| Current memory state | 54% free; historic memory compression and swap activity | Close large apps and stop large Ollama models before training. Avoid treating all 24 GB as usable training memory. |
| Python | 3.12 and 3.13 installed; global default is 3.14.4 | Use Python **3.12** for the first environment; its ML compatibility is the least risky. |
| Tooling | `uv` 0.9.27 and Ollama 0.24.0 installed | The basic runtime/setup is ready; no Python ML training packages are installed yet. |

No dataset or existing training project was found in this directory.

## Locally available models

| Model | Local size | Suitability |
|---|---:|---|
| `qwen3.5:2b` | 2.7 GB, Q8 | Best existing small, current baseline. It supports text, vision, tools, and thinking, but its hybrid `qwen35` architecture should be compatibility-checked before training. |
| `qwen3-vl:8b` | 6.1 GB, Q4 | Inference is suitable. Fine-tune only after learning text-only SFT; multimodal datasets and evaluation add substantial complexity. |
| `lfm2:24b` | 14 GB | Fine for constrained inference. Not appropriate for local fine-tuning with 24 GB unified memory. |
| `qwen3.5:27b` | 17 GB | Inference-only on this machine; leave no credible training headroom. |
| `nemotron-cascade-2:30b`, `glm-4.7-flash:q4_K_M` | 19–24 GB | Not suitable for local training; even inference will compete heavily with macOS for memory. |

Cloud-labelled Ollama models are not local training candidates.

## What to fine-tune, ranked

| Priority | Model choice | Method | Why |
|---:|---|---|---|
| 1 | **Qwen3-4B-Instruct, MLX 4-bit** | LoRA; 1,024-token sequences; batch size 1 | The recommended first project. Strong general quality without making each experiment slow or fragile. It leaves useful memory headroom for macOS and evaluation. |
| 2 | **Qwen3.5 2B source/MLX checkpoint** | LoRA, only after architecture compatibility check | The most current model family visible on this machine and very fast to iterate. Use it where fast feedback matters more than maximum answer quality. The installed Ollama GGUF remains an evaluation baseline, not the input model. |
| 3 | **Qwen3 8B, MLX 4-bit** | LoRA; batch 1; 512–1,024 tokens | A realistic upper-bound experiment after the 4B run works. Expect longer iterations and less headroom; do not run other large local models alongside it. |
| 4 | **Qwen3-VL 8B** | Multimodal LoRA | Suitable only when the real task genuinely uses images or documents. Start text-only unless vision is a requirement. |

I would **not** start with a 24B–30B model, full fine-tuning, long-context training, reinforcement fine-tuning, or a vision model. Each would turn a useful learning project into a memory- and debugging-heavy exercise.

## Why LoRA rather than full fine-tuning

LoRA freezes the base model and trains small adapter matrices. This preserves the general capability of a good pretrained model, reduces memory use dramatically, and produces a small, versionable adapter file. For this machine it is the right default.

Full fine-tuning needs model weights, gradients, optimiser states, and activations. Even a 2B model can exceed 24 GB in a practical optimizer configuration; a 4B or 8B model is firmly outside a comfortable local full-tuning budget. Quantizing inference weights does **not** make full training equivalently cheap.

## Practical learning path

1. **Define one narrow behaviour to improve.** Examples: answer questions from a house style, turn a support request into a structured JSON response, write a specific kind of code review, or classify a small set of document types. Fine-tuning is best for consistent style, format, task procedure, and domain phrasing—not for loading a large knowledge base. Use retrieval for frequently changing or large factual knowledge.

2. **Create high-quality instruction examples.** Begin with roughly 200–1,000 carefully reviewed examples. Keep 10–20% aside as a test set that is never trained on. A few hundred precise examples beat thousands of weak or repetitive examples.

3. **Start with short examples.** Use a maximum sequence length of 512 or 1,024 tokens. The 262k-token context advertised by the installed Qwen models is an inference limit, not a sensible first training length; activation memory grows rapidly with sequence length.

4. **Run a baseline evaluation before training.** Prompt the untouched model with 20–50 held-out tasks, record outputs and an objective rubric, then compare the adapter against exactly the same tasks. A lower training loss alone does not show that the model improved.

5. **Train one small LoRA run, inspect failures, then iterate.** Change one variable at a time: data quality, prompt template, number of adapter layers, learning rate, or training iterations.

6. **Only then try the 8B model.** If the 4B model misses reasoning depth despite good data, moving to 8B may help. If it misses current facts, use retrieval instead of a larger fine-tune.

## Suggested project layout and data format

```text
FineTune/
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl          # never supplied to the trainer
├── adapters/
├── evaluations/
└── notebooks/              # optional; scripts are easier to reproduce
```

Use chat messages that match the chosen model's chat template. For example, one JSON object per line:

```json
{"messages":[{"role":"system","content":"You return concise, valid JSON."},{"role":"user","content":"Extract customer and priority: Acme has a production outage."},{"role":"assistant","content":"{\"customer\":\"Acme\",\"priority\":\"critical\"}"}]}
```

Keep system prompts stable, remove duplicates, and include examples of the failures you want to correct. Never include secrets, private data without permission, or answers that the model should refuse.

## Minimal, reproducible setup

Use a project-local environment rather than the global Python 3.14 installation:

```zsh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -U mlx-lm datasets
```

Then use the current `mlx_lm.lora --help` output to select supported options for the selected model. A deliberately conservative first run looks like this conceptually:

```zsh
python -m mlx_lm.lora \
  --model <MLX-4-bit-Qwen3-4B-Instruct-checkpoint> \
  --train --data ./data \
  --adapter-path ./adapters/first-run \
  --batch-size 1 --max-seq-length 1024 \
  --num-layers 16 --iters 600
```

Exact flags can change between `mlx-lm` releases, so treat this as a starting shape and verify the installed command help before launching. Save the command, package versions, dataset revision, parameters, training loss, evaluation prompts, and results with every run.

After evaluation, keep the adapter separate from the base model while experimenting. When it is proven useful, fuse/convert it with the MLX tooling and package the result for Ollama only if that deployment path supports the model architecture.

## Capacity guardrails

- Begin at 1,024 tokens and batch size 1. Reduce to 512 tokens if memory pressure appears.
- Use a 4-bit base checkpoint plus LoRA; do not choose a Q8 base model for the first training experiment.
- Stop `ollama` sessions serving 14–24 GB models before training. Do not run a large model and an editor/video browser workload at the same time.
- A 4B run should be comfortable; an 8B run is an experiment, not the daily default. Do not rely on 24 GB free just because a quantized 27B model can start for inference.
- Save adapters/checkpoints to the internal SSD and retain the best evaluation result, not every trial indefinitely.
- Training has no guaranteed quality gain. Stop if held-out outputs become more verbose, less factual, or less format-compliant than the base model.

## Recommended next concrete project

Build a text-only instruction-tuning exercise around one measurable behaviour, using a 4-bit **Qwen3-4B-Instruct** MLX checkpoint and LoRA. Create 300 examples, reserve 50 for test, train at 1,024 tokens, and compare against the untouched `qwen3.5:2b`/base-model outputs. This teaches the entire workflow—data curation, chat templates, training, evaluation, and deployment—without exceeding the machine's comfortable operating envelope.

## Useful primary documentation

- [MLX-LM repository and LoRA examples](https://github.com/ml-explore/mlx-lm)
- [MLX examples](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm)
- [Qwen3 model documentation](https://huggingface.co/Qwen/Qwen3-4B)
- [Ollama model and Modelfile documentation](https://docs.ollama.com/)

This assessment is based on direct local inspection of this MacBook and its installed runtimes on 18 July 2026. Model availability and trainer support evolve quickly; before downloading a checkpoint, verify the current MLX-LM model support list and the base model's license.
