# LoRA and QLoRA FAQ

## What is LoRA?

**LoRA** means *Low-Rank Adaptation*. It is a way to fine-tune a pretrained model without changing all of its billions of original parameters.

Instead, LoRA freezes the base model and trains small extra matrices—called **adapters**—at selected layers. At inference time, the adapter's learned adjustment is added to the base model's weights.

In simple terms: the base model stays intact; LoRA learns a compact set of “adjustment notes” for a new task, style, or domain.

## What does LoRA help this project learn?

For this physics tutor, LoRA can improve consistent behaviour such as:

- explaining concepts from first principles;
- showing assumptions, units, and short derivations;
- using a warm, thought-experiment-oriented teaching style;
- acknowledging uncertainty rather than bluffing; and
- maintaining the fictional Einstein-inspired persona boundary.

It will not reliably inject a large new body of facts into a weak model. For changing reference knowledge or a large private corpus, retrieval (RAG) is usually the better tool.

## Why not fully fine-tune the model?

Full fine-tuning updates every base-model weight. Training then needs memory for the weights, gradients, optimizer state, and intermediate activations. That is expensive: even a small multi-billion-parameter model can exceed this Mac's practical 24 GB unified-memory budget.

LoRA trains only a small fraction of parameters, so it uses much less memory, produces small adapter files, and makes experiments easier to version and undo.

## What is a LoRA rank?

The **rank** controls the size and expressive power of the adapter matrices.

- Lower rank: fewer trainable parameters, less memory, smaller adapter, but less capacity to learn a change.
- Higher rank: more capacity, but greater memory use and more risk of overfitting a small dataset.

The right rank depends on data quality and task complexity. Change it only after establishing a baseline with a conservative configuration.

## What is quantization?

Model weights are normally stored as high-precision numbers, commonly 16-bit floating point. **Quantization** stores them in fewer bits—such as 8-bit or 4-bit representations—to reduce memory use and disk size.

For example, a 4-bit model generally needs roughly one quarter of the raw storage of a 16-bit model, plus some metadata and runtime overhead. The trade-off is a small, model-dependent loss of numerical precision and sometimes quality.

Quantization is especially valuable on Apple Silicon because the GPU and operating system share the same unified memory.

## What is QLoRA?

**QLoRA** combines both ideas:

1. Load the frozen base model in a low-bit quantized format, commonly 4-bit.
2. Train LoRA adapters in higher precision on top of that frozen base.

This makes fine-tuning substantially more memory-efficient. The base model remains quantized and unchanged; only the relatively tiny adapter weights are updated.

For this repository, “4-bit MLX base model + LoRA adapter” is the practical equivalent of the QLoRA approach.

## Does a quantized model train every weight in 4-bit?

No. In QLoRA, the quantized base weights are normally **frozen**. They are used during the forward pass, while the LoRA adapters receive gradients and are updated. This is why QLoRA is much cheaper than full fine-tuning.

## Why use a 4-bit base model here instead of the installed Q8 Ollama model?

A 4-bit MLX checkpoint leaves more memory for activations, adapters, and macOS, which makes training more reliable on this 24 GB Mac. The installed `qwen3.5:2b` Q8 GGUF model is useful for inference and baseline comparisons, but it is an Ollama runtime file—not the source/MLX checkpoint the training workflow expects.

## What files come out of a LoRA run?

The key output is an **adapter**: a small collection of learned LoRA weights plus configuration metadata. It depends on its exact base model; it is not a standalone model.

Keep the following together in experiment notes:

- base model name and revision;
- quantization format;
- LoRA configuration and training command;
- dataset revision and split seed;
- adapter directory; and
- held-out evaluation results.

## Can I use the adapter with any model?

No. An adapter is tied to the model architecture and usually the exact base checkpoint it was trained against. A Qwen adapter should not be attached to an unrelated Llama, Gemma, or different Qwen base model.

## Does LoRA make the model an expert after one run?

No. It adjusts behaviour from the examples it sees. Good results require accurate, varied, permissioned examples and held-out evaluation. A small, repetitive dataset can make a model sound more confident without making it more correct.

## Recommended starting point for this project

Use a 4-bit MLX checkpoint of Qwen3-4B-Instruct, LoRA adapters, batch size 1, and a 1,024-token maximum sequence length. First improve a narrow measurable behaviour, evaluate it against held-out physics prompts, then expand the curated dataset before changing model size or rank.
