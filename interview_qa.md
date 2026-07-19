# Physics Tutor LoRA: Interview Q&A

These are example questions a technical interviewer could ask the author of this project, with concise answers grounded in the current implementation.

## Project and architecture

### What did you build?

I built a local supervised fine-tuning project for a physics tutor with a warm, thought-experiment-oriented teaching persona. It uses a Qwen3 4B model on Apple Silicon, MLX-LM for LoRA training, and a locally downloaded 4-bit MLX checkpoint.

### Why did you choose a 4B model?

A 4B model balances general capability with practical training on a MacBook Pro with 24 GB unified memory. It leaves headroom for macOS and activations during LoRA training. Larger 24B to 30B local models may be usable for constrained inference but are not sensible fine-tuning targets on this hardware.

### Why did you choose MLX-LM?

MLX-LM is designed for Apple Silicon and uses the shared-memory GPU efficiently. It supports local model loading, LoRA adapters, generation with an adapter attached, and fusion for MLX-compatible deployment. It avoids depending on CUDA, which is unavailable on this Mac.

### Is this a RAG application?

No. This project intentionally uses pure supervised fine-tuning. The model learns response behaviour, explanations, and problem-solving patterns from the SFT dataset. There is no vector database, embedding pipeline, retriever, or external knowledge lookup.

### What does “Einstein-inspired” mean here?

It means the tutor uses curiosity, conceptual clarity, thought experiments, humility, and clear assumptions as teaching traits associated with Einstein's public reputation. The persona explicitly says it is fictional and must not claim to be Albert Einstein, invent personal memories, or fabricate quotations.

## Fine-tuning approach

### What is LoRA and why did you use it?

LoRA freezes the base-model weights and trains small adapter matrices at selected layers. It is much cheaper than full fine-tuning because it avoids gradients and optimizer state for every base parameter. The adapter is compact, easy to version, and can be applied to the unchanged base model at inference time.

### Is this really QLoRA?

The project uses a 4-bit quantized MLX base model with LoRA adapters, which is the practical QLoRA-style approach for this environment. The quantized base remains frozen while LoRA weights are trained. The exact low-level implementation is MLX-LM rather than the common CUDA and bitsandbytes stack.

### Why not full fine-tune the model?

Full fine-tuning requires memory for model weights, gradients, optimizer states, and activations. That is not comfortable for a 4B model on 24 GB unified memory. LoRA focuses training capacity on the required behaviour while fitting the actual hardware budget.

### What hyperparameters did you start with?

The project uses batch size 1, 1,024 maximum sequence length, 16 adapted layers, LoRA rank 8, learning rate 1e-5, and a 4-bit Qwen3 base. Those are conservative starting values. The prior 600-step run was a smoke-scale run; the larger current dataset needs about 2,500 iterations for a first approximate epoch.

### How do you query the fine-tuned model?

I load the original local MLX model and attach the adapter at inference time:

~~~zsh
.venv/bin/mlx_lm.generate \
  --model configs \
  --adapter-path adapters/qwen3-4b-physics-v1 \
  --prompt "Explain why an object in circular motion accelerates." \
  --max-tokens 300 --temp 0
~~~

Without the adapter path, that command queries only the original base model.

### Why did you not package the model for Ollama first?

It is unnecessary for evaluation because MLX-LM can query the base and adapter together directly. Also, the installed MLX-LM version cannot export Qwen3 to GGUF; its built-in GGUF export supports other model families. An MLX LoRA adapter should not be assumed compatible with Ollama's adapter format.

## Data strategy

### Where does the training data come from?

The main corpus is original synthetic SFT data generated locally from formula-checked numerical templates, conceptual explanation templates, persona-boundary responses, and safety responses. It does not copy textbook prose or depend on a retrieval system.

### How large is the current dataset?

Physics SFT v1 contains 3,000 generated raw records. After combining it with the earlier raw examples and making deterministic disjoint splits, the processed corpus contains 2,459 training records, 273 validation records, and 304 held-out test records.

### What physics topics are represented?

The curriculum includes mechanics, electromagnetism, thermodynamics, waves and optics, quantum/nuclear/relativity topics, conceptual misconceptions, units and assumptions, the fictional-persona boundary, and hazardous-request refusal behaviour.

### How did you avoid leaking test data into training?

The compiler creates disjoint train, validation, and test files from raw records using a deterministic seed. The test file is explicitly marked as evaluation-only and is never passed to the MLX-LM trainer. The compiler also removes exact duplicate user prompts.

### Why use synthetic examples rather than scraping every available physics site?

Public accessibility does not automatically grant training rights, and mixed web material varies greatly in accuracy and style. Original examples make licensing and provenance simpler. The trade-off is that synthetic data can become repetitive or encode mistakes, so it needs subject-matter review and diversity checks.

### What are the main dataset limitations?

The numerical questions are formula-heavy, and many conceptual answers are template-based. That can over-represent short textbook-style answers and under-represent long proofs, experimental design, advanced derivations, and ambiguous real-world questions. The corpus is a serious starter, not a replacement for expert-authored material.

## Quality and evaluation

### How do you validate the data before training?

The project validates every JSONL record for chat-message structure, role order, non-empty content, an assistant final response, and the exact canonical system persona. It has automated tests for validation, deterministic splitting, duplicate prevention, and the generated corpus.

### Why is training loss not enough?

Loss measures how well the model predicts the training-style text, not whether it gives physically correct or useful answers. A model can show lower loss while becoming more confident, more verbose, or more wrong. I compare base and adapted models on held-out prompts using rubrics for principle choice, equations, units, assumptions, uncertainty, safety, and persona adherence.

### What did you test locally?

I confirmed that MLX-LM can load the local Qwen3 4-bit checkpoint and apply the trained adapter. A direct adapter inference produced a correct explanation of centripetal acceleration at roughly 38 tokens per second with about 2.4 GB peak memory.

### What would success look like?

The adapter should improve a representative held-out set without making baseline answers less correct. I would retain it only after manual review finds better conceptual explanations, correct calculations and units, clearer assumptions, and consistent persona behaviour on most test prompts.

### What failure modes do you expect?

Expected failure modes include hallucinated derivations, arithmetic mistakes, incorrect applicability of a formula, overconfident answers, repeated phrasing from synthetic templates, and treating the persona as an identity rather than an instructional style.

## Engineering decisions

### How is the project reproducible?

The repository records the local model path, LoRA configuration, deterministic dataset seed, data compiler, validation scripts, adapter configuration, and evaluation prompts. The corpus generator writes a manifest with category counts and its seed.

### Why keep raw and processed data separately?

Raw data is the editable source material. Processed data is a generated, deterministic train/validation/test split. Separating them lets me revise source examples, regenerate the split, and avoid accidentally editing the evaluation set by hand.

### How do you handle the manually downloaded model?

The checkpoint is stored in the local configs directory and the training configuration uses the relative local path configs. I verified MLX-LM loads it locally, so training does not need to contact Hugging Face.

### What would you improve next?

I would perform physics-expert review on a stratified sample, add more multi-step derivations and qualitative reasoning, build a stronger independent benchmark, run controlled hyperparameter comparisons, inspect loss curves for overfitting, and document model-card limitations. If retrieval were later allowed, I would treat it as a separate design decision rather than silently mixing it into the fine-tuning experiment.

### What would you say about production readiness?

This is an educational prototype, not a production or safety-critical physics authority. It has useful reproducibility and validation scaffolding, but it still needs expert dataset review, broader evaluation, monitoring, and clear user-facing limits before any consequential use.
