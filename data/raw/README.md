# Synthetic physics-tutor starter corpus

This directory contains an original, synthetic dataset generated from general physics knowledge for this project. It is a starter corpus, not an authoritative textbook or a substitute for expert review.

- physics_tutor_train.jsonl contains 28 training examples.
- physics_tutor_valid.jsonl contains 8 held-out validation examples.
- Every record uses the project chat schema. The source persona is intentionally short; scripts/compile_dataset.py replaces it with the canonical persona before training.

The examples are broad enough to exercise the pipeline—mechanics, electromagnetism, thermodynamics, waves, optics, quantum physics, relativity, and safe tutoring behaviour—but are too small to establish real expertise. Before a serious run, have a physics-qualified reviewer check equations, conventions, level, and coverage; add more permissioned examples; and preserve a separate test set never shown to training.

Compile the corpus into the training-ready split with:

~~~zsh
python scripts/compile_dataset.py \
  --input data/raw \
  --output data/processed \
  --seed 42 \
  --valid-fraction 0.15 \
  --test-fraction 0.15
~~~

The provided physics_tutor_valid.jsonl is useful for source-level validation, while the compiler produces the project-wide train, valid, and never-train-on test splits.
