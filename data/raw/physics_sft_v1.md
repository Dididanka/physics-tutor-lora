# Physics SFT v1

Physics SFT v1 is the pure supervised-fine-tuning curriculum for this project. It does not create or use a retrieval system.

The generated JSONL contains 3,000 original synthetic instruction/answer records:

- 660 mechanics records
- 600 electromagnetism records
- 450 thermodynamics records
- 390 waves and optics records
- 390 quantum, nuclear, and relativity records
- 390 conceptual and misconception records
- 120 persona-boundary and safety records

The source generator is scripts/build_physics_sft_corpus.py. It uses a fixed seed, formula-checked calculations, and original explanatory templates. The companion manifest records the seed and category targets.

The raw corpus is compiled with all existing raw examples into the processed training set. Current splits contain 2,459 training records, 273 validation records, and 304 never-train-on test records.

This is a substantial synthetic training draft, not a substitute for expert review. Before treating the adapted model as a physics authority, sample every category, check numerical and conceptual answers, and add independently reviewed examples where the model is weak.
