# experiments/README.md

This directory holds per-run experiment outputs.  
Each run lives under its own subdirectory, created automatically by `ExperimentLogger`.

```
experiments/
  teacher_baseline/
    config.yaml          ← exact config used for this run
    run_info.json        ← git hash, seed, timestamps, hardware
    metrics.csv          ← one row per logged epoch/step (tracked in git)
    metrics.json         ← same data, JSON format (gitignored, large runs)
    run.log              ← full Python log (gitignored)
    checkpoints/
      best_val_f1/       ← model weights + tokenizer (gitignored)
      epoch_5/
  student_distillation/
    ...
```

> **Git policy**: model weights and raw logs are gitignored.  
> `config.yaml`, `run_info.json`, and `metrics.csv` **are** tracked so the full result table is reproducible from git history alone.
