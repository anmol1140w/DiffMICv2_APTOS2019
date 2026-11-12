# DiffMIC trajectory research pipeline (APTOS 2019)

This supplement integrates an APTOS manifest, a reverse-loop capture hook in `pipeline.SR3Sampler`, a resumable feature cache, downstream controls, an explicit classical evidence gate, an optional gate-controlled state-vector QRC, and trust/XAI utilities. It does not overwrite the original DiffMIC model or historical result claims.

## Verified audit and limits of this checkout

- The repository is DiffMIC-v2. `ConditionalModel.forward` predicts noise for the noised class map; `SR3Sampler.sample_high_res` iterates the DDIM scheduler. Its evolving tensor is shaped `[B, classes, N, N]`, with `N` supplied by the DCG patch-attention count (the model's learned conditioning weights are sized for seven positions). This is a class-score diffusion map, not an image latent.
- The current original APTOS loader reads pickle records and uses 224x224 RGB, center-square crop, ImageNet normalization, horizontal/vertical flips and ±30-degree rotation for training. The new manifest loader preserves those conventions, deterministic validation/test transforms, and carries IDs and split names.
- `diffuser_trainer.py` was hard-coded to placental YAML, CUDA device 6, and the placental auxiliary checkpoint. It now accepts a config/resume/output CLI, uses the APTOS split manifest when selected, and no longer assumes GPU 6. Its original frozen-auxiliary DiffMIC-v2 loss path is retained.
- No APTOS CSV or images, DiffMIC checkpoint, auxiliary DCG checkpoint, EfficientSAM checkout, run logs, or baseline artifacts were found in this workspace. The PDF reports historical local APTOS reproduction claims, but this checkout cannot verify them; no result is described here as reproduced.
- Runtime available in this execution host is Python 3.14 with Pillow and PyYAML only; PyTorch, torchvision, NumPy, scikit-learn, Lightning, and pytest are absent. No CUDA tooling/device is available. Consequently model construction, real image tensor loading, training, checkpoint compatibility, trajectory extraction, ML experiments, and simulator execution cannot run here.
- Stage 1/2 evidence in the paper describes v1/v2 training and historical outcomes. This checkout contains only v2 source, not v1, checkpoints, split identifiers, logs, or its claimed outputs. Exact prior Stage 1/2 execution cannot be reconstructed or claimed from this evidence.

## Dataset and splits

Expected APTOS source: `train.csv` with `id_code,diagnosis` columns and matching image files under `train_images/` (or provide `--csv` and `--images`). Grades 0–4 are retained unchanged. Identifier matching is extension-insensitive and case-insensitive. Missing files, duplicate annotation IDs, ambiguous image matches, invalid labels, absent CSV, empty data, and corrupt images fail loudly. No dataset fallback exists.

```bash
python3 -m aptos_research.cli validate-data --root /path/to/aptos --output outputs/aptos_splits.json --seed 2026
```

The stratified split manifest stores exact IDs, file paths, labels, seed, per-class counts, and an identifier digest. Defaults are 70/15/15; fractions are configurable. Use the same manifest for training and feature extraction. For DiffMIC training, supply a compatible pretrained DCG checkpoint:

```bash
DIFFMIC_AUX_CHECKPOINT=/path/to/dcg.pth python3 diffuser_trainer.py --config configs/aptos.yml --output-dir logs --run-name aptos-v2
```

The training command retains the existing v2 architecture/objective and frozen auxiliary path. The dataset is small relative to the model, and the PDF reports late-epoch overfitting; inspect validation-selected checkpoints rather than assuming the final epoch is best. A compatible EfficientSAM dependency and checkpoints must be present before starting training.

## Trajectory and feature experiments

`SR3Sampler.sample_high_res(..., trajectory_callback=...)` captures post-step states only when requested, so normal predictions follow the same path. The extractor deterministically sets the RNG seeds, requests 100 reverse steps by default, samples up to 16 evenly spaced states in reverse-loop order, stores original diffusion timesteps and state indices, then extracts 1/4/8/16 states downstream. Default spatial flattening produces a compact 5x7x7=245-dimensional state when the model's expected shape is confirmed; `--pooling gap` yields five class scores and should be treated as a separate representation. This choice is explicit because class-map score states and semantic embeddings are not interchangeable.

The extractor uses a model/data adapter contract. The provided `diffmic_adapter` builds the actual repository ConditionalModel and DCG, requires a train/validation/test manifest and separately supplied compatible denoiser and auxiliary checkpoints, and validates checkpoint tensor shapes before inference.

```bash
export APTOS_ROOT=/path/to/aptos
export APTOS_SPLITS=outputs/aptos_splits.json
export DIFFMIC_CHECKPOINT=/path/to/conditional_model.ckpt
export DIFFMIC_AUX_CHECKPOINT=/path/to/dcg.ckpt
python3 -m aptos_research.extract --factory aptos_research.diffmic_adapter:factory --checkpoint "$DIFFMIC_CHECKPOINT" --cache outputs/trajectories.sqlite --seed 2026 --inference-steps 100
python3 -m aptos_research.experiments --cache outputs/trajectories.sqlite --output outputs/experiments/classical.json
```

The SQLite cache is per-sample and per-split, supports resume without duplicate IDs, stores each vector/timestep plus label, extraction metadata, seed, and checkpoint hash, and can be checked using:

```bash
python3 -m aptos_research.cli inspect-cache outputs/trajectories.sqlite
```

Static controls include final state, mean-pooled state sets, and validation-selected best single timestep. ESN controls preserve the exact same states with fixed cyclic order permutation. Logistic-regression readouts and the ESN settings/parameter counts/runtime are recorded. An optional GRU comparison uses identical K=16 states, training-only feature scaling, gradient clipping, validation-loss early stopping, and compares ordered and shuffled inputs:

```bash
python3 -m aptos_research.experiments --cache outputs/trajectories.sqlite --output outputs/experiments/classical.json --gru
```

S4 is not silently replaced: no compatible S4 implementation is included in the available environment.

## Gate and QRC

Before examining any held-out test conclusion, construct `gate_input.json` with `validation.labels`, `validation.ordered_predictions`, and `validation.control_predictions` (same validation IDs/order; include endpoint, validation-selected best single timestep, unordered mean, and shuffled ESN). Then run:

```bash
python3 -m aptos_research.cli gate --results gate_input.json --output outputs/experiments/gate.json --minimum-delta 0.01 --alpha 0.05 --bootstrap-samples 5000 --seed 2026
python3 -m aptos_research.run_qrc --gate outputs/experiments/gate.json --cache outputs/trajectories.sqlite --output outputs/experiments/qrc.json
```

The pre-registered rule compares ordered dynamics with the highest-accuracy named validation control using a paired sample bootstrap. PASS requires the 95% interval lower bound and point improvement both exceed 1 percentage point; FAIL means the upper bound is at/below that threshold; otherwise INCONCLUSIVE. The QRC runner writes `SKIPPED_GATE_NOT_PASS` unless the gate is PASS. The included NumPy state-vector reservoir is an engineering candidate only, with RY angle encoding, fixed ring CNOT layer, Pauli-Z readout, optional finite shots and simple readout depolarization. It is not quantum advantage evidence and lacks matched resource accounting against tuned classical alternatives.

## Trustworthiness, explainability, and evidence

`aptos_research.metrics` defines macro F1, one-vs-rest macro AUROC (mean over classes with positive and negative examples), ordinal MAE, quadratic weighted kappa, multiclass Brier score, 10-bin confidence ECE, and confusion matrix. `aptos_research.trust` fits Mahalanobis statistics from training representations only, creates risk-coverage points, and offers a simple brightness perturbation helper. `aptos_research.explain` plots per-grade probability evolution and exposes insertion/deletion curves only when supplied a valid model score and aligned saliency. The DCG saliency map may be used as its actual spatial map; no attention mechanism is invented for the denoiser. External-site generalization and clinical validity are not established by APTOS.

```bash
python3 -m aptos_research.trust_eval --cache outputs/trajectories.sqlite --results outputs/experiments/classical.json --model ordered_esn_k16
python3 -m aptos_research.xai_eval --cache outputs/trajectories.sqlite --sample-id <APTOS_ID> --split test
```

The trajectory plot applies spatial average pooling and softmax to the actual cached class-score maps, so it is explicitly a derived probability view. The trust command writes calibration/risk metrics, a reliability diagram, and train-fitted Mahalanobis distances. It does not report OOD AUROC because no independently labeled OOD cohort is available.

## Tests and experiment status

Run the dependency-light integrity/unit suite with:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q aptos_research tests
```

Real APTOS validation, model smoke tests, checkpoint compatibility, one training/validation step, feature extraction, downstream results, QRC, calibration plots, perturbation faithfulness, and resource measurements are blocked until data, checkpoints, dependencies, and compatible EfficientSAM are supplied. Do not call a synthetic unit test evidence of APTOS integration. Record every measured run in JSON/JSONL with its actual status; no metrics from the historical PDF have been copied into generated experiment results.
