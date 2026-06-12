# FCBFormer + DenseNet121 Breast Ultrasound Pipeline

This repository contains a breast ultrasound analysis pipeline for segmentation,
four-view classification, out-of-fold calibration, and multi-view fusion
evaluation. The current implementation uses FCBFormer-style segmentation and
DenseNet121 classification backbones.

## Features

- Preprocess ultrasound images and masks.
- Fine-tune a segmentation model and generate lesion masks.
- Build four classification views: `full`, `cut_borders`, `border`, and
  `masked`.
- Train 5-fold DenseNet121 classifiers for each view.
- Calibrate thresholds on out-of-fold predictions.
- Evaluate fixed-weight and OOF-search fusion strategies.
- Run a single-image frontend for local inference and report export.

## Repository Layout

```text
.
|-- data/                         # Preprocessing, mask generation, and view building
|-- models/                       # Segmentation model utilities
|-- tests/                        # Existing pytest checks
|-- calibrate_threshold.py        # OOF threshold calibration
|-- evaluate_*fusion.py           # 2-view, 3-view, 4-view fusion evaluation scripts
|-- evaluate_ensemble.py          # Fixed-weight 4-view ensemble evaluation
|-- frontend_app.py               # Local single-image inference frontend
|-- organize_fusion_results.py    # Standardizes result folders and ROC plots
|-- run_combined_pipeline.sh      # End-to-end training/evaluation pipeline
|-- FUSION_SCRIPTS.md             # Detailed fusion-script notes
`-- requirements.txt              # Python dependencies
```

Large runtime artifacts are intentionally excluded from Git:

- `images/`: local datasets, preprocessed images, generated masks, and views.
- `checkpoints/`: pretrained weights and trained model checkpoints.
- `outputs/`: metrics, figures, logs, reports, and intermediate predictions.

See the README files inside those directories for the expected local contents.

## Environment Setup

Use Python 3.10+ and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from
the official PyTorch installation guide before installing the rest of the
requirements.

## Data and Weights

Prepare the following local files before running the full pipeline:

```text
images/full_image/
images/gt_masks/
checkpoints/pretrained/FCBFormer_checkpoint.pt
```

The scripts also generate and consume these local outputs:

```text
images/preprocessed/
images/finetuned/
outputs/models/
outputs/results/
```

Do not upload private medical images, generated masks, trained weights, or large
experiment outputs to GitHub unless you have explicit permission and a clear
data-sharing policy.

## End-to-End Pipeline

The complete training and evaluation workflow is wrapped by:

```bash
bash run_combined_pipeline.sh
```

Useful environment overrides:

```bash
BACKBONE=densenet121 N_SPLITS=5 MAX_EPOCHS=30 SEG_EPOCHS=50 bash run_combined_pipeline.sh
```

For a quick smoke run on a smaller setup:

```bash
N_SPLITS=3 MAX_EPOCHS=2 SEG_EPOCHS=2 BATCH_SIZE=4 bash run_combined_pipeline.sh
```

## Fusion Evaluation

After the model predictions and OOF outputs exist, run the fusion scripts:

```bash
python evaluate_fixed_weight_platt_fusion.py
python evaluate_2view_fusion.py
python evaluate_3view_fusion.py
python evaluate_4view_fusion.py
python organize_fusion_results.py
```

For more details about the fixed-weight and OOF-search fusion families, see
`FUSION_SCRIPTS.md`.

## Frontend Inference

Open the local GUI:

```bash
python frontend_app.py
```

Run one image in headless mode:

```bash
python frontend_app.py --image path/to/image.png
```

The frontend writes local reports and generated images under `outputs/`.

## Tests

Run the existing test suite from the repository root:

```bash
pytest -q
```

Some GUI-related tests require a working Tk installation. If Tk is missing or
misconfigured in the active Python environment, run tests in an environment with
Tk support.

## GitHub Upload Checklist

Before creating the GitHub repository:

1. Review `.gitignore` and confirm `images/`, `outputs/`, and `checkpoints/`
   are not staged.
2. Remove private paths, usernames, and machine-specific notes from any files
   you plan to publish.
3. Confirm the repository contains code, docs, and lightweight metadata only.
4. Choose a license if this project will be shared publicly.
5. Run `git status --short` before committing.

Suggested first commit:

```bash
git init
git add .gitignore .gitattributes README.md requirements.txt FUSION_SCRIPTS.md data models tests *.py *.sh
git status --short
git commit -m "docs: prepare project for GitHub"
```
