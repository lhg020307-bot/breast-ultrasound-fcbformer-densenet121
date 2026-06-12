# FCBFormer + DenseNet121 Breast Ultrasound Classification

This repository provides the source code for a breast ultrasound analysis
pipeline that combines lesion segmentation, four-view image construction,
DenseNet121-based classification, threshold calibration, and multi-view fusion
evaluation.

The project is prepared for competition/research reproducibility. Source code is
hosted on GitHub, while trained model weights are distributed separately because
they are large binary files.

## Code and Model Availability

- Code repository:
  [https://github.com/lhg020307-bot/breast-ultrasound-fcbformer-densenet121](https://github.com/lhg020307-bot/breast-ultrasound-fcbformer-densenet121)
- Trained model weights:
  [Baidu Netdisk](https://pan.baidu.com/s/1LaTKuKrpvP3QYjHU4VIxTg?pwd=s2w8)
- Extraction code:
  `s2w8`

The GitHub repository contains code and lightweight documentation only. Datasets,
trained weights, pretrained checkpoints, generated masks, figures, metrics, and
other experiment outputs are not committed to Git.

## Method Overview

The pipeline contains five major stages:

1. Ultrasound image preprocessing.
2. Lesion segmentation with an FCBFormer-style model.
3. Four-view sample construction:
   `full`, `cut_borders`, `border`, and `masked`.
4. Five-fold DenseNet121 classification for each view.
5. Out-of-fold threshold calibration and multi-view fusion evaluation.

The fusion scripts support fixed-weight Platt fusion and OOF-search weight
Platt fusion. See [FUSION_SCRIPTS.md](FUSION_SCRIPTS.md) for details.

## Repository Layout

```text
.
|-- data/                         # Preprocessing, mask generation, and view construction
|-- models/                       # Segmentation model definition and utilities
|-- tests/                        # Pytest checks for core utilities and frontend helpers
|-- calibrate_threshold.py        # OOF threshold and Platt calibration
|-- evaluate_2view_fusion.py      # Two-view fusion evaluation
|-- evaluate_3view_fusion.py      # Three-view fusion evaluation
|-- evaluate_4view_fusion.py      # Four-view fusion evaluation
|-- evaluate_ensemble.py          # Fixed-weight four-view ensemble evaluation
|-- frontend_app.py               # Local single-image inference frontend
|-- organize_fusion_results.py    # Standardized result tables and ROC plots
|-- run_combined_pipeline.sh      # End-to-end training and evaluation script
|-- requirements.txt              # Python dependencies
`-- FUSION_SCRIPTS.md             # Fusion experiment notes
```

The following directories are intentionally kept lightweight in Git:

```text
images/       # Local datasets and generated view images
checkpoints/  # Local pretrained checkpoints
outputs/      # Local trained models, metrics, logs, and figures
```

## Environment

Python 3.10 or later is recommended.

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

For GPU training or inference, install a PyTorch build that matches the local
CUDA version before installing the remaining dependencies.

## Model Weights

Download the trained model package from Baidu Netdisk:

```text
Link: https://pan.baidu.com/s/1LaTKuKrpvP3QYjHU4VIxTg?pwd=s2w8
Extraction code: s2w8
```

The uploaded package is organized as:

```text
trained_models/
|-- segmentation/
|   `-- best.pt
|-- classification/
|   |-- cls_full_densenet121_fold0_best.pt
|   |-- ...
|   `-- cls_masked_densenet121_fold4_best.pt
|-- calibration/
|   `-- threshold_calibration.json
`-- FCBFormer_checkpoint.pt
```

After downloading, place the files under the local project as follows:

```text
checkpoints/pretrained/FCBFormer_checkpoint.pt
outputs/models/segmentation/best.pt
outputs/models/classification/cls_full_densenet121_fold0_best.pt
outputs/models/classification/cls_full_densenet121_fold1_best.pt
outputs/models/classification/cls_full_densenet121_fold2_best.pt
outputs/models/classification/cls_full_densenet121_fold3_best.pt
outputs/models/classification/cls_full_densenet121_fold4_best.pt
outputs/models/classification/cls_cut_borders_densenet121_fold0_best.pt
outputs/models/classification/cls_cut_borders_densenet121_fold1_best.pt
outputs/models/classification/cls_cut_borders_densenet121_fold2_best.pt
outputs/models/classification/cls_cut_borders_densenet121_fold3_best.pt
outputs/models/classification/cls_cut_borders_densenet121_fold4_best.pt
outputs/models/classification/cls_border_densenet121_fold0_best.pt
outputs/models/classification/cls_border_densenet121_fold1_best.pt
outputs/models/classification/cls_border_densenet121_fold2_best.pt
outputs/models/classification/cls_border_densenet121_fold3_best.pt
outputs/models/classification/cls_border_densenet121_fold4_best.pt
outputs/models/classification/cls_masked_densenet121_fold0_best.pt
outputs/models/classification/cls_masked_densenet121_fold1_best.pt
outputs/models/classification/cls_masked_densenet121_fold2_best.pt
outputs/models/classification/cls_masked_densenet121_fold3_best.pt
outputs/models/classification/cls_masked_densenet121_fold4_best.pt
outputs/results/competition/threshold_calibration.json
```

Only the `*_best.pt` classification checkpoints are required for the provided
evaluation and frontend inference scripts.

## Data Preparation

The dataset is not included in this repository. Prepare the local data according
to the following structure before training or full reproduction:

```text
images/full_image/
images/gt_masks/
```

The preprocessing and view-construction scripts will generate additional local
files under:

```text
images/preprocessed/
images/finetuned/
outputs/results/
```

Medical image data may be subject to privacy, license, or competition-specific
sharing rules. Please follow the relevant data-use policy when preparing or
redistributing datasets.

## Running the Full Pipeline

Run the complete training and evaluation workflow:

```bash
bash run_combined_pipeline.sh
```

Common overrides:

```bash
BACKBONE=densenet121 N_SPLITS=5 MAX_EPOCHS=30 SEG_EPOCHS=50 bash run_combined_pipeline.sh
```

For a short smoke run:

```bash
N_SPLITS=3 MAX_EPOCHS=2 SEG_EPOCHS=2 BATCH_SIZE=4 bash run_combined_pipeline.sh
```

## Fusion Evaluation

If trained models, OOF predictions, and calibration files are already available,
run:

```bash
python evaluate_fixed_weight_platt_fusion.py
python evaluate_2view_fusion.py
python evaluate_3view_fusion.py
python evaluate_4view_fusion.py
python organize_fusion_results.py
```

Standardized fusion results are written under:

```text
outputs/results/competition/
```

## Local Frontend Inference

Open the local GUI:

```bash
python frontend_app.py
```

Run one image in headless mode:

```bash
python frontend_app.py --image path/to/image.png
```

The frontend loads:

```text
outputs/models/segmentation/best.pt
outputs/models/classification/*_best.pt
outputs/results/competition/threshold_calibration.json
```

Generated reports and visual outputs are saved under `outputs/`.

## Tests

Run the test suite from the repository root:

```bash
pytest -q
```

Some frontend tests require a working Tk installation in the active Python
environment.

## Reproducibility Notes

- GitHub stores the reproducible code and documentation.
- Baidu Netdisk stores the trained model weights and calibration file.
- The dataset is not redistributed in this repository.
- Large generated artifacts are ignored by Git through `.gitignore`.
- To reproduce the submitted model outputs, place the downloaded weights in the
  paths listed above, prepare the dataset locally, and run the evaluation or
  full pipeline scripts.

## Citation / Paper Statement

When referencing this implementation in a report or paper, the following wording
can be used:

```text
The source code for model training, four-view construction, fusion evaluation,
and visualization is available at:
https://github.com/lhg020307-bot/breast-ultrasound-fcbformer-densenet121.
Trained model weights are provided separately via Baidu Netdisk:
https://pan.baidu.com/s/1LaTKuKrpvP3QYjHU4VIxTg?pwd=s2w8, extraction code s2w8.
```
