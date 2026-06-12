# Fusion Scripts

This project keeps two fusion families for BUSI evaluation. Both families reuse
existing model predictions and do not retrain classifiers.

## 1. fixed_weight_platt_fusion

Fixed-weight Platt fusion uses the baseline 4-view weights:

- normal: full 0.50, cut_borders 0.35, border 0.10, masked 0.05
- fallback: full 0.70, cut_borders 0.20, border 0.05, masked 0.05

For 2-view and 3-view ablations, weights are normalized over the available
views. Platt calibration is fitted on OOF predictions, and the final threshold
is fixed at 0.5. BUSI is used only for final evaluation.

Main scripts:

- `calibrate_threshold.py`: fits the baseline 4-view Platt calibrator on OOF.
- `evaluate_ensemble.py`: evaluates the baseline 4-view fixed-weight Platt fusion.
- `evaluate_fixed_weight_platt_fusion.py`: canonical entry point for 2-view and 3-view fixed-weight Platt fusion.
- `evaluate_original_style_fusions.py`: historical compatibility entry point for the same fixed-weight Platt fusion logic.
- `organize_fusion_results.py`: copies metrics into the standardized folder and redraws standardized ROC plots.

Standard output folder:

```text
outputs/results/competition/fixed_weight_platt_fusion/
```

## 2. oof_search_weight_platt_fusion

OOF-search weight Platt fusion selects fusion weights and thresholds on OOF
predictions. Platt calibration is also fitted on OOF predictions. BUSI is used
only for final evaluation after OOF decisions are fixed.

Main scripts:

- `evaluate_2view_fusion.py`: searches 2-view pair/weights/threshold on OOF.
- `evaluate_3view_fusion.py`: searches 3-view weights/threshold on OOF.
- `evaluate_4view_fusion.py`: searches 4-view weights/threshold on OOF.
- `organize_fusion_results.py`: copies metrics into the standardized folder and redraws standardized ROC plots.

Standard output folder:

```text
outputs/results/competition/oof_search_weight_platt_fusion/
```

## Recommended reporting set

Use `fixed_weight_platt_fusion` as the primary comparison because 2-view,
3-view, and 4-view are evaluated with the same fusion rule.

Run order for standardized outputs:

```powershell
& 'D:\CondaEnv\LHG\python.exe' evaluate_fixed_weight_platt_fusion.py
& 'D:\CondaEnv\LHG\python.exe' evaluate_2view_fusion.py
& 'D:\CondaEnv\LHG\python.exe' evaluate_3view_fusion.py
& 'D:\CondaEnv\LHG\python.exe' evaluate_4view_fusion.py
& 'D:\CondaEnv\LHG\python.exe' organize_fusion_results.py
```

If only the standardized folders need to be regenerated from existing metrics,
run only:

```powershell
& 'D:\CondaEnv\LHG\python.exe' organize_fusion_results.py
```
