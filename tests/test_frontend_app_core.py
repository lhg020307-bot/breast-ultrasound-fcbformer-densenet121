import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import frontend_app
from frontend_app import (
    FrontendApp,
    ModelRunner,
    apply_platt,
    compute_mask_metrics,
    export_text_report,
    first_image_file_from_clipboard,
    find_doctor_mask_paths,
    generate_classification_views,
    image_metadata_from_path,
    load_doctor_mask,
    mask_quality_summary,
    model_explanation,
    overlay_prediction_with_doctor,
    restore_mask_to_original,
    save_pasted_image,
    view_probability_interpretation,
    weighted_fusion,
)


def test_compute_mask_metrics_matches_expected_overlap():
    pred = np.array(
        [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    target = np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )

    metrics = compute_mask_metrics(pred, target)

    assert math.isclose(metrics["iou"], 0.5, rel_tol=1e-6)
    assert math.isclose(metrics["dice"], 2 * 2 / (3 + 3), rel_tol=1e-6)


def test_find_and_merge_doctor_masks_from_dataset_image(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image_path = tmp_path / "Dataset_BUSI_with_GT" / "benign" / "case.png"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), image)
    mask_a = np.zeros((8, 8), dtype=np.uint8)
    mask_b = np.zeros((8, 8), dtype=np.uint8)
    mask_a[1:3, 1:3] = 255
    mask_b[5:7, 5:7] = 255
    cv2.imwrite(str(image_path.with_name("case_mask.png")), mask_a)
    cv2.imwrite(str(image_path.with_name("case_mask_1.png")), mask_b)

    paths = find_doctor_mask_paths(image_path)
    doctor_mask, found_paths = load_doctor_mask(image_path, target_shape=(8, 8))

    assert [path.name for path in paths] == ["case_mask.png", "case_mask_1.png"]
    assert [path.name for path in found_paths] == ["case_mask.png", "case_mask_1.png"]
    assert doctor_mask[1:3, 1:3].all()
    assert doctor_mask[5:7, 5:7].all()


def test_restore_prediction_mask_to_original_coordinates():
    pred = np.zeros((4, 4), dtype=np.uint8)
    pred[1:3, 1:3] = 1

    restored = restore_mask_to_original(pred, original_shape=(8, 8), crop_box=(2, 2, 6, 6))

    assert restored.shape == (8, 8)
    assert restored[3:5, 3:5].all()
    assert restored[:2].sum() == 0


def test_overlay_prediction_with_doctor_uses_distinct_colors():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    pred = np.zeros((4, 4), dtype=np.uint8)
    doctor = np.zeros((4, 4), dtype=np.uint8)
    pred[1, 1] = 1
    doctor[1, 2] = 1
    pred[2, 2] = 1
    doctor[2, 2] = 1

    overlay = overlay_prediction_with_doctor(image, pred, doctor, alpha=1.0)

    assert overlay[1, 1].tolist() == [0, 0, 255]
    assert overlay[1, 2].tolist() == [0, 255, 0]
    assert overlay[2, 2].tolist() == [0, 255, 255]


def test_weighted_fusion_normalizes_weights_and_applies_platt():
    probs = {"full": 0.8, "cut_borders": 0.6, "border": 0.2, "masked": 0.4}
    weights = {"full": 2.0, "cut_borders": 1.0, "border": 1.0, "masked": 0.0}

    raw = weighted_fusion(probs, weights)
    calibrated = apply_platt(raw, a=1.0, b=0.0)

    assert math.isclose(raw, (0.8 * 2 + 0.6 + 0.2) / 4, rel_tol=1e-6)
    assert math.isclose(calibrated, raw, rel_tol=1e-6)


def test_generate_classification_views_uses_mask_bbox_and_fallback():
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[2:10, 2:10] = (40, 80, 120)
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:8, 5:9] = 1

    views, quality = generate_classification_views(image, mask, margin=1)

    assert set(views) == {"full", "cut_borders", "border", "masked"}
    assert views["full"].shape == image.shape
    assert views["cut_borders"].shape[:2] == (6, 6)
    assert quality["is_valid"] is True
    assert int(cv2.cvtColor(views["masked"], cv2.COLOR_BGR2GRAY).sum()) > 0

    fallback_views, fallback_quality = generate_classification_views(
        image,
        np.zeros((12, 12), dtype=np.uint8),
        margin=1,
    )
    assert fallback_quality["is_valid"] is False
    assert fallback_views["cut_borders"].shape[0] < image.shape[0]


def test_runner_uses_uploaded_mask_without_calling_segmentation(tmp_path, capsys):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[4:20, 4:20] = (70, 120, 180)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    image_path = tmp_path / "case.png"
    mask_path = tmp_path / "case_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    class FakeRunner(ModelRunner):
        def predict_mask(self, image_bgr):  # pragma: no cover - should not run
            raise AssertionError("segmentation should be skipped when a mask is uploaded")

        def predict_view_probability(self, view, image_bgr):
            return {
                "full": 0.8,
                "cut_borders": 0.7,
                "border": 0.3,
                "masked": 0.6,
            }[view]

    result = FakeRunner(project_root=tmp_path).run(image_path, mask_path)

    assert result.mask_source == "uploaded"
    assert result.metrics is not None
    assert math.isclose(result.metrics["iou"], 1.0)
    assert math.isclose(result.metrics["dice"], 1.0)
    assert result.view_probabilities["full"] == 0.8
    assert (tmp_path / "outputs" / "frontend").exists()
    output = capsys.readouterr().out
    assert "IoU:" in output
    assert "Dice:" in output
    assert "结果输出文件夹:" in output


def test_runner_uses_uploaded_mask_without_enhancement(tmp_path, monkeypatch):
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    image_path = tmp_path / "case.png"
    mask_path = tmp_path / "case_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)

    def fail_preprocess(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("enhancement should be skipped when a mask is uploaded")

    monkeypatch.setattr(frontend_app, "preprocess_ultrasound", fail_preprocess)

    class FakeRunner(ModelRunner):
        def predict_mask(self, image_bgr):  # pragma: no cover - should not run
            raise AssertionError("segmentation should be skipped when a mask is uploaded")

        def predict_view_probability(self, view, image_bgr):
            return 0.5

    result = FakeRunner(project_root=tmp_path).run(image_path, mask_path)

    assert result.mask_source == "uploaded"


def test_runner_retrieves_doctor_mask_and_outputs_metrics(tmp_path, monkeypatch, capsys):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[1:7, 1:7] = (80, 120, 160)
    doctor = np.zeros((8, 8), dtype=np.uint8)
    doctor[2:6, 2:6] = 255
    image_path = tmp_path / "Dataset_BUSI_with_GT" / "benign" / "case.png"
    image_path.parent.mkdir(parents=True)
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(image_path.with_name("case_mask.png")), doctor)

    monkeypatch.setattr(frontend_app, "preprocess_ultrasound", lambda img: (img, (0, 0, 8, 8)))

    class FakeRunner(ModelRunner):
        def predict_mask(self, image_bgr):
            pred = np.zeros((8, 8), dtype=np.uint8)
            pred[2:6, 2:6] = 1
            return pred.astype(np.float32), pred

        def predict_view_probability(self, view, image_bgr):
            return 0.5

    result = FakeRunner(project_root=tmp_path).run(image_path)

    assert result.metrics is not None
    assert math.isclose(result.metrics["iou"], 1.0)
    assert math.isclose(result.metrics["dice"], 1.0)
    assert Path(result.saved_files["original"]).exists()
    assert Path(result.saved_files["doctor_mask"]).exists()
    assert Path(result.saved_files["predicted_mask"]).exists()
    assert Path(result.saved_files["comparison_overlay"]).exists()
    output = capsys.readouterr().out
    assert "IoU:" in output
    assert "Dice:" in output


def test_save_pasted_image_writes_clipboard_png(tmp_path):
    image = Image.new("RGB", (5, 4), (120, 30, 200))

    saved_path = save_pasted_image(image, tmp_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".png"
    loaded = cv2.imread(str(saved_path), cv2.IMREAD_COLOR)
    assert loaded.shape[:2] == (4, 5)


def test_first_image_file_from_clipboard_ignores_non_images(tmp_path):
    text_path = tmp_path / "note.txt"
    text_path.write_text("not image", encoding="utf-8")
    image_path = tmp_path / "case.jpeg"
    cv2.imwrite(str(image_path), np.zeros((3, 3, 3), dtype=np.uint8))

    selected = first_image_file_from_clipboard([str(text_path), str(image_path)])

    assert selected == image_path


def test_image_metadata_from_path_extracts_dataset_and_label():
    path = r"D:\data\BUS-UCLM\malignant\FUHI_003.png"

    metadata = image_metadata_from_path(path)

    assert metadata["filename"] == "FUHI_003.png"
    assert metadata["dataset"] == "BUS-UCLM"
    assert metadata["label"] == "malignant"


def test_view_probability_interpretation_uses_expected_ranges():
    assert view_probability_interpretation("full", 0.90) == "完整图像高度支持恶性"
    assert view_probability_interpretation("masked", 0.70) == "Mask 区域倾向恶性"
    assert view_probability_interpretation("border", 0.50) == "边界视图支持度一般"
    assert view_probability_interpretation("cut_borders", 0.20) == "裁剪视图不支持恶性"


def test_mask_quality_summary_formats_status_and_missing_metrics():
    summary = mask_quality_summary(
        {"is_valid": True, "area_ratio": 0.2127, "num_components": 1, "reason": "ok"},
        None,
    )

    assert summary["status"] == "通过"
    assert summary["area_ratio_text"] == "21.27%"
    assert summary["iou_text"] == "未计算"
    assert summary["dice_text"] == "未计算"


def test_model_explanation_mentions_high_and_low_views():
    explanation = model_explanation(
        {"full": 0.96, "cut_borders": 0.91, "border": 0.32, "masked": 0.88},
        "恶性",
    )

    assert "full、cut_borders、masked 视图高度支持恶性判断" in explanation
    assert "border 视图支持度相对较低" in explanation
    assert "融合结果仍明显倾向恶性" in explanation


def test_export_text_report_writes_summary(tmp_path):
    class Result:
        work_id = "5268"
        image_path = str(tmp_path / "FUHI_003.png")
        mask_source = "uploaded"
        output_dir = str(tmp_path)
        malignant_probability = 0.9374
        benign_probability = 0.0626
        predicted_label = "恶性"
        threshold = 0.4243
        raw_fusion_probability = 0.9769
        view_probabilities = {"full": 0.9, "cut_borders": 0.8, "border": 0.5, "masked": 0.95}
        mask_quality = {"is_valid": True, "area_ratio": 0.12, "num_components": 1, "reason": "ok"}
        metrics = None
        saved_files = {}

    report_path = export_text_report(Result())

    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "作品 ID: 5268" in text
    assert "最终预测: 恶性" in text
    assert "四视图恶性概率" in text
    assert "数据集来源" not in text
    assert "数据集: 未知" not in text


def test_export_text_report_writes_known_dataset_source(tmp_path):
    class Result:
        work_id = "5268"
        image_path = str(tmp_path / "Dataset_BUSI_with_GT" / "malignant" / "case.png")
        mask_source = "segmentation"
        output_dir = str(tmp_path)
        malignant_probability = 0.9374
        benign_probability = 0.0626
        predicted_label = "恶性"
        threshold = 0.4243
        raw_fusion_probability = 0.9769
        view_probabilities = {"full": 0.9, "cut_borders": 0.8, "border": 0.5, "masked": 0.95}
        mask_quality = {"is_valid": True, "area_ratio": 0.12, "num_components": 1, "reason": "ok"}
        metrics = None
        saved_files = {}

    report_path = export_text_report(Result())

    text = report_path.read_text(encoding="utf-8")
    assert "数据集来源: Dataset_BUSI_with_GT" in text


def test_frontend_shows_dataset_source_only_when_known():
    app = FrontendApp()
    try:
        app.root.update_idletasks()
        assert not app.dataset_source_name_label.grid_info()
        assert not app.dataset_source_value_label.grid_info()

        app.image_path.set(r"D:\data\Dataset_BUSI_with_GT\malignant\case.png")
        app._update_file_info()
        app.root.update_idletasks()

        assert app.dataset_source_name_label.grid_info()
        assert app.dataset_source_value_label.grid_info()
        assert app.file_info_vars["dataset"].get() == "Dataset_BUSI_with_GT"

        app.image_path.set(r"D:\case.png")
        app._update_file_info()
        app.root.update_idletasks()

        assert not app.dataset_source_name_label.grid_info()
        assert not app.dataset_source_value_label.grid_info()
    finally:
        app.root.destroy()


def test_result_probability_bar_keeps_latest_value_after_redraw(tmp_path):
    class Result:
        predicted_label = "恶性"
        malignant_probability = 0.6742
        benign_probability = 0.3258
        threshold = 0.4243
        raw_fusion_probability = 0.8021

    app = FrontendApp()
    try:
        app._update_result_card(Result())
        app._draw_probability_bar()
        texts = [
            app.prob_canvas.itemcget(item, "text")
            for item in app.prob_canvas.find_all()
            if app.prob_canvas.type(item) == "text"
        ]
        assert "67.42%" in texts
    finally:
        app.root.destroy()


def test_show_result_warns_when_reference_label_conflicts(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    paths = {}
    for name in ["preprocessed", "mask", "overlay", "probability"]:
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), image)
        paths[name] = str(path)
    paths["result_json"] = str(tmp_path / "result.json")

    class Result:
        work_id = "5268"
        image_path = str(tmp_path / "Augmented-OASBUD" / "benign" / "case.png")
        mask_source = "segmentation"
        output_dir = str(tmp_path)
        malignant_probability = 0.7
        benign_probability = 0.3
        predicted_label = "恶性"
        threshold = 0.4243
        raw_fusion_probability = 0.65
        view_probabilities = {"full": 0.7, "cut_borders": 0.7, "border": 0.7, "masked": 0.7}
        mask_quality = {"is_valid": True, "area_ratio": 0.1, "num_components": 1, "reason": "ok"}
        metrics = None
        saved_files = paths

    app = FrontendApp()
    try:
        app._show_result(Result())
        app.root.update_idletasks()

        assert "文件标注为良性" in app.status_text.get()
        assert "模型预测为恶性" in app.status_text.get()
    finally:
        app.root.destroy()


def test_export_report_includes_reference_conflict_warning(tmp_path):
    class Result:
        work_id = "5268"
        image_path = str(tmp_path / "Augmented-OASBUD" / "benign" / "case.png")
        mask_source = "segmentation"
        output_dir = str(tmp_path)
        malignant_probability = 0.7
        benign_probability = 0.3
        predicted_label = "恶性"
        threshold = 0.4243
        raw_fusion_probability = 0.65
        view_probabilities = {"full": 0.7, "cut_borders": 0.7, "border": 0.7, "masked": 0.7}
        mask_quality = {"is_valid": True, "area_ratio": 0.1, "num_components": 1, "reason": "ok"}
        metrics = None
        saved_files = {}

    report_path = export_text_report(Result())

    text = report_path.read_text(encoding="utf-8")
    assert "文件标注: 良性" in text
    assert "文件标注为良性，模型预测为恶性" in text


def test_show_result_uses_original_doctor_prediction_and_comparison_panels(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    paths = {}
    for name in ["original", "preprocessed", "doctor_mask", "mask", "predicted_mask", "overlay", "comparison_overlay"]:
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), image)
        paths[name] = str(path)
    paths["probability"] = str(tmp_path / "probability.png")
    paths["result_json"] = str(tmp_path / "result.json")

    class Result:
        work_id = "5268"
        image_path = str(tmp_path / "case.png")
        mask_source = "segmentation"
        output_dir = str(tmp_path)
        malignant_probability = 0.7
        benign_probability = 0.3
        predicted_label = "恶性"
        threshold = 0.4243
        raw_fusion_probability = 0.65
        view_probabilities = {"full": 0.7, "cut_borders": 0.7, "border": 0.7, "masked": 0.7}
        mask_quality = {"is_valid": True, "area_ratio": 0.1, "num_components": 1, "reason": "ok"}
        metrics = None
        saved_files = paths

    app = FrontendApp()
    try:
        app._show_result(Result())
        app.root.update_idletasks()

        assert app.image_labels["input"].cget("text") == ""
        assert app.image_labels["doctor_mask"].cget("text") == ""
        assert app.image_labels["mask"].cget("text") == ""
        assert app.image_labels["overlay"].cget("text") == ""
    finally:
        app.root.destroy()


def test_frontend_right_panel_has_no_scrollbar():
    app = FrontendApp()
    try:
        app.root.update_idletasks()
        widgets = [app.root]
        for widget in widgets:
            widgets.extend(widget.winfo_children())

        assert not any(widget.winfo_class() == "TScrollbar" for widget in widgets)
    finally:
        app.root.destroy()


def test_frontend_only_accepts_original_image_inputs():
    app = FrontendApp()
    try:
        app.root.update_idletasks()
        widgets = [app.root]
        for widget in widgets:
            widgets.extend(widget.winfo_children())

        button_texts = {
            widget.cget("text")
            for widget in widgets
            if widget.winfo_class() == "Button"
        }
        assert "选择图片" in button_texts
        assert "粘贴图片" in button_texts
        assert "选择 Mask" not in button_texts
        assert "清除 Mask" not in button_texts
    finally:
        app.root.destroy()


def test_frontend_mask_card_shows_iou_and_dice_only_when_computed():
    app = FrontendApp()
    try:
        app.root.update_idletasks()
        assert not any(widget.grid_info() for widget in app.metric_widgets)

        class Result:
            mask_quality = {"is_valid": True, "area_ratio": 0.1, "num_components": 1, "reason": "ok"}
            metrics = {"iou": 0.5, "dice": 0.6667}

        app._update_mask_quality(Result())
        app.root.update_idletasks()

        assert all(widget.grid_info() for widget in app.metric_widgets)
        assert app.mask_vars["iou"].get() == "0.5000"
        assert app.mask_vars["dice"].get() == "0.6667"
    finally:
        app.root.destroy()
