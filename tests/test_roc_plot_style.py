from pathlib import Path


def _source():
    return Path("organize_fusion_results.py").read_text(encoding="utf-8")


def test_roc_legend_is_readable_for_report_exports():
    source = _source()
    assert "LEGEND_FONT_SIZE = 18" in source
    assert "LEGEND_MARKER_SCALE = 1.8" in source
    assert "markerscale=LEGEND_MARKER_SCALE" in source
