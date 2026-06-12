"""Canonical entry point for fixed-weight Platt fusion evaluation.

This wrapper keeps the historical implementation file intact while exposing the
method name used in reports and standardized outputs.
"""

from __future__ import annotations

from evaluate_original_style_fusions import main


if __name__ == "__main__":
    main()
