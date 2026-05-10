"""
Automatic evaluation report generator.
Produces a detailed HTML + CSV report comparing all registered models.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.config import settings
from src.models.registry import ModelRegistry
from src.utils.logger import logger

REPORT_DIR = Path("reports")


class EvaluationReporter:
    """
    Generates evaluation reports from the model registry.

    Usage:
        reporter = EvaluationReporter()
        reporter.generate_summary_report()
        reporter.generate_html_report()
    """

    def __init__(self) -> None:
        self.registry = ModelRegistry()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def generate_summary_report(self) -> pd.DataFrame:
        """
        Create a CSV summary of all model metrics across draws and regions.
        """
        all_models = self.registry.list_models()
        if not all_models:
            logger.warning("No models in registry yet. Train models first.")
            return pd.DataFrame()

        rows = []
        for entry in all_models:
            row = {
                "draw": entry["draw"],
                "region": entry["region"],
                "version": entry["version"],
                "stage": entry["stage"],
                "mae_cv": entry["metrics"].get("mae_cv", None),
                "rmse_cv": entry["metrics"].get("rmse_cv", None),
                "r2_cv": entry["metrics"].get("r2_cv", None),
                "saved_at": entry["saved_at"],
            }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values(["draw", "region", "mae_cv"])
        report_path = REPORT_DIR / f"model_summary_{_today()}.csv"
        df.to_csv(report_path, index=False)
        logger.success(f"Summary report saved: {report_path}")
        return df

    def generate_html_report(self) -> Path:
        """
        Generate a styled HTML report with metrics table and charts.
        """
        df = self.generate_summary_report()
        if df.empty:
            return REPORT_DIR / "empty_report.html"

        # Build HTML
        html = _build_html_report(df)
        report_path = REPORT_DIR / f"evaluation_report_{_today()}.html"
        report_path.write_text(html, encoding="utf-8")
        logger.success(f"HTML report saved: {report_path}")
        return report_path

    def print_leaderboard(self) -> None:
        """Print a ranked leaderboard to console."""
        all_models = self.registry.list_models()
        if not all_models:
            print("No models trained yet.")
            return

        production = [m for m in all_models if m.get("stage") == "Production"]
        staging = [m for m in all_models if m.get("stage") == "Staging"]

        print("\n" + "="*70)
        print("  [LEADERBOARD] BWF RANKING PREDICTION")
        print("="*70)

        print("\n  [PRODUCTION MODELS]")
        print(f"  {'Draw':<6} {'Region':<10} {'Version':<10} {'MAE':>8} {'R²':>8}")
        print(f"  {'-'*50}")
        for m in sorted(production, key=lambda x: x["metrics"].get("mae_cv", 999)):
            mae = m["metrics"].get("mae_cv", "N/A")
            r2  = m["metrics"].get("r2_cv", "N/A")
            mae_str = f"{mae:.1f}" if isinstance(mae, float) else mae
            r2_str  = f"{r2:.4f}" if isinstance(r2, float) else r2
            print(f"  {m['draw']:<6} {m['region']:<10} {m['version']:<10} {mae_str:>8} {r2_str:>8}")

        print(f"\n  [STAGING MODELS] ({len(staging)} total)")
        print("="*70 + "\n")


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _build_html_report(df: pd.DataFrame) -> str:
    """Build a clean styled HTML report."""
    table_html = df.to_html(index=False, classes="metrics-table", border=0, float_format="%.4f")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>BWF Ranking Prediction — Evaluation Report</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
    h1 {{ color: #38bdf8; border-bottom: 2px solid #38bdf8; padding-bottom: 0.5rem; }}
    h2 {{ color: #7dd3fc; margin-top: 2rem; }}
    .metrics-table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    .metrics-table th {{ background: #1e3a5f; color: #38bdf8; padding: 0.75rem 1rem; text-align: left; }}
    .metrics-table td {{ padding: 0.6rem 1rem; border-bottom: 1px solid #1e293b; }}
    .metrics-table tr:hover {{ background: #1e293b; }}
    .badge-production {{ background: #16a34a; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    .badge-staging {{ background: #d97706; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }}
    .footer {{ margin-top: 3rem; color: #64748b; font-size: 0.85em; }}
  </style>
</head>
<body>
  <h1>🏸 BWF Ranking Prediction — Evaluation Report</h1>
  <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>

  <h2>📊 All Models — Metrics Summary</h2>
  {table_html}

  <h2>📈 Metric Definitions</h2>
  <ul>
    <li><strong>MAE</strong>: Mean Absolute Error (lower = better). Average error in BWF ranking points.</li>
    <li><strong>RMSE</strong>: Root Mean Squared Error (lower = better). Penalizes large errors more.</li>
    <li><strong>R²</strong>: R-squared (higher = better). How much variance is explained (1.0 = perfect).</li>
  </ul>

  <div class="footer">
    <p>BWF Ranking Prediction | Phase 2 MLOps | github.com/WooCSSIN</p>
  </div>
</body>
</html>"""
