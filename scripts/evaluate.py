"""
Script to generate evaluation report from registry.
Usage: python scripts/evaluate.py
       python scripts/evaluate.py --html
       python scripts/evaluate.py --leaderboard
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mlops.reporter import EvaluationReporter
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BWF model evaluation report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--leaderboard", action="store_true", help="Print leaderboard to console")
    parser.add_argument("--all", action="store_true", help="Generate all reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reporter = EvaluationReporter()

    if args.leaderboard or args.all:
        reporter.print_leaderboard()

    if args.html or args.all:
        path = reporter.generate_html_report()
        logger.success(f"HTML report: {path}")
    else:
        df = reporter.generate_summary_report()
        if not df.empty:
            logger.info(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
