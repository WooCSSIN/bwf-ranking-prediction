"""
Phase 2 training CLI script with MLflow + Optuna support.
Replaces the Phase 1 train.py with full MLOps capabilities.

Usage:
    python scripts/train.py --draw MS --region Asia
    python scripts/train.py --draw MS --region Asia --tune --trials 50
    python scripts/train.py --draw MS --region Asia --compare-models
    python scripts/train.py --all
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.loader import DataLoader
from src.data.preprocessor import BWFPreprocessor
from src.data.feature_engine import FeatureEngineer
from src.models.trainer import BWFModelTrainer
from src.models.registry import ModelRegistry
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BWF Ranking Prediction model (Phase 2 — MLOps edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train MS/Asia with default params:
    python scripts/train.py --draw MS --region Asia

  Train with Optuna hyperparameter search:
    python scripts/train.py --draw MS --region Asia --tune --trials 50

  Compare LightGBM vs XGBoost vs GradientBoosting:
    python scripts/train.py --draw MS --region Asia --compare-models

  Train all 15 draw × region combinations:
    python scripts/train.py --all
        """,
    )
    parser.add_argument("--draw", choices=settings.DRAWS, default="MS")
    parser.add_argument("--region", choices=settings.REGIONS, default="Global")
    parser.add_argument(
        "--model-type",
        choices=["lightgbm", "xgboost", "gradient_boosting"],
        default="lightgbm",
    )
    parser.add_argument("--data-file", default="bwf_cleaned.csv",
                        help="CSV filename in data/raw/")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare LightGBM vs XGBoost vs GradientBoosting")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    parser.add_argument("--all", action="store_true",
                        help="Train all draw × region combinations")
    parser.add_argument("--promote", action="store_true",
                        help="Auto-promote best model to Production after training")
    return parser.parse_args()


def run_single(
    draw: str,
    region: str,
    data_file: str,
    model_type: str,
    tune: bool,
    trials: int,
    cv_splits: int,
    compare: bool,
    use_mlflow: bool,
    promote: bool,
) -> None:
    logger.info(f"{'='*60}")
    logger.info(f"  Draw={draw} | Region={region} | Model={model_type}")
    logger.info(f"{'='*60}")

    # ── Load & preprocess ──────────────────────────────────────────────────
    loader = DataLoader()
    raw_df = loader.load(data_file)

    preprocessor = BWFPreprocessor()
    clean_df = preprocessor.fit_transform(raw_df)

    # ── Feature engineering ────────────────────────────────────────────────
    fe = FeatureEngineer()
    features_df = fe.transform(clean_df, draw=draw, region=region)

    if features_df.empty:
        logger.warning(f"No data for draw={draw}, region={region} — skipping")
        return

    # ── Train ──────────────────────────────────────────────────────────────
    trainer = BWFModelTrainer(
        draw=draw,
        region=region,
        model_type=model_type,
        use_mlflow=use_mlflow,
        tune_params=tune,
        n_tune_trials=trials,
    )

    if compare:
        comparison_df = trainer.compare_models(
            features_df, fe.feature_columns, n_splits=cv_splits
        )
        logger.info(f"\n{comparison_df.to_string(index=False)}")

        # Save comparison report
        report_path = Path("reports") / f"model_comparison_{draw}_{region}.csv"
        report_path.parent.mkdir(exist_ok=True)
        comparison_df.to_csv(report_path, index=False)
        logger.info(f"Comparison saved: {report_path}")

    model, metrics = trainer.train(
        df=features_df,
        feature_cols=fe.feature_columns,
        target_col="points",
        n_splits=cv_splits,
    )
    model_path = trainer.save()

    # ── Registry ───────────────────────────────────────────────────────────
    registry = ModelRegistry()
    version = registry.save(
        model=model,
        feature_names=fe.feature_columns,
        draw=draw,
        region=region,
        metrics=metrics,
        mlflow_run_id=getattr(trainer, "_mlflow_run_id", None),
    )

    if promote:
        registry.promote(draw, region, version)

    logger.success(f"Done: {draw}/{region} — version={version}, MAE={metrics['mae_cv']:.1f}")


def main() -> None:
    args = parse_args()
    use_mlflow = not args.no_mlflow

    if args.all:
        logger.info(f"Training ALL {len(settings.DRAWS) * len(settings.REGIONS)} models...")
        for draw in settings.DRAWS:
            for region in settings.REGIONS:
                try:
                    run_single(
                        draw=draw, region=region,
                        data_file=args.data_file,
                        model_type=args.model_type,
                        tune=args.tune,
                        trials=args.trials,
                        cv_splits=args.cv_splits,
                        compare=args.compare_models,
                        use_mlflow=use_mlflow,
                        promote=args.promote,
                    )
                except Exception as e:
                    logger.error(f"Failed {draw}/{region}: {e}")
    else:
        run_single(
            draw=args.draw, region=args.region,
            data_file=args.data_file,
            model_type=args.model_type,
            tune=args.tune,
            trials=args.trials,
            cv_splits=args.cv_splits,
            compare=args.compare_models,
            use_mlflow=use_mlflow,
            promote=args.promote,
        )


if __name__ == "__main__":
    main()
