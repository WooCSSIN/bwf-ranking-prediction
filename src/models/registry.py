"""
Model registry: versioned storage and retrieval of trained models.
Keeps a local JSON manifest alongside MLflow for offline use.
"""
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.utils.logger import logger

REGISTRY_MANIFEST = "models/registry.json"


class ModelRegistry:
    """
    Manages versioned model artifacts locally + in MLflow.

    Every saved model is recorded in models/registry.json with:
    - version, draw, region, metrics, file path, timestamp

    Usage:
        registry = ModelRegistry()
        version = registry.save(model, features, draw="MS", region="Asia", metrics={...})
        model, meta = registry.load(draw="MS", region="Asia")
    """

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(REGISTRY_MANIFEST)
        self._manifest: Dict = self._load_manifest()

    # ── Public API ───────────────────────────────────────────────────────────

    def save(
        self,
        model: Any,
        feature_names: List[str],
        draw: str,
        region: str,
        metrics: Dict[str, float],
        version: Optional[str] = None,
        mlflow_run_id: Optional[str] = None,
    ) -> str:
        """
        Persist model and record in registry manifest.

        Returns:
            Version string (e.g. 'v1.2.0').
        """
        version = version or self._next_version(draw, region)
        model_key = self._key(draw, region)
        filename = f"lgbm_{draw.lower()}_{region.lower()}_{version}.pkl"
        model_path = self.models_dir / filename

        payload = {
            "model": model,
            "features": feature_names,
            "version": version,
            "draw": draw,
            "region": region,
        }
        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

        entry = {
            "version": version,
            "filename": filename,
            "draw": draw,
            "region": region,
            "metrics": metrics,
            "feature_count": len(feature_names),
            "mlflow_run_id": mlflow_run_id,
            "saved_at": datetime.utcnow().isoformat(),
            "stage": "Staging",
        }

        if model_key not in self._manifest:
            self._manifest[model_key] = []
        self._manifest[model_key].append(entry)
        self._save_manifest()

        logger.success(f"Model saved: {filename} ({version}) → {model_path}")
        return version

    def load(
        self,
        draw: str,
        region: str,
        version: Optional[str] = None,
        stage: Optional[str] = "Production",
    ) -> tuple:
        """
        Load a model from the registry.

        Args:
            draw:    Draw code.
            region:  Region name.
            version: Specific version (None = latest production model).
            stage:   Stage filter ('Staging' or 'Production').

        Returns:
            (model, metadata_dict)
        """
        entry = self._find_entry(draw, region, version, stage)
        if entry is None:
            raise FileNotFoundError(
                f"No model found for draw={draw}, region={region}, "
                f"version={version or 'latest'}, stage={stage}"
            )

        model_path = self.models_dir / entry["filename"]
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        logger.info(f"Loaded model: {entry['filename']} ({entry['version']})")
        return payload["model"], entry

    def promote(self, draw: str, region: str, version: str) -> None:
        """Promote a model version to Production stage."""
        model_key = self._key(draw, region)
        entries = self._manifest.get(model_key, [])

        for entry in entries:
            if entry["version"] == version:
                entry["stage"] = "Production"
                logger.success(f"Promoted {draw}/{region} {version} → Production")
            elif entry["stage"] == "Production":
                entry["stage"] = "Archived"    # Demote previous Production

        self._save_manifest()

    def list_models(self, draw: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
        """List all registered models, optionally filtered."""
        all_entries = []
        for key, entries in self._manifest.items():
            all_entries.extend(entries)

        if draw:
            all_entries = [e for e in all_entries if e["draw"].upper() == draw.upper()]
        if region:
            all_entries = [e for e in all_entries if e["region"] == region]

        return sorted(all_entries, key=lambda x: x["saved_at"], reverse=True)

    def get_best_model(self, draw: str, region: str, metric: str = "mae_cv") -> Optional[Dict]:
        """Return the entry with the best (lowest) metric value."""
        entries = self._manifest.get(self._key(draw, region), [])
        if not entries:
            return None
        return min(entries, key=lambda e: e["metrics"].get(metric, float("inf")))

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _key(self, draw: str, region: str) -> str:
        return f"{draw.upper()}_{region}"

    def _next_version(self, draw: str, region: str) -> str:
        entries = self._manifest.get(self._key(draw, region), [])
        next_num = len(entries) + 1
        return f"v{next_num}.0.0"

    def _find_entry(
        self,
        draw: str,
        region: str,
        version: Optional[str],
        stage: Optional[str],
    ) -> Optional[Dict]:
        entries = self._manifest.get(self._key(draw, region), [])
        if not entries:
            return None
        if version:
            return next((e for e in entries if e["version"] == version), None)
        # Return latest matching stage
        filtered = [e for e in entries if stage is None or e.get("stage") == stage]
        return filtered[-1] if filtered else entries[-1]

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)
