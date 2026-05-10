# 🏸 BWF Badminton Ranking Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.1-02C3BB?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

**A production-grade AI system for predicting BWF Badminton World Rankings up to 2035**

*Hybrid LightGBM + LSTM · REST API · MLOps · Docker · CI/CD*

</div>

---

## 📖 Overview

This project transforms raw BWF ranking history into **future ranking predictions** using a hybrid ensemble of:
- **LightGBM** — 15 specialized regressors (5 draws × 3 regions) trained on time-series features
- **LSTM** — Deep learning model capturing long-range sequential patterns

The system is built as a **production-ready AI Engineering showcase**, following MLOps best practices:

| Capability | Stack |
|---|---|
| ML Training | LightGBM, scikit-learn, Walk-Forward CV |
| Experiment Tracking | MLflow |
| REST API | FastAPI + Pydantic |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Streamlit + Evidently |

---

## 🏗️ Project Structure

```
bwf-ranking-prediction/
├── src/                    # Core library
│   ├── config.py           # Centralized settings (pydantic-settings)
│   ├── data/
│   │   ├── loader.py       # Raw data ingestion & validation
│   │   ├── preprocessor.py # Cleaning pipeline
│   │   └── feature_engine.py # Feature engineering (lag, rolling, momentum)
│   ├── models/
│   │   ├── trainer.py      # LightGBM training with walk-forward CV
│   │   └── predictor.py    # Model inference
│   ├── evaluation/
│   │   └── metrics.py      # MAE, RMSE, R², MAPE, Spearman corr
│   └── utils/
│       └── logger.py       # Structured logging (loguru)
├── api/                    # FastAPI REST API (Phase 3)
├── scripts/                # CLI entry points
│   └── train.py            # python scripts/train.py --draw MS --region Asia
├── tests/                  # Pytest unit & integration tests
├── data/
│   ├── raw/                # Raw BWF CSV files (git-ignored)
│   └── processed/          # Processed datasets
├── models/                 # Trained .pkl models (git-ignored, use DVC)
├── .github/workflows/      # CI/CD pipelines (Phase 5)
├── pyproject.toml          # Project metadata & tool configs
├── requirements.txt        # Pinned dependencies
└── Makefile                # Developer shortcuts
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/WooCSSIN/HYBRID-BADMINTON-RANKING-SERIES-PREDICTION-MODEL.git
cd HYBRID-BADMINTON-RANKING-SERIES-PREDICTION-MODEL

# Install all dependencies
pip install -e ".[all]"
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Prepare Data
```bash
# Place your BWF CSV file in:
# data/raw/bwf_ranking.csv
```

### 4. Train a Model
```bash
# Train MS (Men's Singles) model for Asia region
python scripts/train.py --draw MS --region Asia

# Train all 15 combinations
make train-all
```

### 5. Start the API (Phase 3)
```bash
make api
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## 🎯 BWF Draws & Regions

| Draw Code | Description |
|---|---|
| `MS` | Men's Singles |
| `WS` | Women's Singles |
| `MD` | Men's Doubles |
| `WD` | Women's Doubles |
| `XD` | Mixed Doubles |

**Regions:** `Global` · `Asia` · `Europe`

---

## 📊 Model Architecture

### LightGBM (Primary)
- **15 specialized models**: one per (draw, region) pair
- **Features**: 20+ lag/rolling/momentum features per player
- **Validation**: 5-fold walk-forward cross-validation
- **Target**: Monthly BWF ranking points

### Feature Engineering
```
Lag features:       points & rank at t-1, t-3, t-6, t-12
Rolling stats:      mean/std/max over 3, 6, 12-month windows
Trend features:     ΔPoints (1m, 3m), % change
Momentum:           consecutive months in Top-10 streak
Seasonality:        sin/cos month encoding
Field-relative:     points vs. field mean/max
```

---

## 🛠️ Developer Commands

```bash
make install       # Install core dependencies
make install-dev   # Install all + dev tools
make lint          # Run ruff linter
make format        # Run black formatter
make test          # Run pytest with coverage
make train         # Train one model (DRAW=MS REGION=Asia)
make train-all     # Train all 15 combinations
make api           # Start FastAPI dev server
make clean         # Remove cache files
```

---

## 🗺️ Roadmap

- [x] **Phase 1**: Code restructure & engineering best practices
- [ ] **Phase 2**: MLflow experiment tracking & model registry
- [ ] **Phase 3**: FastAPI REST API with Swagger docs
- [ ] **Phase 4**: Docker containerization
- [ ] **Phase 5**: GitHub Actions CI/CD + Cloud deployment
- [ ] **Phase 6**: Streamlit monitoring dashboard

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ by <a href="https://github.com/WooCSSIN">Hà Nhật Nguyên Vũ</a>
</div>
