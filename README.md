# 📊 Data Analysis & AI Insight Platform

A Flask-based data analysis workspace for uploading real datasets, exploring them through multiple analysis tabs, exporting artifacts, and using AI-assisted narrative summaries when Gemini quota is available.

## What this project does right now

- Upload and analyze **CSV / Excel / JSON / TXT** datasets.
- Generate analysis across dedicated views:
  - **Overview**
  - **Interactive**
  - **Detailed Analysis (forecast-focused)**
  - **Correlation**
  - **Categories**
- Provide export actions directly from the UI:
  - cleaned CSV
  - AI summary HTML
  - static plots ZIP
  - full report HTML
  - full report PDF
- Support **three UI themes**: AMOLED, Light, Dark.
- Offer a restored **Research Labs hub** with pages for:
  - Forecast
  - Anomaly
  - Quality
  - Change Points
  - Conformal
  - SHAP
  - Multivariate

---

## 🖼️ Current UI gallery (captured April 2026)

All screenshots below were captured from the current app using datasets in `datasets/`:

- `datasets/Project 1 - Weather Dataset.csv`
- `datasets/Life Expectancy Data.csv`

### Upload page (theme variants)

| AMOLED | Light | Dark |
|---|---|---|
| ![Upload AMOLED](screenshots/upload_amoled.png) | ![Upload Light](screenshots/upload_light.png) | ![Upload Dark](screenshots/upload_dark.png) |

### Analysis tabs (Weather dataset)

| View | Screenshot |
|---|---|
| Overview (AMOLED) | ![Weather Overview AMOLED](screenshots/weather_overview_amoled.png) |
| Overview (Light) | ![Weather Overview Light](screenshots/weather_overview_light.png) |
| Overview (Dark) | ![Weather Overview Dark](screenshots/weather_overview_dark.png) |
| Interactive (AMOLED) | ![Weather Interactive AMOLED](screenshots/weather_interactive_amoled.png) |
| Interactive (Light) | ![Weather Interactive Light](screenshots/weather_interactive_light.png) |
| Interactive (Dark) | ![Weather Interactive Dark](screenshots/weather_interactive_dark.png) |
| Detailed Analysis | ![Weather Detailed](screenshots/weather_detailed_light.png) |
| Correlation | ![Weather Correlation](screenshots/weather_correlation_light.png) |
| Categories | ![Weather Categories](screenshots/weather_categories_light.png) |
| Export controls in-page | ![Weather Exports](screenshots/weather_exports_header.png) |

> ✅ Categories behavior update: the active temporal-axis column is now filtered out of category charts to avoid self-count temporal noise.

### Research Labs pages (Weather dataset)

| Page | Screenshot |
|---|---|
| Labs Hub (Light) | ![Weather Labs Hub Light](screenshots/weather_labs_hub_light.png) |
| Labs Hub (Dark) | ![Weather Labs Hub Dark](screenshots/weather_labs_hub_dark.png) |
| Forecast Lab (Light) | ![Weather Labs Forecast Light](screenshots/weather_labs_forecast_light.png) |
| Forecast Lab (Dark) | ![Weather Labs Forecast Dark](screenshots/weather_labs_forecast_dark.png) |
| Anomaly Lab | ![Weather Labs Anomaly](screenshots/weather_labs_anomaly_light.png) |
| Quality Lab | ![Weather Labs Quality](screenshots/weather_labs_quality_light.png) |
| Change Points Lab | ![Weather Labs Change Points](screenshots/weather_labs_changepoints_light.png) |
| Conformal Lab | ![Weather Labs Conformal](screenshots/weather_labs_conformal_light.png) |
| SHAP Lab | ![Weather Labs SHAP](screenshots/weather_labs_shap_light.png) |
| Multivariate Lab | ![Weather Labs Multivariate](screenshots/weather_labs_multivariate_light.png) |

### Additional dataset scenario (Life Expectancy)

| View | Screenshot |
|---|---|
| Overview (Light) | ![Life Overview Light](screenshots/life_overview_light.png) |
| Overview (Dark) | ![Life Overview Dark](screenshots/life_overview_dark.png) |
| Interactive (Light) | ![Life Interactive Light](screenshots/life_interactive_light.png) |
| Detailed Analysis | ![Life Detailed](screenshots/life_detailed_light.png) |
| Correlation | ![Life Correlation](screenshots/life_correlation_light.png) |
| Categories | ![Life Categories](screenshots/life_categories_light.png) |
| Labs Hub | ![Life Labs Hub](screenshots/life_labs_hub_light.png) |

---

## ✅ Current capabilities (implemented)

### Data ingestion and preparation
- Upload pipeline with hashed filenames and session-aware caching.
- Supported extensions: `.csv`, `.xlsx`, `.xls`, `.json`, `.txt`.
- Numeric coercion and missing-value profiling.
- Datetime inference for temporal plotting.

### Analysis views
- **Overview**: table preview, summary stats, missing values, AI panel.
- **Interactive**: Plotly history/anomaly/forecast overlays and distributions.
- **Detailed Analysis**: static plots per column including trend/distribution/forecast and STL (when applicable).
- **Correlation**: Spearman/Pearson matrix and heatmaps.
- **Categories**: categorical frequency charts for non-numeric columns, excluding the active temporal axis column.

### Export and reporting
- `download cleaned CSV`
- `download AI summary HTML`
- `download static plots ZIP`
- `download full report HTML`
- `download full report PDF`

### AI integration
- Gemini-backed dataset summaries and Q&A when available.
- Model fallback and cache-aware behavior.
- Graceful degraded/offline messaging when AI is unavailable.

### Runtime safety and performance
- Talisman security headers and limiter integration.
- Local caching for expensive computations (numeric transforms, anomalies, correlation, heatmaps, forecasts).
- Downsampling-aware interactive handling for large datasets.

---

## 🚧 Work in progress

The project currently has active ongoing work in these areas:

- **Research labs depth**: lab pages are restored and navigable, with incremental backend depth being expanded across advanced methods.
- **AI prompt/runtime robustness**: improving long-summary reliability, quota-aware fallbacks, and response quality consistency.
- **UI/UX iteration**: refining layout density, theme consistency, and per-view ergonomics for large datasets.
- **Export consistency improvements**: ongoing harmonization between on-page charts and ZIP/PDF outputs.

---

## 🚀 Future advancements

Planned evolution areas include:

- richer advanced analytics in labs (e.g., more robust explainability and uncertainty pipelines),
- stronger asynchronous processing for very large datasets,
- enhanced multi-dataset comparative workflows,
- expanded report customization and selective export packs,
- optional user preferences/profile persistence and API-oriented workflows.

---

## 🛠️ Tech stack

- **Backend**: Flask, Python 3.11+
- **Data**: pandas, numpy, statsmodels, scikit-learn
- **Visualization**: matplotlib, plotly
- **AI**: Google Gemini integration
- **Quality**: pytest, ruff, mypy

---

## 🚀 Quick start

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Create `.env` (already gitignored) and add secrets if AI features are needed.
4. Run the app with `python app.py`.
5. Open `http://127.0.0.1:5000`.

Sample datasets are available under `datasets/`.

---

## 💻 Development checks

- Lint: `python -m ruff check .`
- Type check: `python -m mypy app.py check_models.py`
- Tests: `python -m pytest -q tests`

VS Code tasks for these checks are already configured.

---

## 🧪 Notes on screenshot refresh

This README has been updated to reflect current routes/pages/themes and to include a broad capture set for:

- all main analysis views,
- all lab pages,
- multiple themes,
- multiple real datasets from the repository.

---

## ⚠️ Disclaimer

AI output is assistive and may be imperfect. For high-stakes decisions, validate conclusions with domain and statistical review.
