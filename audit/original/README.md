# 🩺 GlycoRisk: End-to-End Diabetes Risk Assessment & Lifestyle Simulation

An evidence-based Machine Learning web application for **Non-Invasive Diabetes Risk Screening & Interactive Lifestyle Simulation**, trained and benchmarked on national **NHANES (National Health and Nutrition Examination Survey)** datasets.

---

## 🌟 Key Features

1. **🔬 Evidence-Based Machine Learning Engine**:
   - Trained on adult cohorts ($\ge 20$ years old) from multi-table NHANES data (`demographic`, `examination`, `labs`, `diet`, `medications`, `questionnaire`).
   - Clinical ground truth label combining HbA1c ($\ge 6.5\%$), Fasting Plasma Glucose ($\ge 126\text{ mg/dL}$), physician diagnosis, and anti-diabetic medications.
   - Non-invasive feature focus for consumer web screening: anthropometrics, blood pressure, dietary sugar/fiber, sleep quality, and physical activity.
   - 5-Fold Stratified Cross-Validation benchmark across Logistic Regression, Random Forest, ExtraTrees, and Gradient Boosting.

2. **🔍 Local SHAP Explainability**:
   - Powered by TreeSHAP explainability (`shap.TreeExplainer`).
   - Breaks down individual predictions into **Top Risk Accelerators** (positive SHAP impact) and **Protective Factors** (negative SHAP impact) with clinically grounded advice.

3. **🎛️ Real-Time "What-If" Lifestyle Simulator**:
   - Interactive reactive sliders for waist reduction, weight loss, dietary sugar cuts, fiber boost, and exercise adoption.
   - Provides instant recalculated risk scores, absolute risk reduction, and motivational summary takeaways.

4. **⚡ Modern, Minimalist Web Architecture**:
   - **Backend**: FastAPI REST API with Pydantic schema validation and CORS.
   - **Frontend**: Next.js 14 (App Router) + Tailwind CSS, crafted with a calm health-tech aesthetic (inspired by Apple Health, Notion, and Linear).

---

## 📁 Repository Structure

```
diabetes-ml-app/
├── dataset/                      # NHANES CSV raw data
│   ├── demographic.csv
│   ├── examination.csv
│   ├── labs.csv
│   ├── diet.csv
│   ├── medications.csv
│   └── questionnaire.csv
├── ml_pipeline/                  # Part 1: Data Science & ML Engine
│   ├── __init__.py
│   ├── preprocess.py             # Data ingestion, cohort filtering, target & feature engineering
│   ├── eda.py                    # Exploratory Data Analysis & summary statistics
│   ├── train.py                  # 5-Fold Stratified CV benchmarking & model selection
│   ├── evaluate.py               # Model evaluation & SHAP artifact exporter
│   └── artifacts/                # Serialized production artifacts
│       ├── model_pipeline.joblib # Full Preprocessor + Estimator
│       ├── shap_explainer.joblib # TreeSHAP explainer & background distribution
│       ├── metadata.json         # Feature statistics, test metrics, and percentiles
│       └── eda_summary.json      # Dataset statistics and correlations
├── backend/                      # Part 2: FastAPI Server
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI application entry point & CORS
│   │   ├── schemas.py            # Pydantic input/output schemas
│   │   ├── model_service.py      # Inference engine & SHAP local explainability
│   │   └── config.py             # Configuration & path management
│   └── requirements.txt          # Backend dependencies
├── frontend/                     # Part 3: Next.js + Tailwind Minimal UI
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx        # Root layout
│   │   │   ├── page.tsx          # Single-page application orchestrator
│   │   │   └── globals.css       # Base CSS styling & design tokens
│   │   ├── components/
│   │   │   ├── AssessmentForm.tsx # 2-step streamlined non-invasive form
│   │   │   ├── RiskGauge.tsx      # Semi-circular SVG risk meter
│   │   │   ├── RiskDrivers.tsx    # SHAP feature impact visualization cards
│   │   │   ├── WhatIfSimulator.tsx# Real-time reactive slider simulator
│   │   │   └── RecommendationCard.tsx # Actionable checklist & medical disclaimer
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript interfaces
│   │   └── lib/
│   │       └── api.ts            # Fetch API client
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   └── postcss.config.js
└── README.md
```

---

## 📊 Machine Learning Model Benchmarks

Evaluated on $n = 5,654$ adult respondents with holdout test split ($20\%$, $n = 1,131$):

| Model | 5-Fold CV ROC-AUC | Test ROC-AUC | Sensitivity (Recall) | Specificity | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **0.7845** | **0.7790** | **71.69%** | **68.75%** | **0.4432** |
| ExtraTrees Classifier | 0.7819 | 0.7781 | 73.05% | 67.82% | 0.4470 |
| Logistic Regression (Balanced) | 0.7830 | 0.7765 | 73.40% | 67.20% | 0.4467 |
| HistGradientBoosting | 0.7734 | 0.7682 | 67.23% | 71.05% | 0.4361 |

### Top 5 Predictive Features (Global SHAP Importance)
1. **Age (`age`)**: Metabolic degradation and cumulative insulin resistance with age.
2. **Waist-to-Height Ratio (`waist_to_height_ratio`)**: Primary proxy for visceral/abdominal fat.
3. **Waist Circumference (`waist_cm`)**: Central adiposity biomarker.
4. **Hypertension Status (`hypertension_flag`)**: Co-morbidity of metabolic syndrome.
5. **Daily Sugar Intake (`daily_sugar_g`)**: Glycemic load and dietary stress.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ / 3.12
- Node.js 18+ & npm
- `uv` (recommended) or standard `python -m venv`

### 1. Set Up Python Environment & Run ML Pipeline
```bash
# Create and activate virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install Python dependencies
uv pip install -r backend/requirements.txt

# Run ML Pipeline (Train -> Evaluate -> Export Artifacts)
PYTHONPATH=. python ml_pipeline/preprocess.py
PYTHONPATH=. python ml_pipeline/eda.py
PYTHONPATH=. python ml_pipeline/train.py
PYTHONPATH=. python ml_pipeline/evaluate.py
```

### 2. Start FastAPI Backend Server

**Option A: Running from Project Root (`CPE310 AI/`):**
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Option B: Running from inside `backend/` directory (`CPE310 AI/backend/`):**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at:
- **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Start Next.js Frontend
```bash
# In a new terminal tab
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔌 API Reference

### `POST /api/predict`
Calculates diabetes risk score and local SHAP factor contributions.
```json
// Request
{
  "age": 55,
  "gender": "male",
  "weight_kg": 90,
  "height_cm": 172,
  "waist_cm": 104,
  "systolic_bp": 138,
  "diastolic_bp": 86,
  "daily_sugar_g": 95,
  "daily_fiber_g": 12,
  "hypertension_history": true,
  "smoking_history": false,
  "vigorous_activity": false,
  "sleep_trouble": true
}
```

```json
// Response
{
  "risk_score": 52.4,
  "risk_category": "High Risk (Action Recommended)",
  "risk_tier": "high",
  "percentile": 72.0,
  "key_drivers": [
    {
      "feature": "Age",
      "user_value": "55.0",
      "impact": "+22.4%",
      "direction": "negative",
      "advice": "Advancing age is a natural metabolic risk factor; proactive screening and strength training help maintain metabolic rate."
    },
    {
      "feature": "Waist-to-Height Ratio",
      "user_value": "0.60",
      "impact": "+15.8%",
      "direction": "negative",
      "advice": "Central adiposity (waist-to-height > 0.5) is strongly correlated with visceral fat and insulin resistance."
    }
  ],
  "protective_factors": [
    {
      "feature": "Smoking History",
      "user_value": "Non-smoker",
      "impact": "-5.2%",
      "direction": "positive",
      "advice": "Absence of active smoking habits protects vascular and metabolic health."
    }
  ],
  "bmi": 30.4,
  "waist_to_height_ratio": 0.6
}
```

### `POST /api/simulate`
Simulates lifestyle intervention outcomes.
```json
// Request
{
  "baseline": { ... },
  "adjustments": {
    "delta_weight_kg": -6.0,
    "delta_waist_cm": -7.0,
    "delta_sugar_pct": -40.0,
    "delta_fiber_g": 10.0,
    "active_exercise": true,
    "manage_hypertension": true
  }
}
```

---

## ⚕️ Medical Disclaimer
This application is designed as an educational, non-invasive risk screening tool based on population-level statistical associations from NHANES data. It does **not** provide clinical diagnosis or substitute for professional medical advice. Always consult a qualified physician or healthcare provider for clinical diagnosis and treatment plans.
