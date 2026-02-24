<div align="center">

<br/>

<h1>🏭 PredictiveMaintenance.AI</h1>
<h3>Production-Grade Machine Failure Prediction for Smart Factories</h3>

<br/>

<p>
  <img src="https://img.shields.io/badge/R-4.x-276DC3?style=flat-square&logo=r&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-Champion_Model-EC6C00?style=flat-square"/>
  <img src="https://img.shields.io/badge/Plumber-REST_API-4B9CD3?style=flat-square"/>
  <img src="https://img.shields.io/badge/SMOTE-Imbalance_Handling-9B59B6?style=flat-square"/>
  <img src="https://img.shields.io/badge/caret-ML_Pipeline-E74C3C?style=flat-square"/>
  <img src="https://img.shields.io/badge/TailwindCSS-Dashboard-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-UCI_AI4I_2020-2ECC71?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square"/>
</p>

<br/>

> A **full production machine learning system** that predicts machine failure risk in real time using the AI4I 2020 manufacturing dataset. Built with an R-based training pipeline, a live **Plumber REST API**, and an interactive **web dashboard** that gives maintenance engineers instant failure risk scores, color-coded alerts, and physics-based sensor readouts — all from a browser.

<br/>

```
10,000 Machine Records  ──►  Feature Engineering  ──►  SMOTE + CV  ──►  XGBoost Champion
     AI4I 2020 Dataset              7 Physics Features        5-Fold                  │
                                                                                       ▼
                               Web Dashboard  ◄──  Plumber REST API  ◄──  best_model.rds
                               Real-Time UI         /predict endpoint       Risk % + Drivers
```

</div>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [System Architecture](#-system-architecture)
3. [Dataset](#-dataset)
4. [Feature Engineering](#-feature-engineering)
5. [ML Pipeline](#-ml-pipeline)
   - [Data Preprocessing](#data-preprocessing)
   - [Class Imbalance Handling](#class-imbalance-handling)
   - [Model Training & Selection](#model-training--selection)
   - [Model Performance Results](#model-performance-results)
6. [Plumber REST API](#-plumber-rest-api)
7. [Interactive Web Dashboard](#-interactive-web-dashboard)
   - [Dashboard Panels Explained](#dashboard-panels-explained)
   - [Risk Level System](#risk-level-system)
8. [Project Structure](#-project-structure)
9. [Quickstart](#-quickstart)
10. [Dependencies](#-dependencies)
11. [Serialized Artifacts](#-serialized-artifacts)
12. [Key Design Decisions](#-key-design-decisions)
13. [Limitations & Future Work](#-limitations--future-work)
14. [References](#-references)

---

## 🔭 Project Overview

Unplanned machine downtime costs the manufacturing industry billions annually. Traditional time-based maintenance schedules are either too frequent (wasteful) or too infrequent (failure-prone). This project delivers a **condition-based maintenance system** — one that monitors real operational sensor data and predicts failure risk before a breakdown occurs.

The system answers three research questions:

1. **Can machine failures be predicted accurately from sensor data?** Yes — XGBoost achieves an F1-score of 0.625 and AUC of 0.949 on held-out test data.
2. **Which operational parameters most predict failure?** Torque (r = 0.19), tool wear (r = 0.11), and air temperature (r = 0.08) are the dominant drivers.
3. **Can a real-time predictive system be deployed for practical use?** Yes — the trained model is packaged as a live REST API and served through an interactive web dashboard requiring no installation.

### What the System Delivers

- A trained **XGBoost classifier** that predicts failure probability from 6 sensor inputs
- A **Plumber REST API** (`/predict` endpoint) that processes inputs, runs all feature engineering server-side, and returns a JSON payload with failure risk percentage and driver flags
- An **interactive HTML dashboard** that a maintenance engineer can open in any browser to get real-time failure risk assessments with color-coded alerts and physics-based sensor readouts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PredictiveMaintenance.AI — System Map                        │
├────────────────────────┬───────────────────────┬───────────────────────────────┤
│  TRAINING PIPELINE     │  DEPLOYMENT LAYER      │  PRESENTATION LAYER           │
│  (R Script)            │  (R / Plumber)         │  (HTML / JS / Tailwind)       │
│                        │                        │                               │
│  ai4i2020.csv          │  best_model.rds ───────►  POST /predict                │
│       │                │  dmy_transformer.rds   │       │                       │
│       ▼                │  preproc_params.rds    │       ▼                       │
│  Clean + Remove Leak   │                        │  Risk % (0-100)               │
│       │                │  create_feature_df()   │  SVG Gauge                    │
│       ▼                │  (Identical to Train)  │  Recommendation Card          │
│  70/30 Time Split      │       │                │  Key Risk Drivers Panel       │
│       │                │       ▼                │  Sensor Readout Panel         │
│       ▼                │  dmy_transformer ──────►  Color-coded Alerts           │
│  Feature Engineering   │  (One-Hot Encode)      │  (Green / Yellow / Red)       │
│  (7 Features)          │       │                │                               │
│       │                │       ▼                │                               │
│       ▼                │  XGBoost Predict       │                               │
│  SMOTE + RepeatedCV    │  type = "prob"         │                               │
│  (Optimise Sensitivity)│       │                │                               │
│       │                │       ▼                │                               │
│       ▼                │  JSON Response:        │                               │
│  GLMNET vs DT vs XGB   │  risk, power_kw,       │                               │
│  GridSearch            │  temp_diff_k,          │                               │
│       │                │  torque_speed_ratio,   │                               │
│       ▼                │  is_high_torque,       │                               │
│  Save 3 Artifacts      │  is_low_speed,         │                               │
│  (preproc/dmy/model)   │  is_extreme_wear       │                               │
└────────────────────────┴───────────────────────┴───────────────────────────────┘
```

---

## 📊 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

The dataset contains 10,000 chronologically ordered observations from a real manufacturing facility, spanning 6 months of machine operation. Each row represents a single operational snapshot with binary failure labeling.

### Raw Variables

| Variable | Range | Description |
|---|---|---|
| `UDI` | 1–10,000 | Unique identifier *(dropped — causes overfitting)* |
| `Product ID` | — | Machine serial number *(dropped — causes overfitting)* |
| `Type` | L, M, H | Product category: Low / Medium / High speed |
| `Air Temperature [K]` | 295.3–304.5 | Ambient environment temperature |
| `Process Temperature [K]` | 305.7–313.8 | Machine operating temperature |
| `Rotational Speed [rpm]` | 1,168–2,886 | Spindle rotational velocity |
| `Torque [Nm]` | 3.8–76.6 | Load applied to the machine |
| `Tool Wear [min]` | 0–253 | Cumulative tool usage duration |
| `Machine Failure` | 0 / 1 | **Target** — binary failure label |
| `TWF, HDF, PWF, OSF, RNF` | 0 / 1 | Failure mechanism flags *(dropped — data leakage)* |

### Class Distribution

The dataset is **severely imbalanced**, reflecting real manufacturing conditions:

| Class | Count | Percentage |
|---|---|---|
| No Failure (0) | 9,661 | 96.61% |
| Failure (1) | 339 | **3.39%** |
| **Imbalance Ratio** | | **1:28** |

Failure rate by product type: **Type L → 3.92%** · **Type M → 2.77%** · **Type H → 2.09%**

Without intervention, any classifier achieves 96% accuracy by predicting "No Failure" for every row. This is addressed explicitly — see [Class Imbalance Handling](#class-imbalance-handling).

### Failure Mechanism Breakdown

| Failure Type | Code | Count | Rate |
|---|---|---|---|
| Heat Dissipation Failure | HDF | 115 | 1.15% |
| Overstrain Failure | OSF | 98 | 0.98% |
| Power Failure | PWF | 95 | 0.95% |
| Tool Wear Failure | TWF | 46 | 0.46% |
| Random Failure | RNF | 19 | 0.19% |

---

## ⚙️ Feature Engineering

Seven domain-informed features are constructed on top of the raw sensor readings, capturing physics-based relationships, operational stress conditions, and tool degradation signals that raw sensors alone cannot express.

These features are computed **identically** in both the training pipeline and the live API — a critical design requirement to prevent training-serving skew.

| # | Feature | Formula / Logic | Type | Purpose |
|---|---|---|---|---|
| 1 | `power_kw` | `Torque × RPM / 1000` | Continuous | Physical power consumption — high power signals mechanical stress |
| 2 | `temp_diff_k` | `Process Temp − Air Temp` | Continuous | Operational heat generation — large differential suggests thermal strain |
| 3 | `torque_speed_ratio` | `Torque / (RPM + ε)` | Continuous | Load-normalized speed — captures heavy load at low speed |
| 4 | `tool_wear_category` | Cut into `{New, Moderate, Worn, Critical}` | Ordered factor | Degradation stage — quartile-based binning of raw wear minutes |
| 5 | `is_high_torque` | `Torque > 95th percentile (training set)` | Binary flag | Above 95th percentile torque stress condition |
| 6 | `is_low_speed` | `RPM < 5th percentile (training set)` | Binary flag | Unusually low rotational speed |
| 7 | `is_extreme_wear` | `Tool Wear > 95th percentile (training set)` | Binary flag | Near end-of-life tool degradation alert |

> **Critical note:** The percentile thresholds for features 5–7 are computed **only on training data** and saved to `preproc_params.rds`. The API loads these at startup to ensure the same boundaries are applied at prediction time. This prevents a subtle but serious form of data leakage where test-set statistics would otherwise contaminate threshold definitions.

---

## 🤖 ML Pipeline

### Data Preprocessing

**Leakage prevention — two layers:**

1. **Column removal:** `UDI` and `Product ID` are dropped (unique identifiers that cause overfitting). The five failure mechanism flags (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are also dropped because they are consequences of failure, not precursors. Using them would constitute direct data leakage.

2. **Temporal train-test split:** Data is split **chronologically**, not randomly. The first 70% of rows (7,000 observations) form the training set; the remaining 30% (3,000 rows) form the test set. This mirrors real deployment conditions.

```r
split_index <- floor(nrow(data_cleaned) * 0.70)
train_raw <- data_cleaned[1:split_index, ]
test_raw  <- data_cleaned[(split_index+1):nrow(data_cleaned), ]
```

**One-hot encoding:** After feature engineering, the categorical `type` column (L / M / H) and `tool_wear_category` are one-hot encoded using a `dummyVars` transformer fitted exclusively on training data, then saved to `dmy_transformer.rds`.

### Class Imbalance Handling

The 1:28 class imbalance is addressed using **upsampling within each cross-validation fold** (`sampling = "up"` in `trainControl`):

- Applied **inside** each fold, not before splitting — preventing inflated CV scores
- Ensures the model learns genuine failure patterns rather than exploiting majority-class frequency
- The primary training metric is **Sensitivity (Recall)**, prioritising failure detection over false-alarm reduction

### Model Training & Selection

Three candidate models are trained under identical conditions using **Repeated 5-Fold Cross-Validation** (5 folds × 2 repeats = 10 total evaluation folds), with a parallel backend on all available CPU cores:

**Model 1 — Regularized Logistic Regression (GLMNET)**
- Elastic Net regularization with `alpha = 1` (Lasso penalty)
- Lambda grid: 5 values from 0.0001 to 0.1
- Features centered and scaled before fitting

**Model 2 — Decision Tree (rpart)**
- Complexity parameter tuned via `tuneLength = 10`
- Interpretable rule-based classifier — serves as a transparency benchmark

**Model 3 — XGBoost (xgbTree) — Champion**
- Gradient-boosted tree ensemble with explicit hyperparameter search:

```r
tuneGrid_xgb <- expand.grid(
  nrounds          = c(100, 150),   # Boosting rounds
  max_depth        = c(4, 6),       # Tree depth
  eta              = 0.1,           # Learning rate
  gamma            = 0,
  colsample_bytree = 0.8,           # Feature subsampling per tree
  min_child_weight = 1,
  subsample        = 0.8            # Row subsampling per tree
)
```

**Champion selection:** The winning model is selected by **F1-Score on the held-out test set** — the correct metric for imbalanced classification, balancing precision and recall. The winner is saved as `best_model.rds`.

### Model Performance Results

| Model | F1-Score | Sensitivity | Precision | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression (GLMNET) | 0.2939 | 0.5902 | 0.1957 | 0.942 | 0.845 |
| Decision Tree (rpart) | 0.4481 | 0.6721 | 0.3361 | 0.966 | 0.829 |
| **XGBoost (xgbTree) ✓** | **0.6250** | **0.5738** | **0.6863** | **0.986** | **0.949** |

**Why XGBoost wins:**

XGBoost's F1-Score of 0.625 is **40% better than Decision Tree** and **112% better than GLMNET**. Its precision of 0.686 means 68.6% of flagged machines will genuinely fail — critical for avoiding alert fatigue. At a 10% false positive rate, XGBoost achieves ~98% sensitivity versus GLMNET's 90%, meaning 8 additional real failures caught per 100. The Decision Tree catches slightly more failures in aggregate (sensitivity 0.672 vs 0.574) but generates **triple the false alarms** (precision 0.336 vs 0.686) — in practice this erodes maintenance team trust and leads to predictions being ignored.

---

## 🌐 Plumber REST API

**File:** `Plumber_API_for_Model_1.R`

The trained XGBoost model is served as a production REST API using R's `{plumber}` package. The API is fully self-contained — it loads the three serialized artifacts at startup and exposes a single prediction endpoint.

### Startup Sequence

```
API starts
   ├── Load best_model.rds        (trained XGBoost caret object)
   ├── Load dmy_transformer.rds   (one-hot encoder)
   └── Load preproc_params.rds    (training-set quantile thresholds)
```

If any artifact is missing, the API stops with a clear error directing the user to re-run the training pipeline.

### Endpoint

```
POST  /predict
```

**Parameters** (query string):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"L"` | Machine type — `L`, `M`, or `H` |
| `air_temp` | float | `25` | Air temperature in **°C** (converted to Kelvin server-side) |
| `process_temp` | float | `35` | Process temperature in **°C** (converted to Kelvin server-side) |
| `rpm` | float | `1500` | Rotational speed in rpm |
| `torque` | float | `40` | Torque in Nm |
| `wear` | float | `10` | Tool wear in minutes |

> The API accepts temperatures in **Celsius** for user convenience. The `create_feature_df()` function converts them internally (`+ 273.15`) to match the Kelvin scale used during training.

**Response (JSON):**

```json
{
  "risk": 23.4,
  "power_kw": 60.0,
  "temp_diff_k": 10.2,
  "torque_speed_ratio": 0.0267,
  "is_high_torque": false,
  "is_low_speed": false,
  "is_extreme_wear": false
}
```

The API returns not just the risk score but all engineered feature values. This allows the dashboard to surface the model's internal reasoning — showing engineers exactly which operational conditions are driving the risk.

### CORS Configuration

A `@filter cors` hook handles browser pre-flight `OPTIONS` requests, allowing the static HTML dashboard to call the API from any origin without cross-origin errors:

```r
res$setHeader("Access-Control-Allow-Origin", "*")
res$setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
```

### Starting the API

```r
library(plumber)
pr("Plumber_API_for_Model_1.R") %>% pr_run(port = 19894)
```

The API will be live at `http://127.0.0.1:19894`.

---

## 🖥️ Interactive Web Dashboard

**File:** `index_1.html`

The dashboard is a **single self-contained HTML file** — no server, no framework install, no build step required. Open it directly in any modern browser. It is built with **Tailwind CSS** and vanilla JavaScript on a dark-mode UI with gradient accents, smooth CSS animations, staggered card fade-ins, and a consistent three-tier color risk system.

### How to Launch

1. Start the Plumber API (see [Quickstart](#-quickstart))
2. Open `index_1.html` in Chrome, Firefox, or Edge
3. Adjust sliders and click **Get Prediction**

---

### Dashboard Panels Explained

The interface is divided into a **left input column** and a **right results area** containing four distinct panels.

---

#### 🎛️ Left Column — Machine Parameters

The input panel where engineers enter current operational readings for the machine under assessment.

| Input | Control Type | Range |
|---|---|---|
| **Machine Type** | Dropdown | L, M, or H |
| **Air Temperature (°C)** | Slider with live readout | 15 – 35°C |
| **Process Temperature (°C)** | Slider with live readout | 25 – 45°C |
| **Rotational Speed (rpm)** | Slider with live readout | 1,100 – 3,000 rpm |
| **Torque (Nm)** | Slider with live readout | 3 – 80 Nm |
| **Tool Wear (minutes)** | Slider with live readout | 0 – 270 min |

Each slider displays its current value in real time as the user drags it. The gradient **Get Prediction** button submits all values to the API and updates all four output panels simultaneously. During the API call, the button shows an animated spinner and is disabled to prevent duplicate requests. A red error box appears automatically if the API is unreachable.

---

#### 📊 Panel 1 — Failure Risk Gauge

A custom **SVG arc gauge** displays the failure probability as a percentage (0–100%). The gauge arc animates smoothly to the new value on each prediction via CSS transitions (`transition: stroke-dashoffset 0.5s ease-in-out`). Both the arc stroke color and the central percentage text transition through the risk color system:

| Risk Zone | Color | Visual State |
|---|---|---|
| 0–29% | 🟢 Green (`#22c55e`) | Short arc, green text |
| 30–69% | 🟡 Yellow (`#eab308`) | Medium arc, yellow text |
| 70–100% | 🔴 Red (`#ef4444`) | Near-full arc, red text |

---

#### 🚦 Panel 2 — Recommendation Card

Adjacent to the gauge, this card translates the raw risk percentage into a **human-readable operational recommendation** with a contextual SVG icon and actionable guidance text. The card's top accent border color matches the active risk zone.

| Risk Level | Icon | Header | Sub-text |
|---|---|---|---|
| **Low** (< 30%) | ✅ Green checkmark circle | *All Systems Normal* | "Machine operating within safe parameters." |
| **Warning** (30–69%) | ⚠️ Yellow triangle | *Warning: Schedule Inspection* | "Model detects moderate risk. Check for high torque or tool wear." |
| **Danger** (≥ 70%) | 🚫 Red stop circle | *DANGER: Stop Machine* | "HIGH RISK. Investigate high torque, power, and tool wear immediately." |

---

#### 🔍 Panel 3 — Key Risk Drivers

This panel surfaces the **four strongest predictors of failure** with values populated from the API's JSON response. Values that breach their risk threshold **animate with a red pulsing glow** (`@keyframes pulse-red`) to draw the engineer's attention to the specific root cause driving the risk score.

| Driver | Value Source | Highlights Red When |
|---|---|---|
| **Torque (Nm)** | Raw slider input | `is_high_torque = TRUE` AND overall risk > 30% |
| **Tool Wear (min)** | Raw slider input | `is_extreme_wear = TRUE` AND overall risk > 30% |
| **Power (kW)** | Computed by API → `power_kw` | `power_kw > 60` AND overall risk > 30% |
| **Torque/Speed Ratio** | Computed by API → `torque_speed_ratio` | `ratio > 0.05` AND overall risk > 30% |

This panel answers the *"why"* behind the risk score — the engineer sees not just that a machine is at risk, but which specific parameters are responsible.

---

#### 🧮 Panel 4 — Sensor Readout (Engineered Features)

Displays the three continuous engineered features computed server-side, giving engineers visibility into derived physics values rather than raw inputs alone:

| Feature | Color | Formula |
|---|---|---|
| **Power (kW)** | Blue | `Torque × RPM / 1000` |
| **Temperature Difference (K)** | Yellow | `Process Temp − Air Temp` |
| **Torque / Speed Ratio** | Indigo | `Torque / RPM` |

All panels reset to `"..."` placeholder state if an API error occurs, with a red error banner showing the connection status and troubleshooting hint.

---

### Risk Level System

A consistent three-tier color system governs the entire dashboard UI, applied simultaneously across the gauge, recommendation card, and key driver highlights:

```
< 30%   →  GREEN   →  All Systems Normal     →  No action required
30–70%  →  YELLOW  →  Warning                →  Schedule preventive inspection
≥ 70%   →  RED     →  DANGER: Stop Machine   →  Immediate investigation required
```

---

## 📁 Project Structure

```
PredictiveMaintenance.AI/
│
├── AI4I_Production_Pipeline_2.R    # Full training pipeline — run this first
├── Plumber_API_for_Model_1.R       # Plumber REST API — run second
├── app_1.R                         # Alternative API entry point (identical logic)
├── index_1.html                    # Self-contained web dashboard — open in browser
│
├── ai4i2020.csv                    # Source dataset (10,000 observations, 14 columns)
│
├── best_model.rds                  # [GENERATED] Serialized XGBoost champion model
├── dmy_transformer.rds             # [GENERATED] One-hot encoder (dummyVars object)
├── preproc_params.rds              # [GENERATED] Training-set quantile thresholds
│
└── README.md
```

> The three `.rds` files are generated by `AI4I_Production_Pipeline_2.R`. They must exist in the same directory as the API script before the API or dashboard will function.

---

## ⚡ Quickstart

### Step 1 — Install R Dependencies

```r
install.packages(c(
  "tidyverse", "caret", "rpart", "janitor",
  "glmnet", "ranger", "xgboost", "doParallel",
  "pROC", "plumber"
))
```

### Step 2 — Run the Training Pipeline

```r
source("AI4I_Production_Pipeline_2.R")
```

Expected terminal output:
```
✅ Registered parallel backend with N cores.
✅ Successfully saved 'preproc_params.rds'
✅ Successfully saved 'dmy_transformer.rds'
🏆 BEST MODEL (by F1-Score) IS: XGBoost
Successfully saved 'best_model.rds'
--- PIPELINE COMPLETE ---
```

> ⏱️ **Expected runtime:** 5–15 minutes depending on hardware, due to the repeated cross-validation grid search across three model families.

### Step 3 — Start the Plumber API

Open a **new** R session or terminal in the project directory:

```r
library(plumber)
pr("Plumber_API_for_Model_1.R") %>% pr_run(port = 19894)
```

Expected output:
```
Loading model...
Loading preprocessor...
Loading preprocessor parameters...
API models loaded. Ready to listen.
Running plumber API at http://127.0.0.1:19894
```

### Step 4 — Open the Dashboard

```bash
# macOS
open index_1.html

# Linux
xdg-open index_1.html

# Windows — double-click the file, or:
start index_1.html
```

### Step 5 — Get a Prediction

1. Select a Machine Type from the dropdown (L, M, or H)
2. Drag the sliders to reflect current sensor readings
3. Click **Get Prediction**
4. All four panels update simultaneously with risk score, recommendation, driver flags, and sensor readout

### Test via cURL

```bash
# Example: High-risk scenario — high torque, extreme wear, low speed
curl -X POST \
  "http://127.0.0.1:19894/predict?type=L&air_temp=28&process_temp=40&rpm=1200&torque=65&wear=220"
```

Expected response:
```json
[{"risk":87.3,"power_kw":78.0,"temp_diff_k":12.0,"torque_speed_ratio":0.0542,
  "is_high_torque":true,"is_low_speed":true,"is_extreme_wear":true}]
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tidyverse` | Data wrangling, `ggplot2` visualizations, `tibble` / `dplyr` |
| `caret` | Unified ML interface, `trainControl`, `confusionMatrix`, `dummyVars` |
| `xgboost` | Gradient-boosted tree champion model |
| `glmnet` | Regularized logistic regression baseline |
| `rpart` | Decision tree baseline |
| `ranger` | Fast Random Forest (loaded in API for package compatibility) |
| `doParallel` | Parallel backend — speeds up cross-validation |
| `pROC` | ROC-AUC calculation and curve generation |
| `janitor` | Column name normalization via `clean_names()` |
| `plumber` | REST API framework for R |

---

## 🗃️ Serialized Artifacts

Three `.rds` files bridge the training pipeline and the live API:

| File | Contents | Consumed By |
|---|---|---|
| `best_model.rds` | Full `caret` train object — XGBoost model weights, tuned hyperparameters, CV history | API `/predict` — final prediction step |
| `dmy_transformer.rds` | `dummyVars` encoder fitted on training features — encodes `type` and `tool_wear_category` | API — one-hot encoding of incoming inputs |
| `preproc_params.rds` | Named list: `q_torque_95`, `q_speed_05`, `q_wear_95` — training-set percentile thresholds | API — binary flag computation in `create_feature_df()` |

Together, these three files guarantee **zero training-serving skew** — the exact same feature transformations applied during training are identically replicated at prediction time.

---

## 🧠 Key Design Decisions

**Why chronological train-test split instead of random?**
Machine operational data is a time series. A random split allows the model to train on future observations and validate on past ones — look-ahead bias that inflates evaluation metrics. A chronological split at row 7,000 mirrors real deployment: predictions are always made on data the model has never seen.

**Why persist quantile thresholds in `preproc_params.rds`?**
The binary flags `is_high_torque`, `is_low_speed`, and `is_extreme_wear` are defined relative to training-set percentiles. If the API recomputed these thresholds on incoming live data, they would drift over time and produce different feature values than what the model was trained on — a form of silent production degradation. Loading fixed training-set thresholds at API startup guarantees consistency.

**Why optimise for Sensitivity during cross-validation?**
In predictive maintenance, a **missed failure** (false negative) costs far more than a **false alarm** (false positive). A missed failure may result in unplanned downtime, safety incidents, or catastrophic equipment damage. Optimising for Sensitivity during training biases the model toward catching failures, accepting some additional false alarms.

**Why use F1-Score for final model selection rather than Sensitivity?**
A model can trivially achieve 100% sensitivity by flagging every machine as failing — it becomes useless. F1-Score requires both precision and recall to be high simultaneously. It is the correct champion-selection metric for severely imbalanced classification where the minority class is the one that matters.

**Why does the API return engineered features in its JSON response?**
The dashboard populates the Key Risk Drivers and Sensor Readout panels using server-computed values (`power_kw`, `temp_diff_k`, `torque_speed_ratio`, `is_high_torque`, etc.). This keeps feature engineering logic in a single place — the API — rather than duplicating it in client-side JavaScript. It is the single source of truth for all derived quantities.

**Why are temperatures entered in Celsius in the dashboard but stored in Kelvin?**
The model was trained on Kelvin values (the raw dataset format). The API transparently handles the conversion (`+ 273.15`) so that engineers interact with the more intuitive Celsius scale without needing to understand the internal data format.

---

## ⚠️ Limitations & Future Work

### Current Limitations

- **Single facility dataset** — trained on one manufacturing environment; performance on machines from other facilities requires validation before production deployment
- **No temporal degradation modelling** — each observation is treated independently; the model does not capture the trajectory or rate of change of sensor readings over time
- **Static thresholds** — quantile thresholds for binary flags are fixed at training time and do not adapt as operational baselines shift

### Future Enhancements

- [ ] **Per-failure-type models** — train separate classifiers for each of the five failure modes (HDF, OSF, PWF, TWF, RNF) to provide mechanism-specific early warnings
- [ ] **Temporal feature engineering** — add rolling window statistics (mean, standard deviation, rate of change) over configurable intervals to capture degradation trajectories
- [ ] **SHAP explainability** — include per-prediction SHAP values in the API response to power a feature attribution panel in the dashboard
- [ ] **Online learning** — implement continuous model retraining as new labeled failure events accumulate, preventing drift over the production lifecycle
- [ ] **Multi-facility validation** — test generalizability on publicly available datasets from other manufacturing environments
- [ ] **Dockerized deployment** — containerize the Plumber API for cloud deployment (AWS ECS, Google Cloud Run, Azure Container Instances)
- [ ] **Threshold calibration UI** — allow engineers to tune the Green / Yellow / Red risk thresholds in the dashboard to match their specific cost-of-downtime vs. cost-of-inspection tradeoff
- [ ] **Batch scoring endpoint** — add a `POST /predict_batch` endpoint accepting a CSV of machine readings for fleet-wide risk assessment

---

## 📚 References

1. Matzka, S. (2020). *Explainable Artificial Intelligence for Predictive Maintenance: A Digital Twin case study*. ICML Workshop on Explainable AI for Earth and Environment.

2. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794).

3. He, H., & Garcia, E. A. (2009). *Learning from imbalanced data*. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.

4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.

5. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why should I trust you?": Explaining the predictions of any classifier*. Proceedings of the 22nd ACM SIGKDD Conference (pp. 1135–1144).

6. NIST. (2018). *Framework for Cybersecurity and Maintenance in Manufacturing*. NIST SP 800-171r2.

---

<div align="center">

<br/>

Built with R · XGBoost · Plumber · Tailwind CSS

*"From sensor data to maintenance decision — in milliseconds."*

<br/>

</div>
