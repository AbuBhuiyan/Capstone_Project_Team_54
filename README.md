# Capstone Project Team_54

This repository contains the complete data analytics and machine learning pipeline for the **Admissions Data Analytics Capstone Project**. It includes scripts for **data preprocessing**, **feature engineering**, **exploratory data analysis (EDA)**, **model training**, and **SHAP-based explainability**.

---

## 📁 Folder Structure

```
Capstone Project Team_54/
│
├── data/
│   └── Anonymised - 20250925_capstone_admissions.csv
│
├── model/
│   └── best_model_20250925_capstone_admissions.pkl
│
├── results/
│   ├── feature_importance_XGBoost.csv
│   ├── scored_leads.csv
│   ├── shap_feature_importance_table.csv
│   └── plots/
│       ├── *.png
│
├── src/
│   ├── full_pipeline_capstone_project_team_54.py
│   └── insights_capstone_project_team_54.py
│
├── README.md
└── requirements.txt
```

---
Capstone Project Team_54/
│
├── data/
│   └── Anonymised - 20250925_capstone_admissions.csv
│
├── model/
│   └── best_model_20250925_capstone_admissions.pkl
│
├── results/
│   ├── feature_importance_XGBoost.csv
│   ├── scored_leads.csv
│   ├── shap_feature_importance_table.csv
│   └── plots/
│       ├── *.png
│
├── src/
│   ├── full_pipeline_capstone_project_team_54.py
│   ├── insights_capstone_project_team_54.py
│   ├── insights_capstone_project-team_54.ipynb
│   └── full_pipeline_capstone_project_team_54.ipynb
│
├── README.md
└── requirements.txt

##  Running the Project

### 1️⃣ Change Directory

Open **CMD** and navigate to the `src` folder:

```bash
cd "C:\Users\<user name>\Capstone Project Team_54\src"
```

---

### 2️⃣ Run the Full Pipeline Script

#### **Development Mode**

Runs the complete end-to-end ML workflow — data preprocessing, feature engineering, model training, evaluation, visualization, and saving the best model.

```bash
python full_pipeline_capstone_project_team_54.py --mode develop
```

**Actions Performed:**

* Loads the admissions dataset from `data/`
* Cleans and preprocesses data
* Adds derived features (e.g., AGE_NUM, IS_DOMESTIC, etc.)
* Trains multiple ML models (Logistic Regression, Random Forest, XGBoost, MLP, Naive Bayes)
* Performs GridSearchCV tuning for XGBoost
* Evaluates models with metrics: Accuracy, Precision, Recall, F1, ROC_AUC
* Generates and saves:

  * Feature importance plots
  * ROC curves
  * Confusion matrix
  * Model performance comparison
* Saves best-performing model as `.pkl` in `model/`
* Outputs performance metrics to `results/plots/`

---

#### **Usage Mode**

Uses the trained model to score new/unseen data and generate lead scores.

```bash
python full_pipeline_capstone_project_team_54.py --mode usage
```

**Actions Performed:**

* Loads saved best model (`best_model_20250925_capstone_admissions.pkl`)
* Predicts lead scores and categories (`Hot`, `Warm`, `Cold`)
* Aggregates predictions at the `PERSONID` level
* Saves scored results to `results/scored_leads.csv`

---

### 3️⃣ Run the Insights Script Separately (Optional)

For EDA and SHAP explanations only:

```bash
python insights_capstone_project_team_54.py 
```


---

##  Overview of `full_pipeline_capstone_project_team_54.py`

### 🔹 Imports

Includes libraries for data handling, visualization, preprocessing, model training, and evaluation:

* **Core:** pandas, numpy, os, re, argparse
* **Visualization:** matplotlib, seaborn
* **ML Models:** LogisticRegression, RandomForestClassifier, XGBClassifier, MLPClassifier, MultinomialNB
* **Preprocessing:** OneHotEncoder, StandardScaler, SimpleImputer, ColumnTransformer, PCA
* **Evaluation:** accuracy, precision, recall, f1, ROC_AUC, confusion_matrix, roc_curve
* **Utilities:** joblib for saving/loading models

---

### 🔹 Feature Engineering (`add_features`)

* Converts **AGE_RANGE** to a numeric midpoint (`AGE_NUM`)
* One-hot encodes **SEX**
* Generates binary flags for:

  * Domestic status (`IS_DOMESTIC`)
  * Scholarship (`HAS_SCHOLARSHIP`)
  * Disability (`HAS_DISABILITY`)
  * Current and Research students
  * Australian campus flag (`IS_AUS_CAMPUS`)
  * Indigenous indicators (`IS_ATSI`)
* Groups faculties and courses based on frequency
* Creates time-based features: `DAYS_TO_ENROLL`, `RECENCY_DAYS`, `AVG_ACTION_GAP`
* Defines the binary target `lead_target` = 1 if *ENROL* or *ACCEPT*

---

### 🔹 Preprocessing (`preprocess_data`)

* Removes irrelevant or unknown entries
* Filters valid course types (`PG`, `UG`, `NAWD`)
* Handles missing values using `SimpleImputer`
* Standardizes numerical variables using `StandardScaler`
* One-hot encodes categorical variables

Returns:

```python
X, y, preprocessor
```

---

### 🔹 Model Training (`get_models`)

Defines multiple ML pipelines:

* Logistic Regression
* Random Forest
* XGBoost (GridSearch tuned)
* Neural Network (MLP with PCA)
* Naive Bayes (categorical-only)

Each model is wrapped in a pipeline with preprocessing steps.

---

### 🔹 Evaluation & Visualization

Functions include:

* `evaluate_and_score()` — Fits model, computes metrics, and assigns lead categories (`Hot`, `Warm`, `Cold`).
* `plot_metrics()` — Bar chart of model comparison.
* `plot_roc_curves()` — ROC-AUC visualization for all models.
* `plot_confusion_matrix()` — Confusion matrix heatmap for best model.
* `plot_feature_importance()` — Displays and saves top 15 important features.
* `rank_top_models()` — Ranks models based on F1-score.

---

### 🔹 Model Saving and Scoring

* Best-performing model is saved as:

  ```
  model/best_model_20250925_capstone_admissions.pkl
  ```
* In usage mode, predictions and lead scores are saved as:

  ```
  results/scored_leads.csv
  ```

---

## 📊 Outputs

| Output File                                   | Description                          |
| --------------------------------------------- | ------------------------------------ |
| `feature_importance_XGBoost.csv`              | Ranked feature importances           |
| `metrics_comparison.png`                      | Model performance comparison         |
| `roc_curves.png`                              | ROC-AUC visualization                |
| `confusion_matrix_<Model>.png`                | Confusion matrix for top model       |
| `scored_leads.csv`                            | Predicted lead scores and categories |
| `best_model_20250925_capstone_admissions.pkl` | Saved trained model                  |

---



---

## 📘 Notes

* The script automatically detects execution mode (`develop` or `usage`) via command-line arguments.
* All output artifacts are saved under the `results/` and `model/` folders.
* Modular functions allow reuse of preprocessing, model evaluation, and visualization components.

---

**Author:** Abu BHUIYAN
**University:** Canberra University — Master of Data Science
**Date:** October 2025
