# Capstone Project Team_54

This repository contains the complete **end-to-end machine learning pipeline** for the **Admissions Data Analytics Capstone Project**. It includes scripts and notebooks for **data preprocessing**, **feature engineering**, **exploratory data analysis (EDA)**, **model training**, and **SHAP-based explainability**.

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
│   ├── insights_capstone_project_team_54.py
│   ├── full_pipeline_capstone_project_team_54.ipynb
│   └── insights_capstone_project-team_54.ipynb
│
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## Running the Project

### 1️⃣ Change Directory

Open **CMD** or **Anaconda Prompt** and navigate to the `src` folder:

```bash
cd "C:\Users\<user name>\Capstone Project Team_54\src"
```

---

### 2️⃣ Run Python Scripts

#### **Development Mode**

Executes the complete machine learning pipeline — data preparation, feature engineering, model training, evaluation, and visualization.

```bash
python full_pipeline_capstone_project_team_54.py --mode develop
```

✅ **What Happens:**

* Loads the admissions dataset from `/data`
* Performs data cleaning and feature engineering
* Trains ML models (Logistic Regression, Random Forest, XGBoost, MLP, Naive Bayes)
* Evaluates performance using Accuracy, Precision, Recall, F1, and ROC_AUC
* Saves all plots and the best-performing model

---

#### **Usage Mode**

Applies the saved best model to new/unseen data and produces lead scores.

```bash
python full_pipeline_capstone_project_team_54.py --mode usage
```

✅ **What Happens:**

* Loads the trained model from `/model`
* Predicts lead probabilities and assigns categories (`Hot`, `Warm`, `Cold`)
* Saves scored results to `results/scored_leads.csv`

---

### 3️⃣ Run Insights Script (EDA + SHAP Explanations)

#### **Non-Interactive Mode (Saves Plots):**

```bash
python insights_capstone_project_team_54.py 
```


---

### 4️⃣ Run Jupyter Notebooks (Interactive Mode)

If you prefer **notebooks** for interactive analysis or presentation:

| Notebook                                       | Purpose                                                        |
| ---------------------------------------------- | -------------------------------------------------------------- |
| `insights_capstone_project-team_54.ipynb`      | Interactive EDA and SHAP visualization                         |
| `full_pipeline_capstone_project_team_54.ipynb` | Full ML pipeline (feature engineering → training → evaluation) |

#### **Launch Jupyter:**

```bash
jupyter notebook
```

Then open either notebook from:

```
src/
```

---

##  Overview of the Full Pipeline

### 🔹 Feature Engineering

* Converts categorical ranges (e.g., AGE_RANGE → AGE_NUM)
* Creates binary flags (domestic status, scholarship, disability, etc.)
* Generates time-based features (`DAYS_TO_ENROLL`, `RECENCY_DAYS`, `AVG_ACTION_GAP`)
* Defines the target variable (`lead_target` = 1 if *ENROL* or *ACCEPT*)

---

### 🔹 Preprocessing

* Handles missing values with `SimpleImputer`
* Scales numerical features with `StandardScaler`
* Encodes categorical features via `OneHotEncoder`
* Returns processed training data for modeling

---

### 🔹 Model Training

Trains multiple algorithms wrapped in Scikit-learn pipelines:

* Logistic Regression
* Random Forest
* XGBoost (GridSearch tuned)
* Multilayer Perceptron (MLP)
* Naive Bayes (categorical baseline)

Each model is compared on **F1-score**, and the best one is saved.

---

### 🔹 Evaluation & Visualization

* **Confusion Matrix** and **ROC Curves** for each model
* **Feature Importance (XGBoost & SHAP)**
* **Performance Comparison Bar Charts**
* **Ranked Model Summary**

---

### 🔹 Model Saving & Scoring

* Best-performing model → `model/best_model_20250925_capstone_admissions.pkl`
* Scored outputs → `results/scored_leads.csv`
* Feature importance table → `results/feature_importance_XGBoost.csv`

---

## 📊 Outputs

| File                                          | Description                          |
| --------------------------------------------- | ------------------------------------ |
| `feature_importance_XGBoost.csv`              | Feature importance scores            |
| `metrics_comparison.png`                      | Model performance comparison         |
| `roc_curves.png`                              | ROC curve comparison                 |
| `confusion_matrix_<Model>.png`                | Confusion matrix visualization       |
| `scored_leads.csv`                            | Predicted lead categories and scores |
| `best_model_20250925_capstone_admissions.pkl` | Serialized trained model             |

---


Install them with:

```bash
pip install -r requirements.txt
```

---

## 🧾 .gitignore

Typical `.gitignore` configuration for this project:

```
# Byte-compiled / cache
__pycache__/
*.py[cod]
*.pyo

# Data & model artifacts
/data/
*.pkl
*.csv

# Plots and results
/results/plots/
*.png

# Environment and IDE
.env
.vscode/
.ipynb_checkpoints/
.DS_Store

# Virtual environments
venv/
env/
```

---

## ⚙️ .gitattributes

Ensures consistent line endings and proper file handling across platforms:

```
# Handle line endings automatically
* text=auto

# Treat notebooks as binary to avoid merge conflicts
*.ipynb binary
```

---

## 📘 Notes

* The project supports both **script-based automation** and **interactive notebook workflows**.
* All generated outputs are stored in structured directories for traceability.
* Modular code design allows easy updates, retraining, or scaling for new datasets.

---

**Author:** Abu BHUIYAN
**University:** Canberra University — Master of Data Science
**Date:** October 2025
