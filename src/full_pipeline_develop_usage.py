# ===============================
# Imports
# ===============================
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import joblib
from joblib import dump

from matplotlib import colormaps

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer




# ===============================
# Feature Engineering
# ===============================
def add_features(df):
    df_model = df.copy()

    # AGE_RANGE → numeric midpoint
    
    
    def age_midpoint(age_range):
        try:
            s = str(age_range).strip().lower()
            if "-" in s:  # e.g. "18-24"
                lo, hi = map(int, re.findall(r"\d+", s))
                return (lo + hi) / 2
            elif "to" in s:  # e.g. "18 to 24"
                lo, hi = map(int, re.findall(r"\d+", s))
                return (lo + hi) / 2
            elif "+" in s:  # e.g. "65+"
                return int(re.findall(r"\d+", s)[0])
            elif s.isdigit():  # single age
                return int(s)
            else:
                return np.nan
        except:
            return np.nan

    df_model["AGE_NUM"] = df_model["AGE_RANGE"].apply(age_midpoint)
    # SEX → One-hot
    df_model = pd.get_dummies(df_model, columns=["SEX"], prefix="SEX", drop_first=True)

    # Domestic flag
    df_model["IS_DOMESTIC"] = df_model["STUDENT_INT_DOM"].map(lambda x: 1 if str(x).lower().startswith("dom") else 0)

    # Flags
    df_model["HAS_SCHOLARSHIP"] = df_model["SCHOLARSHIP_IND"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["HAS_DISABILITY"] = df_model["DISABILITY_IND"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["IS_CURRENT_STUDENT"] = df_model["CURRENT_STUDENT_IND"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["IS_RESEARCH_STUDENT"] = df_model["RESEARCH_STUDENT_IND"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["IS_AUS_CAMPUS"] = df_model["AUSTRALIAN_CAMPUS_FLAG"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["IS_ATSI"] = df_model[["ABORIG_TORRES", "ATSI_IND"]].apply(lambda row: 1 if "Y" in row.values else 0, axis=1)

    # Faculty grouping
    top_faculties = df_model["FACULTY_DESCRIPTION"].value_counts().nlargest(5).index
    df_model["FACULTY_GROUPED"] = df_model["FACULTY_DESCRIPTION"].apply(lambda x: x if x in top_faculties else "Other")
    df_model = pd.get_dummies(df_model, columns=["FACULTY_GROUPED"], prefix="FACULTY", drop_first=True)

    # Course flags and grouping
    df_model["IS_RESEARCH_COURSE"] = df_model["RESEARCH_COURSE_FLAG"].map(lambda x: 1 if str(x) == "Y" else 0)
    df_model["IS_UCC_COURSE"] = df_model["UCC_COURSE_FLAG"].map(lambda x: 1 if str(x) == "Y" else 0)
    top_courses = df_model["COURSE_CD_FULL_ALT"].value_counts().nlargest(10).index
    df_model["COURSE_GROUPED"] = df_model["COURSE_CD_FULL_ALT"].apply(lambda x: x if x in top_courses else "Other")
    df_model = pd.get_dummies(df_model, columns=["COURSE_GROUPED"], prefix="COURSE", drop_first=True)

    # Course status
    if "COURSE_STATUS" in df_model.columns:
        df_model = pd.get_dummies(df_model, columns=["COURSE_STATUS"], prefix="CSTATUS", drop_first=True)

    # Engagement features
    df_model["ADM_ACTION_DATE"] = pd.to_datetime(df_model["ADM_ACTION_DATE"], errors="coerce")
    first_action = df_model.groupby("PERSONID")["ADM_ACTION_DATE"].transform("min")
    df_model["DAYS_TO_ENROLL"] = (df_model["ADM_ACTION_DATE"] - first_action).dt.days
    course_counts = df.groupby(["PERSONID", "COURSE_CD_FULL_ALT"]).size().reset_index(name="COURSE_PREF_COUNT")
    df_model = df_model.merge(course_counts, on=["PERSONID", "COURSE_CD_FULL_ALT"], how="left")
    last_action = df_model.groupby("PERSONID")["ADM_ACTION_DATE"].transform("max")
    df_model["RECENCY_DAYS"] = (last_action - df_model["ADM_ACTION_DATE"]).dt.days
    df_model["AVG_ACTION_GAP"] = df_model.groupby("PERSONID")["ADM_ACTION_DATE"].transform(
        lambda x: x.diff().mean().days if x.nunique() > 1 else 0
    )

    # Person-level target
    df_model["lead_target"] = df_model["ADM_ACTION_DETAIL_GROUP"].str.upper().isin(["ENROL", "ACCEPT"]).astype(int)
    df_target = df_model.groupby("PERSONID", as_index=False)["lead_target"].max()
    df_model = df_model.drop(columns=["lead_target"]).merge(df_target, on="PERSONID", how="left")

    return df_model

# ===============================
# Preprocessing
# ===============================


def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=['Unnamed: 0', 'PERSONID', 'COURSE_CD', 'COURSE_VERSION',
                          'ADM_ACTION_DETAIL_DESCRIPTION', 'HOME_COUNTRY',
                          'RES_AUS_POSTCODE', 'MAILING_POSTCODE'], errors='ignore')
    
    df = df.drop_duplicates().reset_index(drop=True)

    # ------------------------
    # Remove rows with invalid values
    # ------------------------
    invalid_condition = (
        (df['BIRTH_COUNTRY'].isin(['NO INFORMATION ON COUNTRY OF BIRTH'])) |
        (df['STUDENT_INT_DOM'] == 'Not Specified') |
        (df['FEE_PAYING_GROUP_DESCRIPTION'] == '<<Unknown>>') |
        (df['LOCATION_TYPE'] == '<<Unknown>>') |
        (df['LOCATION_DESCRIPTION'] == '<<Unknown>>') |
        (df['LOCATION_COUNTRY'] == '<<Unknown>>') |        
        (df['DEEWR_CITIZENSHIP'].isin([
            'NO INFORMATION ON STUDENT CITIZENSHIP', 'Not Specified'
        ])) |
        (df['ABORIG_TORRES'] == 'Not Specified')
    )
    df = df[~invalid_condition].copy()
    # -------------------------
    # Keep only valid COURSE_TYPE_GROUP1_BKEY values
    # -------------------------
    valid_course_types = ["PG", "UG", "NAWD"]
    df['COURSE_TYPE_GROUP1_BKEY'] = df['COURSE_TYPE_GROUP1_BKEY'].astype(str).str.strip().str.upper()
    df = df[df['COURSE_TYPE_GROUP1_BKEY'].isin(valid_course_types)].reset_index(drop=True)

    # -------------------------
    # Drop rows with missing AGE_NUM
    # -------------------------
    if 'AGE_NUM' in df.columns:
        df = df.dropna(subset=['AGE_NUM']).reset_index(drop=True)
        
    # -------------------------
    # Prepare features and target
    # -------------------------
    X = df.drop(columns=['ADM_ACTION_DETAIL_GROUP', 'lead_target'], errors='ignore')
    y = df['lead_target']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features),
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features)
        ]
    )

    return X, y, preprocessor



# ===============================
# Model Definitions
# ===============================
def get_models(preprocessor, categorical_features, y_train):
    preprocessor_nb = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    ratio = (len(y_train)-sum(y_train))/sum(y_train)
    pca = PCA(n_components=50, random_state=42)

    models = {
        'Logistic Regression': Pipeline([('preprocessor', preprocessor),
                                         ('classifier', LogisticRegression(max_iter=1000,class_weight='balanced'))]),
        'Random Forest': Pipeline([('preprocessor', preprocessor),
                                   ('classifier', RandomForestClassifier(n_estimators=100,max_depth=15,
                                                                         min_samples_split=10,random_state=42,
                                                                         n_jobs=-1,class_weight='balanced'))]),
        'XGBoost': Pipeline([('preprocessor', preprocessor),
                             ('classifier', XGBClassifier(n_estimators=300,learning_rate=0.1,
                                                         max_depth=5,subsample=0.8,colsample_bytree=0.8,
                                                         scale_pos_weight=ratio,eval_metric='auc',
                                                         random_state=42,n_jobs=-1))]),
        'Neural Network (PCA)': Pipeline([('preprocessor', preprocessor),('pca',pca),
                                          ('classifier', MLPClassifier(hidden_layer_sizes=(50,),
                                                                       max_iter=200,random_state=42,
                                                                       early_stopping=True,n_iter_no_change=10,
                                                                       learning_rate_init=0.01,verbose=False))]),
        'Naive Bayes': Pipeline([('preprocessor', preprocessor_nb),
                                 ('classifier', MultinomialNB())])
    }

    param_grids = {
        'XGBoost': {'classifier__n_estimators':[100,200,300],'classifier__max_depth':[3,5,7],
                    'classifier__learning_rate':[0.01,0.1,0.2],'classifier__subsample':[0.8,1.0],
                    'classifier__colsample_bytree':[0.8,1.0]}
    }

    return models, param_grids




# ===============================
# Evaluation & Scoring
# ===============================
def evaluate_and_score(model, X_train, X_test, y_train, y_test):
    """
    Train the model, predict on X_test, calculate performance metrics,
    assign dynamic Hot/Warm/Cold categories to lead scores, and return
    results and scored DataFrame.
    """
    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Performance metrics
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0)
    }
    if y_prob is not None:
        results['ROC_AUC'] = roc_auc_score(y_test, y_prob)

    # Prepare scored DataFrame
    scored_df = X_test.copy()
    scored_df['Actual'] = y_test.values
    scored_df['Predicted'] = y_pred
    scored_df['Lead Score'] = (y_prob * 100).round(2) if y_prob is not None else 0

    # -----------------------------
    # Dynamic Hot/Warm/Cold categorization
    # -----------------------------
    hot_threshold = np.percentile(scored_df['Lead Score'], 90)  # top 10%
    warm_threshold = np.percentile(scored_df['Lead Score'], 50)  # median

    def categorize_lead(score, hot_threshold=hot_threshold, warm_threshold=warm_threshold):
        if score >= hot_threshold:
            return "Hot"
        elif score >= warm_threshold:
            return "Warm"
        else:
            return "Cold"

    scored_df['Category'] = scored_df['Lead Score'].apply(lambda x: categorize_lead(x, hot_threshold, warm_threshold))

    return results, scored_df


# ===============================
# Plotting Functions
# ===============================
def save_plot(fig, filename, results_dir):
    path = os.path.join(results_dir,'plots',filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"✔ Saved plot → {path}")

def plot_metrics(results_df, results_dir):
    metrics_to_plot = ['Accuracy','Precision','Recall','F1','ROC_AUC']
    fig, ax = plt.subplots(figsize=(12,6))
    results_df[metrics_to_plot].plot(kind='bar',ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.set_xticklabels(results_df.index,rotation=45,ha='right')
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.show()
    save_plot(fig,'metrics_comparison.png',results_dir)

def plot_roc_curves(models,X_test,y_test,results_dir):
    fig, ax = plt.subplots(figsize=(10,7))
    for name, model in models.items():
        try:
            if hasattr(model,"predict_proba"):
                y_prob = model.predict_proba(X_test)[:,1]
                fpr,tpr,_ = roc_curve(y_test,y_prob)
                roc_auc = auc(fpr,tpr)
                ax.plot(fpr,tpr,label=f"{name} (AUC={roc_auc:.2f})")
        except: continue
    ax.plot([0,1],[0,1],'k--',lw=2)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.show()
    save_plot(fig,'roc_curves.png',results_dir)

def plot_confusion_matrix(model,X_test,y_test,model_name,results_dir):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    labels = ['Not Converted (0)','Converted (1)']
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    plt.show()
    save_plot(fig,f'confusion_matrix_{model_name.replace(" ","_")}.png',results_dir)


# ===============================
# Feature Importance
# ===============================
def plot_feature_importance(model, preprocessor, X, model_name="Model", results_dir="results", save_csv=False):
    try:
        # Get feature names
        cat_features = preprocessor.transformers_[0][1].get_feature_names_out(
            preprocessor.transformers_[0][2]
        )
        num_features = preprocessor.transformers_[1][2]
        feature_names = np.concatenate([cat_features, num_features])

        # Get importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            print(f"  {model_name} does not provide feature importances.")
            return None

        # Full DataFrame (all features, not just top-N)
        fi_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by="Importance", ascending=False)

        # --- Save CSV with all features ---
        if save_csv:
            csv_path = os.path.join(results_dir, f'feature_importance_{model_name.replace(" ","_")}.csv')
            fi_df.to_csv(csv_path, index=False)
            print(f"✔ Saved feature importance CSV → {csv_path}")

        # --- Plot only top 15 features ---
        top_fi = fi_df.head(15)
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = colormaps.get_cmap("viridis")
        colors = [cmap(i / len(top_fi)) for i in range(len(top_fi))]

        sns.barplot(
            x="Importance",
            y="Feature",
            data=top_fi,
            palette=colors,
            ax=ax,
            dodge=False,
            legend=False
        )

        ax.set_title(f"Top 15 Features - {model_name}")
        fig.tight_layout()
        plt.show()
        save_plot(fig, f'feature_importance_{model_name.replace(" ","_")}.png', results_dir)        
        plt.close(fig)

        return fi_df
    except Exception as e:
        print(f"  Could not plot feature importance for {model_name}: {e}")
        return None


# ===============================
# Rank Top Models
# ===============================
def rank_top_models(results_df, metric='F1', top_n=3):
    ranked = results_df.sort_values(by=metric, ascending=False).head(top_n)
    print(f"\nTop {top_n} Models (ranked by {metric}):")
    print(ranked)
    return ranked

# ===============================
# Main Execution
# ===============================
if __name__ == "__main__":
    # -----------------------------
    # Determine mode
    # -----------------------------
    default_mode = "usage"  # fallback if no command-line argument

    try:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", default=default_mode, help="Run mode: develop or usage")
        args, unknown = parser.parse_known_args()
        mode = args.mode
    except Exception:
        # fallback if argparse fails (e.g., Jupyter or VSCode interactive run)
        mode = default_mode

    print(f"Running in {mode} mode")
    

    # Get absolute path relative to project root
    base_dir = os.path.dirname(__file__)  # src/
    data_path = os.path.abspath(os.path.join(base_dir, "..", "data", "Anonymised - 20250925_capstone_admissions.csv"))
  
    results_dir = os.path.abspath(os.path.join(base_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)  

    
  

    # Load and preprocess dataset
    #  Load dataset
    df = pd.read_csv(data_path)
    df['ADM_ACTION_DATE'] = pd.to_datetime(df['ADM_ACTION_DATE'], dayfirst=True, errors='coerce')
    df = add_features(df)  #  feature engineering function

    

    if mode == "develop":
        
        # Preprocess features
        X, y, preprocessor = preprocess_data(df)  # Returns X, y, preprocessor
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        # Get models and parameter grids
        models, param_grids = get_models(preprocessor, categorical_features, y_train)

        # GridSearch for XGBoost
        if 'XGBoost' in models:
            xgb_grid = GridSearchCV(
                models['XGBoost'],
                param_grids['XGBoost'],
                cv=3,
                scoring='f1',
                n_jobs=1
            )
            xgb_grid.fit(X_train, y_train)
            models['XGBoost'] = xgb_grid.best_estimator_
            print("✔ XGBoost best params:", xgb_grid.best_params_)

        all_results, scored_outputs, fitted_models = {}, {}, {}

        
        # Train & Evaluate all models
        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                results, scored_df = evaluate_and_score(model, X_train, X_test, y_train, y_test)
                all_results[name] = results
                scored_outputs[name] = scored_df
                fitted_models[name] = model
                print(f"{name} trained successfully.")
            except Exception as e:
                print(f"  {name} failed: {e}")
    
        results_df = pd.DataFrame(all_results).T
        print("\n Model Performance Comparison:")
        print(results_df)
    
        # Visualizations
        plot_metrics(results_df, results_dir)
        plot_roc_curves(fitted_models, X_test, y_test, results_dir)
        
        # Rank Top Models
        top_models = rank_top_models(results_df, metric='F1', top_n=3)

        # Feature Importance & Confusion Matrix only for the best model
        best_model_name = top_models.index[0]
        best_model = fitted_models[best_model_name]

        # Confusion matrix (only for best model)
        plot_confusion_matrix(best_model, X_test, y_test, best_model_name, results_dir)

        
        # Feature importance (only for best model)
        try:
            print(f"\nFeature Importance - {best_model_name}:")
            fi = plot_feature_importance(
            best_model.named_steps['classifier'],
            preprocessor,
            X,
            model_name=best_model_name,
            results_dir=results_dir
            )
            if fi is not None:
                # Save full feature importance CSV (sorted descending)
                fi_path = os.path.join(results_dir, f'feature_importance_{best_model_name.replace(" ","_")}.csv')
                fi.to_csv(fi_path, index=False)
                print(f"✔ Feature importance CSV saved → {fi_path}")

                # Print top 15 features to console
                print("\nTop 15 Features:")
                print(fi.head(15).to_string(index=False))
        except Exception as e:
            print(f" Skipping feature importance for {best_model_name}: {e}")



        # Ensure model folder exists inside project root
        # Handle Jupyter/Notebook vs Script execution
        try:
            base_dir = os.path.dirname(__file__)
        except NameError:
            base_dir = os.getcwd()   # fallback if __file__ not defined

        model_dir = os.path.join(base_dir, "..", "model")
        os.makedirs(model_dir, exist_ok=True)


        # Save best model
        best_model_path = os.path.join(model_dir, "best_model_20250925_capstone_admissions.pkl")
        dump(best_model, best_model_path)

        print(f"✔ Best model ({best_model_name}) saved → {best_model_path}")

    
        
    elif mode == "usage":
        

        # -----------------------------
        # Determine base directories
        # -----------------------------
        try:
            base_dir = os.path.dirname(__file__)
        except NameError:
            base_dir = os.getcwd()

        model_dir = os.path.join(base_dir, "..", "model")
        results_dir = os.path.join(base_dir, "..", "results")
        plots_dir = os.path.join(results_dir, "plots")

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # -----------------------------
        # Load trained best model
        # -----------------------------
        best_model_path = os.path.join(model_dir, "best_model_20250925_capstone_admissions.pkl")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"❌ Model not found at: {best_model_path}")

        best_model = joblib.load(best_model_path)
        print(f"✅ Loaded best model from: {best_model_path}")

        # -----------------------------
        # Prepare features for scoring
        # -----------------------------
        feature_cols = [col for col in df.columns if col not in ["ADM_ACTION_DETAIL_GROUP", "lead_target"]]
        X_new = df[feature_cols].copy()

        # -----------------------------
        # Predict probabilities
        # -----------------------------
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_new)[:, 1]
            predicted_lead = best_model.predict(X_new)
        elif hasattr(best_model, "decision_function"):
            y_raw = best_model.decision_function(X_new).reshape(-1, 1)
            y_prob = MinMaxScaler().fit_transform(y_raw).ravel()
            predicted_lead = (y_prob >= 0.5).astype(int)
        else:
            y_prob = best_model.predict(X_new)
            predicted_lead = (y_prob >= 0.5).astype(int)
            print("⚠️ Model does not support probabilities; using raw predictions.")

        # -----------------------------
        # Convert to lead scores (0–100)
        # -----------------------------
        df["Lead Score"] = (y_prob * 100).round(2)
        df["predicted_lead"] = predicted_lead

        # -----------------------------
        # Dynamic Hot/Warm/Cold categorization
        # -----------------------------
        hot_threshold = np.percentile(df["Lead Score"], 90)  # top 10%
        warm_threshold = np.percentile(df["Lead Score"], 50)  # median

        def categorize_dynamic(score):
            if score >= hot_threshold:
                return "Hot"
            elif score >= warm_threshold:
                return "Warm"
            else:
                return "Cold"

        df["Category"] = df["Lead Score"].apply(categorize_dynamic)

        # -----------------------------
        # Aggregate at PERSONID level
        # -----------------------------
        agg_rules = {
        "Lead Score": "mean",
        "Category": lambda x: x.mode()[0] if not x.mode().empty else "Cold",
        "predicted_lead": "max"
        }
        if "lead_target" in df.columns:
            agg_rules["lead_target"] = "max"

        person_scores = df.groupby("PERSONID", as_index=False).agg(agg_rules)

        # Reorder columns exactly as desired
        cols = ["PERSONID", "Lead Score", "Category", "lead_target", "predicted_lead"]
        person_scores = person_scores.reindex(columns=cols)

        # -----------------------------
        # Save results
        # -----------------------------
        output_csv = os.path.join(results_dir, "scored_leads_summary.csv")
        person_scores.to_csv(output_csv, index=False)
        print(f"💾 Scored leads summary saved to: {output_csv}")
        print("\n📊 Sample of summary scored leads:")
        print(person_scores.head(10).to_string(index=False))

        # -----------------------------
        # Plot Lead Score distribution
        # -----------------------------
        plt.figure(figsize=(10, 6))
        sns.histplot(df["Lead Score"], bins=30, kde=True, color="skyblue")
        plt.axvline(hot_threshold, color="red", linestyle="--", label=f"Hot Threshold ({hot_threshold:.1f})")
        plt.axvline(warm_threshold, color="orange", linestyle="--", label=f"Warm Threshold ({warm_threshold:.1f})")
        plt.title("Lead Score Distribution with Hot/Warm Thresholds")
        plt.xlabel("Lead Score")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "lead_score_distribution.png"))
        plt.show()

        # -----------------------------
        # Plot Category Count Bar
        # -----------------------------
        plt.figure(figsize=(7, 5))
        category_counts = person_scores["Category"].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
        for i, v in enumerate(category_counts.values):
            plt.text(i, v + 1, str(v), ha='center', fontweight='bold')
        plt.title("Number of Leads by Category")
        plt.xlabel("Category")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "lead_category_counts.png"))
        plt.show()

        # -----------------------------
        # Plot Category Percentage Pie
        # -----------------------------
        plt.figure(figsize=(6, 6))
        plt.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%", colors=sns.color_palette("viridis", 3))
        plt.title("Lead Category Percentage Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "lead_category_pie.png"))
        plt.show()

