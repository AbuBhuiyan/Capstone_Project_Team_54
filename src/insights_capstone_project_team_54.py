import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse
import sys
import os 
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ------------------------------
# Annotate bars helper
# ------------------------------
def annotate_bars(ax, fmt="{:.0f}"):
    """Add value labels on top of bars"""
    for p in ax.patches:
        height = p.get_height()
        if np.isnan(height) or height == 0:
            continue
        ax.annotate(fmt.format(height),
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=9, color="black")

# ------------------------------
# Sanitize filenames for Windows
# ------------------------------
def safe_filename(title):
    # Replace any character not a letter, number, or underscore with '_'
    filename = re.sub(r'[^A-Za-z0-9]+', '_', title.strip())
    return filename + ".png"



# ------------------------------
# Feature Engineering
# ------------------------------
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

    #  target
    df_model["lead_target"] = df_model["ADM_ACTION_DETAIL_GROUP"].str.upper().isin(["ENROL", "ACCEPT"]).astype(int)
    

    return df_model


# ------------------------------
#  Insights & plotting
# ------------------------------
def additional_insights_row(df, save_plots=False, plot_dir="plots"):
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    print("\nAdditional Insights (Row-level):")

    # ------------------------------
    # 1. Distributions
    # ------------------------------
    distributions = {
        "Age Distribution": ("AGE_NUM", "hist", 20, "Age", "Count"),
        "Disability Distribution": ("HAS_DISABILITY", "count", None, "Has Disability", "Count"),
        "Lead Target Distribution": ("lead_target", "count", None, "Lead Target", "Count"),
        "Scholarship Distribution": ("HAS_SCHOLARSHIP", "count", None, "Has Scholarship", "Count")
    }

    for title, (col, plot_type, bins, xlabel, ylabel) in distributions.items():
        if col not in df.columns:
            continue

        print(f"\n=== {title} (first 10 rows) ===")
        print(df[[col]].head(10))

        fig, ax = plt.subplots(figsize=(8, 5))
        if plot_type == "hist":
            sns.histplot(df[col].dropna(), bins=bins, kde=True, ax=ax)
        elif plot_type == "count":
            sns.countplot(x=col, data=df, palette="viridis", ax=ax)
            annotate_bars(ax, fmt="{:.0f}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if save_plots:
            plt.savefig(os.path.join(plot_dir, safe_filename(title)), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        
        

    # ------------------------------
    # 2. Conversion helper
    # ------------------------------
    def conversion_plot(df, group_col, title):
        conv = df.groupby(group_col)["lead_target"].mean() * 100
        if conv.empty:
            return

        print(f"\n=== {title} ===")
        print(conv)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=conv.index, y=conv.values, palette="viridis", ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        annotate_bars(ax, fmt="{:.1f}%")

        if save_plots:
            plt.savefig(os.path.join(plot_dir, safe_filename(title)), bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        
        
    # --- Apply conversion plots ---
    if "HAS_SCHOLARSHIP" in df.columns:
        conversion_plot(df, "HAS_SCHOLARSHIP", "Conversion by Scholarship")
    if "IS_DOMESTIC" in df.columns:
        conversion_plot(df, "IS_DOMESTIC", "Conversion: Domestic vs International")
    if "IS_RESEARCH_STUDENT" in df.columns:
        conversion_plot(df, "IS_RESEARCH_STUDENT", "Conversion by Research Students")

    # Gender
    gender_map = {"SEX_M": "Male", "SEX_U": "Unknown", "SEX_X": "Other"}
    gender_cols = [c for c in gender_map if c in df.columns]
    if gender_cols:
        df_gender = df.copy()
        df_gender["Gender"] = df_gender[gender_cols].idxmax(axis=1).map(gender_map)
        conversion_plot(df_gender, "Gender", "Conversion by Gender")

    # Faculty
    faculty_cols = [c for c in df.columns if c.startswith("FACULTY_")]
    for col in faculty_cols:
        conversion_plot(df, col, f"Conversion by {col}")

    # Recency
    if "RECENCY_DAYS" in df.columns:
        df["RECENCY_BUCKET"] = pd.cut(
            df["RECENCY_DAYS"],
            bins=[-1, 7, 30, 90, 365, 10000],
            labels=["<1w", "1w-1m", "1m-3m", "3m-1y", ">1y"]
        )
        conversion_plot(df, "RECENCY_BUCKET", "Conversion by Recency Bucket")

    # Fee paying group
    if "FEE_PAYING_GROUP_DESCRIPTION" in df.columns:
        fee_summary = df.groupby("FEE_PAYING_GROUP_DESCRIPTION", observed=True)["lead_target"].agg(
            count="count", conversions="sum").reset_index()
        if not fee_summary.empty:
            fee_summary["conversion_rate"] = (fee_summary["conversions"] / fee_summary["count"]) * 100
            print("\n=== Conversion by Fee Paying Group ===")
            print(fee_summary)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="FEE_PAYING_GROUP_DESCRIPTION", y="conversion_rate", data=fee_summary,
                        palette="viridis", ax=ax)
            ax.set_title("Conversion by Fee Paying Group")
            ax.set_ylabel("Conversion Rate (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
            annotate_bars(ax, fmt="{:.1f}%")

            if save_plots:
                plt.savefig(os.path.join(plot_dir, safe_filename("Conversion by Fee Paying Group")), bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        


    # Domestic breakdown
    if "IS_DOMESTIC" in df.columns and "FEE_PAYING_GROUP_DESCRIPTION" in df.columns:
        df_domestic_fee = df[
            (df["IS_DOMESTIC"] == 1) & 
            (df["FEE_PAYING_GROUP_DESCRIPTION"].isin([
                "Commonwealth Supported Places (CSP)", "Domestic Fee-Paying"
            ]))
        ]
        if not df_domestic_fee.empty:
            fee_conv = df_domestic_fee.groupby("FEE_PAYING_GROUP_DESCRIPTION", observed=True)["lead_target"].agg(
                count="count", conversions="sum").reset_index()
            fee_conv["conversion_rate"] = (fee_conv["conversions"] / fee_conv["count"]) * 100
            print("\n=== Conversion Rate: Domestic Students by Fee Paying Type ===")
            print(fee_conv)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="FEE_PAYING_GROUP_DESCRIPTION", y="conversion_rate", data=fee_conv,
                        palette="viridis", ax=ax)
            ax.set_title("Conversion Rate: Domestic Students by Fee Paying Type")
            ax.set_ylabel("Conversion Rate (%)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
            annotate_bars(ax, fmt="{:.1f}%")

            if save_plots:
                plt.savefig(os.path.join(plot_dir, safe_filename("Conversion Rate: Domestic Students by Fee Paying Type")), bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        


    # Conversion by Course Type
    if "COURSE_TYPE_GROUP1_BKEY" in df.columns:
        df["COURSE_TYPE_GROUP1_BKEY"] = df["COURSE_TYPE_GROUP1_BKEY"].fillna("NAWD")
        course_order = ["UG", "PG", "NAWD"]

        course_counts = df.groupby(
            ["COURSE_TYPE_GROUP1_BKEY", "lead_target"], observed=True
        ).size().reset_index(name="count")

        full_index = pd.MultiIndex.from_product([course_order, [0, 1]],
                                                names=["COURSE_TYPE_GROUP1_BKEY", "lead_target"])
        course_counts = course_counts.set_index(["COURSE_TYPE_GROUP1_BKEY", "lead_target"]) \
                                     .reindex(full_index, fill_value=0).reset_index()

        if not course_counts.empty:
            print("\n=== Conversion Counts by Course Type (UG/PG/NAWD) ===")
            print(course_counts)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x="COURSE_TYPE_GROUP1_BKEY", y="count", hue="lead_target",
                        data=course_counts, palette="viridis", ax=ax)
            ax.set_title("Conversion Counts by Course Type (UG/PG/NAWD)")
            ax.set_ylabel("Number of Students")
            ax.set_xlabel("Course Type")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
            annotate_bars(ax, fmt="{:.0f}")
            ax.legend(title="Converted", labels=["No (0)", "Yes (1)"])

            if save_plots:
                plt.savefig(os.path.join(plot_dir, safe_filename("Conversion_By_Course_Type")), bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
        

def train_best_model(df):
    # Drop target
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

    # Keep only numeric/boolean/categorical (ignore raw strings, dates, IDs)
    allowed_types = ["int16", "int32", "int64", "float16", "float32", "float64", "bool"]
    X = X.select_dtypes(include=allowed_types)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

   
    
    # Train XGBoost with best params
    model = xgb.XGBClassifier(
        colsample_bytree=1.0,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=300,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test  



# ------------------------------
# Run SHAP analysis
# ------------------------------
def shap_insights(model, X_train, X_test, save_plots=True, save_dir="../results/plots"):
    os.makedirs(save_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ------------------------------
    #  Compute SHAP importance table
    # ------------------------------
    shap_importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean |SHAP Value|": np.abs(shap_values).mean(axis=0)
    }).sort_values("Mean |SHAP Value|", ascending=False)

    print("\n=== SHAP Feature Importance (Top 20) ===")
    print(shap_importance.head(20))

    shap_table_path = os.path.join(os.path.dirname(save_dir), "shap_feature_importance_table.csv")
    shap_importance.to_csv(shap_table_path, index=False)
    print(f" Saved SHAP importance table → {shap_table_path}")

    # ------------------------------
    # Global importance (bar)
    # ------------------------------
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "shap_feature_importance.png"), bbox_inches="tight", dpi=200)
        print(f" Saved SHAP bar plot → {save_dir}/shap_feature_importance.png")
        plt.close()
    else:
        plt.show()

    # ------------------------------
    # Beeswarm (distribution of impact)
    # ------------------------------
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_dir, "shap_beeswarm.png"), bbox_inches="tight", dpi=200)
        print(f" Saved SHAP beeswarm plot → {save_dir}/shap_beeswarm.png")
        plt.close()
    else:
        plt.show()

    # ------------------------------
    # Example local explanation for first row
    # ------------------------------
    idx = 0
    log_odds = explainer.expected_value + shap_values[idx, :].sum()
    prob = 1 / (1 + np.exp(-log_odds))
    print(f"f(x) = {log_odds:.2f}, Probability = {prob:.3%}")

    abs_shap = np.abs(shap_values[idx, :])
    top_idx = np.argsort(abs_shap)[-6:]  # top 6 features

    # Get force plot as a matplotlib figure
    fig = shap.force_plot(
        explainer.expected_value,
        shap_values[idx, top_idx],
        X_test.iloc[idx, top_idx],
        matplotlib=True,
        show=False
    )

    if save_plots:
        fig.savefig(os.path.join(save_dir, "shap_force_row0_top10.png"), bbox_inches="tight", dpi=200)
        print(f" Saved SHAP force plot (Top 10 features) → {save_dir}/shap_force_row0_top10.png")
        plt.close(fig)
    else:
        plt.show()

# ------------------------------
# Main function
# ------------------------------
def main(args=None):
    parser = argparse.ArgumentParser(description="Generate insights and SHAP plots from admissions CSV")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--save", action="store_true", help="Save plots instead of showing them")
    parser.add_argument("--plot_dir", default="../results/plots", help="Directory to save plots")

    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Ensure output folder exists
    os.makedirs(args.plot_dir, exist_ok=True)

    # ------------------------------
    # Load dataset
    # ------------------------------
    df = pd.read_csv(args.csv)
    print(f"Loaded CSV with shape: {df.shape}")

    # ------------------------------
    # Add engineered features + EDA
    # ------------------------------
    df = add_features(df)
    additional_insights_row(df, save_plots=args.save, plot_dir=args.plot_dir)

    # ------------------------------
    # Train model
    # ------------------------------
    model, X_train, X_test, y_train, y_test = train_best_model(df)

    # ------------------------------
    # Run SHAP insights (show or save)
    # ------------------------------
    shap_insights(model, X_train, X_test, save_plots=args.save, save_dir=args.plot_dir)

    print("\n✅ Done: All plots have been " + ("saved" if args.save else "displayed") + ".")


# ------------------------------ 
# Run script
# ------------------------------
if __name__ == "__main__":
    # Define base folders
    try:
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except NameError:
        BASE_DIR = os.getcwd()

    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    csv_file = "Anonymised - 20250925_capstone_admissions.csv"
    csv_path = os.path.join(DATA_DIR, csv_file)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    # Detect interactive mode
    if len(sys.argv) == 1 or hasattr(sys, "ps1") or "ipykernel" in sys.modules:
        print("Running in interactive mode ,saving *.csv in results folder and *.PNG in results/plots(VSCode/Jupyter)")
        df = pd.read_csv(csv_path)
        df = add_features(df)
        additional_insights_row(df, save_plots=True, plot_dir=PLOTS_DIR)  # show plots interactively

        model, X_train, X_test, y_train, y_test = train_best_model(df)
        shap_insights(model, X_train, X_test, save_plots=True, save_dir=PLOTS_DIR)  # show SHAP plots interactively
    else:
        main()

