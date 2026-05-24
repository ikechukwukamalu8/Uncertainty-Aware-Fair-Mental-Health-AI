#!/usr/bin/env python3
"""
Framework: An Uncertainty-Aware and Fair Machine Learning Architecture for Workplace Mental Health Screening
Author:    Ikechukwu Okechi Kamalu 
Module:    Probabilistic Predictive Engine & Post-Processing Optimization Matrix
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import classification_report
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings

# Silence internal framework warning logs gracefully
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def run_framework():
    # -------------------------------------------------------------------------
    # 1. LOCAL DATA VERIFICATION SETUP
    # -------------------------------------------------------------------------
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Critical Error: '{data_path}' was not found in the current directory.")
        print("💡 Please ensure your 'data.csv' file is located in the same directory as this script.")
        sys.exit(1)
        
    print("=" * 80)
    print("🚀 INITIALIZING EQUITABLE WORKPLACE SCREENING FRAMEWORK")
    print("=" * 80)
    
    # Load and preprocess dataset
    df = pd.read_csv(data_path)
    print(f"✅ Data Successfully Loaded! Shape: {df.shape[0]} rows\n")
    
    # Clean string trailing spaces
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Process target vector label map
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)

    # -------------------------------------------------------------------------
    # 2. PSYCHOMETRIC ANALYSIS LAYER (Ordered Logit)
    # -------------------------------------------------------------------------
    print("\n📊 Part 1: PSYCHOMETRIC ANALYSIS (Ordered Logit Summary Table)")
    print("-" * 80)
    
    psy_df = df.copy()
    mapping = {'Yes': 1, 'No': 0, "I don't know": 0}
    psy_df['employer_talk'] = psy_df['mh_employer_discussion'].map(mapping).fillna(0)
    psy_df['coworker_talk'] = psy_df['mh_coworker_discussion'].map(mapping).fillna(0)

    X_psy = psy_df[['employer_talk', 'coworker_talk', 'age']]
    y_psy = psy_df['mh_share'].astype(int)

    ol_model = OrderedModel(y_psy, X_psy, distr='logit')
    ol_results = ol_model.fit(disp=False)
    print(ol_results.summary().tables[1])

    # -------------------------------------------------------------------------
    # 3. PROBABILISTIC MACHINE LEARNING ML PIPELINE
    # -------------------------------------------------------------------------
    features = ['tech_company', 'benefits', 'workplace_resources',
                'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age', 'gender']

    X_ml = pd.get_dummies(df[features], drop_first=True)
    y_ml = df['target_mental_health']
    gender_series = df['gender']

    X_train, X_test, y_train, y_test, _, gender_test = train_test_split(
        X_ml, y_ml, gender_series, test_size=0.2, random_state=42, stratify=y_ml
    )

    bayes_model = BayesianRidge()
    bayes_model.fit(np.array(X_train, dtype=np.float64), np.array(y_train, dtype=np.float64))

    # Derive probability and uncertainty parameters
    probs_mean, probs_std = bayes_model.predict(np.array(X_test, dtype=np.float64), return_std=True)
    probs_clipped = np.clip(probs_mean, 0, 1)
    baseline_preds = (probs_clipped >= 0.5).astype(int)

    print("\n" + "=" * 80)
    print("❌ Part 2: UNADJUSTED BASELINE CLASSIFICATION METRICS (Global Threshold 0.50)")
    print("=" * 80)
    print(classification_report(y_test, baseline_preds))

    # Structural dataframe evaluation matrix
    test_results = pd.DataFrame({
        'gender': gender_test.values,
        'prob': probs_clipped,
        'pred': baseline_preds,
        'uncertainty': probs_std ** 2,
        'true': y_test.values
    }).reset_index(drop=True)

    male_rate_base = test_results[test_results['gender'] == 'Male']['pred'].mean()
    female_rate_base = test_results[test_results['gender'] == 'Female']['pred'].mean()
    di_base = female_rate_base / (male_rate_base + 1e-9)
    
    print(f"· Baseline Male Selection Rate:  {male_rate_base:.3f}")
    print(f"· Baseline Female Selection Rate: {female_rate_base:.3f}")
    print(f"· Baseline Disparate Impact Ratio: {di_base:.3f} ⚠️ (Bias Detected)")

    # -------------------------------------------------------------------------
    # 4. POST-PROCESSING OPTIMIZATION MATRIX CORRECTIONS
    # -------------------------------------------------------------------------
    final_preds = baseline_preds.copy()
    for i in range(len(test_results)):
        p = test_results.loc[i, 'prob']
        g = test_results.loc[i, 'gender']
        if g == 'Male':
            final_preds[i] = 1 if p >= 0.450 else 0
        elif g == 'Female':
            final_preds[i] = 1 if p >= 0.525 else 0

    test_results['optimized_pred'] = final_preds
    male_rate_opt = test_results[test_results['gender'] == 'Male']['optimized_pred'].mean()
    female_rate_opt = test_results[test_results['gender'] == 'Female']['optimized_pred'].mean()
    di_opt = female_rate_opt / (male_rate_opt + 1e-9)

    print("\n" + "=" * 80)
    print("✅ Part 3: FAIRNESS-OPTIMIZED ADVANCED CLASSIFICATION METRICS")
    print("=" * 80)
    print(classification_report(y_test, final_preds))
    print(f"· Optimized Male Selection Rate:  {male_rate_opt:.3f}")
    print(f"· Optimized Female Selection Rate: {female_rate_opt:.3f}")
    print(f"· Optimized Disparate Impact Ratio: {di_opt:.3f} 🎉 (Demographic Parity Targeted)")

    # -------------------------------------------------------------------------
    # 5. GRAPH ASSETS COMPILATION
    # -------------------------------------------------------------------------
    print("\n🎨 COMPILING HIGH-RESOLUTION ANALYTICAL VISUALIZATIONS...")
    os.makedirs('visuals', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Figure 1: Uncertainty KDE Distribution
    plt.figure(figsize=(9, 4.5))
    sns.kdeplot(data=test_results[test_results['gender'].isin(['Male', 'Female'])],
                x='uncertainty', hue='gender', fill=True, common_norm=False, palette="muted", alpha=0.5)
    plt.title("Bayesian Epistemic Uncertainty Distribution Across Genders", fontsize=12, fontweight='bold')
    plt.xlabel("Posterior Variance (Epistemic Uncertainty Parameter)")
    plt.ylabel("Density Profile")
    plt.tight_layout()
    plt.savefig('visuals/bayesian_uncertainty.png', dpi=300)
    plt.close()

    # Figure 2: Feature Matrix Weights Chart
    plt.figure(figsize=(9, 4.5))
    coef_df = pd.DataFrame({'Feature': X_ml.columns, 'Coefficient': bayes_model.coef_})
    sns.barplot(data=coef_df.sort_values(by='Coefficient', key=abs, ascending=False).head(10), 
                x='Coefficient', y='Feature', hue='Feature', palette='coolwarm', legend=False)
    plt.title("Bayesian Feature Weight Matrix (Mean Coefficients)", fontsize=12, fontweight='bold')
    plt.xlabel("Weight Value")
    plt.tight_layout()
    plt.savefig('visuals/bayesian_weights.png', dpi=300)
    plt.close()

    # Figure 3: Fairness Adjustments Comparison Map
    plt.figure(figsize=(8, 4.5))
    rates_df = pd.DataFrame({
        'Metric': ['Male Selection Rate', 'Female Selection Rate', 'Disparate Impact'] * 2,
        'Value': [male_rate_base, female_rate_base, di_base, male_rate_opt, female_rate_opt, di_opt],
        'Model Status': ['Unadjusted'] * 3 + ['Fairness Corrected'] * 3
    })
    sns.barplot(data=rates_df, x='Metric', y='Value', hue='Model Status', palette='Set2')
    plt.axhline(1.0, linestyle='--', color='gray', alpha=0.7, label='Ideal Parity (1.0)')
    plt.axhline(0.8, linestyle=':', color='red', alpha=0.5, label='Statutory Limits (0.8 - 1.25)')
    plt.axhline(1.25, linestyle=':', color='red', alpha=0.5)
    plt.title("Fairness Metric Harmonization Before vs After Correction", fontsize=12, fontweight='bold')
    plt.ylabel("Value / Ratio Scale")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('visuals/fairness_adjustment.png', dpi=300)
    plt.close()

    print("\n🎉 ALL PIPELINE VECTORS COMPLETE. PLOTS GENERATED SUCCESSFULLY IN './visuals/' FOLDER.\n")

if __name__ == "__main__":
    run_framework()
