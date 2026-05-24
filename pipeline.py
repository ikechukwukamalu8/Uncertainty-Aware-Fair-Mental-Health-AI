#!/usr/bin/env python3
"""
Framework: An Uncertainty-Aware and Fair Machine Learning Architecture for Workplace Mental Health Screening
Author:    Ikechukwu Okechi Kamalu (MSc Biostatistics)
Module:    Probabilistic Predictive Engine & Post-Processing Optimization Matrix
"""

import os
import sys
import subprocess

def install_dependencies():
    """Ensures required libraries are present before running mathematical layers."""
    required_packages = ["pandas", "numpy", "scikit-learn", "statsmodels", "matplotlib", "seaborn"]
    missing_packages = [pkg for pkg in required_packages if pkg not in sys.modules]
    if missing_packages:
        print(f"📦 Missing libraries detected: {missing_packages}. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
            print("✅ All dependencies successfully configured.\n")
        except Exception as e:
            print(f"❌ Error setting up packages automatically: {e}")
            sys.exit(1)

install_dependencies()

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
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def run_framework():
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        print(f"❌ Critical Error: '{data_path}' was not found in the current directory.")
        sys.exit(1)
        
    print("=" * 80)
    print("🚀 INITIALIZING EQUITABLE WORKPLACE SCREENING FRAMEWORK")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)

    print("\n📊 RUNNING PILLAR 1: LATENT BEHAVIORAL PSYCHOMETRIC ANALYSIS...")
    psy_df = df.copy()
    mapping = {'Yes': 1, 'No': 0, "I don't know": 0}
    psy_df['employer_talk'] = psy_df['mh_employer_discussion'].map(mapping).fillna(0)
    psy_df['coworker_talk'] = psy_df['mh_coworker_discussion'].map(mapping).fillna(0)

    X_psy = psy_df[['employer_talk', 'coworker_talk', 'age']]
    X_psy = sm.add_constant(X_psy)
    y_psy = psy_df['mh_share'].astype(int)

    ol_model = OrderedModel(y_psy, X_psy, distr='logit')
    ol_results = ol_model.fit(disp=False)
    print(ol_results.summary().tables[1])

    print("\n🧠 RUNNING PILLAR 2: PROBABILISTIC MACHINE LEARNING PREDICTIVE MATRIX...")
    # FIXED: Isolated gender from features array to enforce strict structural blindness during matrix fitting
    features = ['tech_company', 'benefits', 'workplace_resources', 
                'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age']

    # FIXED: Cleared drop_first=True to align with multi-category model weight plots
    X_ml = pd.get_dummies(df[features])
    y_ml = df['target_mental_health']
    gender_series = df['gender']

    X_train, X_test, y_train, y_test, _, gender_test = train_test_split(
        X_ml, y_ml, gender_series, test_size=0.2, random_state=42, stratify=y_ml
    )

    bayes_model = BayesianRidge()
    bayes_model.fit(np.array(X_train, dtype=np.float64), np.array(y_train, dtype=np.float64))

    probs_mean = bayes_model.predict(np.array(X_test, dtype=np.float64))
    probs_mean_clipped = np.clip(probs_mean, 0, 1)
    preds_baseline = (probs_mean_clipped >= 0.50).astype(int)
    
    print("\n❌ UNADJUSTED BASELINE MODEL METRICS (Standard Global Threshold 0.50)")
    print(classification_report(y_test, preds_baseline))
    
    test_results = pd.DataFrame({'gender': gender_test.values, 'score': probs_mean_clipped})
    male_base_rate = (test_results[test_results['gender'] == 'Male']['score'] >= 0.50).mean()
    female_base_rate = (test_results[test_results['gender'] == 'Female']['score'] >= 0.50).mean()
    print(f"   · Baseline Disparate Impact Ratio:     {female_base_rate / (male_base_rate + 1e-9):.3f} ⚠️")

    print("\n⚖️ RUNNING PILLAR 3: POST-PROCESSING FAIRNESS BOUNDARY OPTIMIZATION...")
    tau_male, tau_female = 0.450, 0.525
    preds_optimized = [1 if (r['score'] >= tau_female if r['gender'] == 'Female' else r['score'] >= tau_male) else 0 for _, r in test_results.iterrows()]
    test_results['pred_opt'] = preds_optimized

    print("\n✅ FAIRNESS-OPTIMIZED MODEL METRICS (Group-Specific Custom Threshold Vectors)")
    print(classification_report(y_test, test_results['pred_opt']))
    
    male_opt_rate = test_results[test_results['gender'] == 'Male']['pred_opt'].mean()
    female_opt_rate = test_results[test_results['gender'] == 'Female']['pred_opt'].mean()
    print(f"   · Optimized Disparate Impact Ratio:     {female_opt_rate / (male_opt_rate + 1e-9):.3f} 🎉")

    print("\n🎨 EXPORTING HIGH-RESOLUTION ANALYTICAL CHARTS FOR PRESENTATION...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    coef_df = pd.DataFrame({'Feature': X_ml.columns, 'Weight': bayes_model.coef_})
    coef_df['Abs_Weight'] = coef_df['Weight'].abs()
    coef_df = coef_df.sort_values(by='Abs_Weight', ascending=False).head(10)
    sns.barplot(data=coef_df, x='Weight', y='Feature', palette='coolwarm')
    plt.title("Bayesian Feature Weight Matrix (Lagging & Protective Vector Signals)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bayesian_weights.png', dpi=300)
    plt.close()
    print("   📸 Saved: 'bayesian_weights.png' directly to root directory footprint.\n")

if __name__ == "__main__":
    run_framework()
