import streamlit as st
import pandas as pd
import numpy as np
import os

# Page configurations
st.set_page_config(
    page_title="Workplace Mental Health Screening Engine",
    page_icon="🧠",
    layout="wide"
)

# App Title & Signature Header
st.title("An Uncertainty-Aware and Fair Machine Learning Architecture for Workplace Mental Health Screening")
st.markdown("### **Author:** Ikechukwu Okechi Kamalu (MSc Biostatistics)")
st.markdown("---")

# Left Sidebar controls
st.sidebar.header("🕹️ Framework Control Panel")
st.sidebar.markdown("Use these parameters to toggle model visibility and evaluate post-processing metrics dynamically.")
show_psychometric = st.sidebar.checkbox("Show Psychometric Layer", value=True)
show_ml = st.sidebar.checkbox("Show Predictive ML Layer", value=True)
show_fairness = st.sidebar.checkbox("Show Algorithmic Fairness Engine", value=True)

# Main Body Intro
st.markdown("""
### 📝 Project Abstract
This computational framework processes a harmonized multi-year cross-sectional survey dataset ($N = 1,242$) 
to deliver equitable clinical screening and psychometric behavioral insights. The system explicitly balances 
statistical classification performance with algorithmic ethics to actively mitigate historical demographic biases.
""")

# Section 1: Psychometric Table
if show_psychometric:
    st.markdown("---")
    st.header("📊 Pillar 1: Latent Behavioral Psychometric Analysis")
    st.markdown("#### **Table 1:** Psychometric Determinants of Disclosure Willingness (Ordered Logit)")
    
    table1_data = {
        "Predictor Variable": ["Employer Discussion (Yes)", "Coworker Discussion (Yes)", "Age (Continuous)"],
        "Coefficient (β)": [0.2492, 0.5833, -0.0004],
        "Standard Error": [0.117, 0.111, 0.006],
        "z-value": [2.123, 5.250, -0.059],
        "p-value": ["0.034*", "0.000***", "0.953"],
        "95% Conf. Interval": ["[0.019, 0.479]", "[0.366, 0.801]", "[-0.012, 0.012]"]
    }
    st.table(pd.DataFrame(table1_data))
    st.caption("*Note: *p < 0.05, ***p < 0.001. Model estimated via Maximum Likelihood Estimation (N = 1,242).*")

# Section 2: Machine Learning Performance
if show_ml:
    st.markdown("---")
    st.header("🧠 Pillar 2: Probabilistic Machine Learning Predictive Matrix")
    st.markdown("#### **Table 2:** Algorithmic Vulnerability Screening Performance (Holdout Set)")
    
    table2_data = {
        "Framework Mode": ["Unadjusted Baseline", "Unadjusted Baseline", "Fairness-Optimized", "Fairness-Optimized"],
        "Target Risk Class": ["Class 0 (No Risk)", "Class 1 (Elevated Risk)", "Class 0 (No Risk)", "Class 1 (Elevated Risk)"],
        "Precision": [0.55, 0.70, 0.49, 0.68],
        "Recall (Sensitivity)": [0.31, 0.86, 0.22, 0.88],
        "F1-Score": [0.40, 0.77, 0.30, 0.76],
        "Sample Support": [87, 162, 87, 162],
        "Global Accuracy": ["67.0%", "67.0%", "65.0%", "65.0%"]
    }
    st.dataframe(pd.DataFrame(table2_data), width='stretch')

# Section 3: Fairness Optimization & Charts
if show_fairness:
    st.markdown("---")
    st.header("⚖️ Pillar 3: Post-Processing Fairness Boundary Optimization")
    st.markdown("#### **Table 3:** Algorithmic Fairness Audit & Boundary Calibration")
    
    table3_data = {
        "Fairness Metric Indicator": [
            "Male Selection Rate", "Female Selection Rate", "Disparate Impact (DI) Ratio", 
            "Statistical Equity Status", "Applied Boundary Threshold (τ)"
        ],
        "Unadjusted Baseline Model": ["73.0%", "93.2%", "1.276", "Fairness Violation (DI > 1.25)", "τ = 0.500"],
        "Fairness-Optimized Model": ["83.4%", "83.6%", "1.002", "Parity Achieved", "Male = 0.450 / Female = 0.525"],
        "Operational Target Status": ["Internal Base", "Internal Base", "1.00 (Perfect Parity)", "Passes Statutory Limits", "Dynamic Vector"]
    }
    st.table(pd.DataFrame(table3_data))
    
    st.markdown("### 📉 Analytical Visualizations Matrix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Bayesian Feature Weight Matrix")
        if os.path.exists("visuals/bayesian_weights.png"):
            st.image("visuals/bayesian_weights.png", width='stretch')
        else:
            st.info("Image 'visuals/bayesian_weights.png' will load once uploaded to GitHub.")

        st.markdown("#### 2. Algorithmic Fairness Optimization Matrix")
        if os.path.exists("visuals/fairness_adjustment.png"):
            st.image("visuals/fairness_adjustment.png", width='stretch')
        else:
            st.info("Image 'visuals/fairness_adjustment.png' will load once uploaded to GitHub.")

    with col2:
        st.markdown("#### 3. Epistemic Uncertainty Distribution Across Genders")
        if os.path.exists("visuals/bayesian_uncertainty.png"):
            st.image("visuals/bayesian_uncertainty.png", width='stretch')
        else:
            st.info("Image 'visuals/bayesian_uncertainty.png' will load once uploaded to GitHub.")

st.markdown("---")
st.markdown("⚙️ *Framework running live in production environment.*")
