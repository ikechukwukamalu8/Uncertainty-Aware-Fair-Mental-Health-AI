import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import BayesianRidge

# Set up elite web interface parameters
st.set_page_config(
    page_title="Workplace Mental Health AI", 
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Fair & Uncertainty-Aware Decision Support System")
st.markdown("### Interactive Research Dashboard — *King's College London Presentation*")
st.markdown("---")

# Cached Data Ingestion Guard
@st.cache_data
def load_data():
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        st.error(f"❌ Critical Error: '{data_path}' was not found in the root directory.")
        st.info("💡 Please ensure your 'data.csv' dataset file is uploaded right next to this script.")
        st.stop()
    df = pd.read_csv(data_path)
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)
    return df

df = load_data()
features = ['tech_company', 'benefits', 'workplace_resources', 'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age', 'gender']

# Standardize dummy encoding vectors based on background survey distribution
X = pd.get_dummies(df[features], drop_first=True)
y = df['target_mental_health']

# Fit background Bayesian Predictive Matrix
model = BayesianRidge()
model.fit(np.array(X, dtype=np.float64), np.array(y, dtype=np.float64))

# =====================================================================
# 📋 INTERACTIVE SIDEBAR CONTROL PANEL
# =====================================================================
st.sidebar.header("📋 Input Employee Profile")
age = st.sidebar.slider("Age of Employee", 18, 65, 30)
gender = st.sidebar.selectbox("Gender Identity", ["Male", "Female", "Other"])
tech_company = st.sidebar.selectbox("Is it a Tech Company?", ["Yes", "No"])
benefits = st.sidebar.selectbox("Offers Mental Health Benefits?", ["Yes", "No", "I don't know"])
resources = st.sidebar.selectbox("Provides Mental Health Resources?", ["Yes", "No", "I don't know"])
emp_discuss = st.sidebar.selectbox("Discussed MH with Employer?", ["Yes", "No"])
cowork_discuss = st.sidebar.selectbox("Discussed MH with Coworkers?", ["Yes", "No"])
med_coverage = st.sidebar.selectbox("Provides Medical Coverage?", ["Yes", "No"])
mh_share = st.sidebar.slider("Willingness to share mental health issues with friends/family", 0, 10, 5)

# Build a one-row evaluation dataframe from the sidebar choices
user_input = pd.DataFrame([{
    'age': age, 'mh_share': mh_share, 'tech_company': tech_company, 'benefits': benefits,
    'workplace_resources': resources, 'mh_employer_discussion': emp_discuss,
    'mh_coworker_discussion': cowork_discuss, 'medical_coverage': med_coverage, 'gender': gender
}])

# Align the user inputs with the model's structural feature columns
user_encoded = pd.get_dummies(user_input).reindex(columns=X.columns, fill_value=0)

# =====================================================================
# 🔮 REAL-TIME COMPUTATIONAL COMPUTATION LAYER
# =====================================================================
prob_mean, prob_std = model.predict(np.array(user_encoded, dtype=np.float64), return_std=True)
risk_probability = np.clip(prob_mean[0], 0, 1)
epistemic_uncertainty = prob_std[0]**2

# Apply post-processing group-specific threshold corrections
if gender == 'Male':
    thresh = 0.450
elif gender == 'Female':
    thresh = 0.525
else:
    thresh = 0.500

classification = "Elevated Risk Profile" if risk_probability >= thresh else "Stable Profile"

# =====================================================================
# 📊 PRESENTATION VIEWPORTS LAYOUT
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Post-Processed Decision Support Vector")
    st.metric(label="Calculated Vulnerability Score", value=f"{risk_probability*100:.1f}%")
    st.write(f"Group Optimization Boundary Threshold applied: **{thresh}**")
    
    if classification == "Elevated Risk Profile":
        st.error(f"Classification Outcome: **{classification}**")
    else:
        st.success(f"Classification Outcome: **{classification}**")

with col2:
    st.subheader("⚠️ Quantified Informational Noise Profile")
    st.metric(label="Epistemic Uncertainty (Posterior Variance)", value=f"{epistemic_uncertainty:.4f}")
    st.info("💡 **Methodology Context:** The optimization engine automatically applies post-processing corrections to adjust boundaries for borderline cases in response to severe class skews in the multi-year survey origin data.")

# =====================================================================
# 📉 EXPORTED GRAPH MATRIX SECTION (ROOT DIRECTORY PATHS)
# =====================================================================
st.markdown("---")
st.subheader("📉 Background Analytical Visualizations")
st.markdown("These reference figures display the global dataset distribution features directly from your root repository mapping.")

fig_col1, fig_col2, fig_col3 = st.columns(3)

with fig_col1:
    st.markdown("##### 1. Feature Weights Matrix")
    if os.path.exists("bayesian_weights.png"):
        st.image("bayesian_weights.png", width='stretch')
    else:
        st.caption("ℹ️ *Image 'bayesian_weights.png' not detected in root directory.*")

with fig_col2:
    st.markdown("##### 2. Uncertainty Distributions")
    if os.path.exists("bayesian_uncertainty.png"):
        st.image("bayesian_uncertainty.png", width='stretch')
    else:
        st.caption("ℹ️ *Image 'bayesian_uncertainty.png' not detected in root directory.*")

with fig_col3:
    st.markdown("##### 3. Fairness Harmonization")
    if os.path.exists("fairness_adjustment.png"):
        st.image("fairness_adjustment.png", width='stretch')
    else:
        st.caption("ℹ️ *Image 'fairness_adjustment.png' not detected in root directory.*")
