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
st.markdown("### Interactive Research Dashboard")
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
    
    # Strip whitespace anomalies
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)
    return df

df = load_data()

# FIXED: Isolated gender out of training features matrix to maintain structural blindness
features = ['tech_company', 'benefits', 'workplace_resources', 
            'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age']

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
gender = st.sidebar.selectbox("Gender Identity (Used for Equity Adjustment Only)", ["Male", "Female", "Other"])
tech_company = st.sidebar.selectbox("Is it a Tech Company?", ["Yes", "No"])
benefits = st.sidebar.selectbox("Offers Mental Health Benefits?", ["Yes", "No", "I don't know"])
resources = st.sidebar.selectbox("Provides Mental Health Resources?", ["Yes", "No", "I don't know"])
emp_discuss = st.sidebar.selectbox("Discussed MH with Employer?", ["Yes", "No"])
cowork_discuss = st.sidebar.selectbox("Discussed MH with Coworkers?", ["Yes", "No"])
med_coverage = st.sidebar.selectbox("Provides Medical Coverage?", ["Yes", "No"])

# MODIFIED: Expanded slider UI to a granular 1 to 10 scale for user convenience
mh_share_ui = st.sidebar.slider("Willingness to share mental health issues (Scale 1-10)", 1, 10, 5)

# ALGORITHMIC ADAPTATION: Map the 1-10 scale down to the model's expected 0-2 baseline scale
# 1-3 maps to 0 (Low), 4-7 maps to 1 (Medium), 8-10 maps to 2 (High)
if mh_share_ui <= 3:
    mh_share_mapped = 0
elif mh_share_ui <= 7:
    mh_share_mapped = 1
else:
    mh_share_mapped = 2

# Build a one-row evaluation dataframe matching structural expectations
user_input = pd.DataFrame([{
    'tech_company': tech_company, 'benefits': benefits, 'workplace_resources': resources,
    'mh_employer_discussion': emp_discuss, 'mh_coworker_discussion': cowork_discuss,
    'medical_coverage': med_coverage, 'mh_share': mh_share_mapped, 'age': age
}])

# =====================================================================
# 🛠️ ALIGNMENT TRANSFORMATION BRIDGE (ROBUST COUPLING)
# =====================================================================
# Concatenate user row with features blueprint structure to preserve categorical options
blueprint_df = pd.DataFrame(columns=features)
user_padded = pd.concat([blueprint_df, user_input], ignore_index=True)

# Generate identical structural dummies, enforcing drop_first constraint symmetrically
user_encoded = pd.get_dummies(user_padded, drop_first=True)
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)
user_encoded = user_encoded.astype(np.float64)

# =====================================================================
# 🔮 REAL-TIME COMPUTATIONAL COMPUTATION LAYER
# =====================================================================
prob_mean, prob_std = model.predict(np.array(user_encoded, dtype=np.float64), return_std=True)
risk_probability = np.clip(prob_mean[0], 0, 1)
epistemic_uncertainty = float(prob_std[0]**2)

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
# 📉 EXPORTED GRAPH MATRIX SECTION (ROOT DIRECTORY FIXED MATCHES)
# =====================================================================
st.markdown("---")
st.subheader("📉 Background Analytical Visualizations")
st.markdown("These reference figures display the global dataset distribution features directly from your root repository mapping.")

fig_col1, fig_col2, fig_col3 = st.columns(3)

with fig_col1:
    st.markdown("##### 1. Feature Weights Matrix")
    if os.path.exists("bayesian_weights.png"):
        st.image("bayesian_weights.png", use_container_width=True)
    else:
        st.caption("ℹ️ *Image 'bayesian_weights.png' not detected in root directory.*")

with fig_col2:
    st.markdown("##### 2. Uncertainty Distributions")
    if os.path.exists("bayesian_uncertainty.png"):
        st.image("bayesian_uncertainty.png", use_container_width=True)
    else:
        st.caption("ℹ️ *Image 'bayesian_uncertainty.png' not detected in root directory.*")

with fig_col3:
    st.markdown("##### 3. Fairness Harmonization")
    if os.path.exists("fairness_adjustment.png"):
        st.image("fairness_adjustment.png", use_container_width=True)
    else:
        st.caption("ℹ️ *Image 'fairness_adjustment.png' not detected in root directory.*")
