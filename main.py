import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import BayesianRidge

st.set_page_config(
    page_title="Workplace Mental Health AI", 
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Fair & Uncertainty-Aware Decision Support System")
st.markdown("### Interactive Research Dashboard")
st.markdown("---")

@st.cache_resource
def load_and_train_cached_engine():
    """Ingests dataset substrates and trains stable background calculation structures safely."""
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        return None, None
    df = pd.read_csv(data_path)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)
    
    # Mirror pipeline feature array updates for alignment consistency
    features = ['tech_company', 'benefits', 'workplace_resources', 
                'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age']
    X_ml = pd.get_dummies(df[features])
    y_ml = df['target_mental_health']
    
    model = BayesianRidge()
    model.fit(np.array(X_ml, dtype=np.float64), np.array(y_ml, dtype=np.float64))
    return model, list(X_ml.columns)

bayes_model, model_columns = load_and_train_cached_engine()

if bayes_model is None:
    st.error("❌ Critical Error: 'data.csv' was not discovered in your current working repository directory.")
    st.stop()

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
mh_share = st.sidebar.slider("Willingness to share mental health concerns", 0, 10, 5)

# Build matching vector dictionary frame
user_input = pd.DataFrame([{
    'age': age, 'mh_share': mh_share, 'tech_company': tech_company, 'benefits': benefits,
    'workplace_resources': resources, 'mh_employer_discussion': emp_discuss,
    'mh_coworker_discussion': cowork_discuss, 'medical_coverage': med_coverage
}])

# =====================================================================
# 🛠️ ALIGNMENT TRANSFORMATION BRIDGE
# =====================================================================
user_encoded = pd.get_dummies(user_input)
final_features = pd.DataFrame(0, index=[0], columns=model_columns)
for col in user_encoded.columns:
    if col in final_features.columns:
        final_features[col] = user_encoded[col].values

# =====================================================================
# 🔮 REAL-TIME COMPUTATIONAL COMPUTATION LAYER
# =====================================================================
prob_mean, prob_std = bayes_model.predict(np.array(final_features, dtype=np.float64), return_std=True)
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
    st.write(f"Group Optimization Boundary Threshold applied: **{thresh}** (`Cohort: {gender}`)")
    
    if classification == "Elevated Risk Profile":
        st.error(f"Classification Outcome: **{classification}**")
    else:
        st.success(f"Classification Outcome: **{classification}**")

with col2:
    st.subheader("⚠️ Quantified Informational Noise Profile")
    st.metric(label="Epistemic Uncertainty (Posterior Variance)", value=f"{epistemic_uncertainty:.4f}")
    st.info("💡 **Methodology Context:** The optimization engine automatically applies group-specific boundaries post-hoc to remove disparate demographic skew while ensuring the model parameters remain completely blind to protected identity data during calculation.")

# =====================================================================
# 📉 EXPORTED GRAPH MATRIX SECTION
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
        st.caption("ℹ️ *Image 'bayesian_weights.png' not yet exported into the root folder.*")

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
