import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import BayesianRidge
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

st.set_page_config(
    page_title="Workplace Mental Health AI", 
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Fair & Uncertainty-Aware Decision Support System")
st.markdown("### Interactive Research Dashboard")
st.markdown("---")

# =====================================================================
# 🪙 CLOUD ENGINE TRAINING (CACHED TO PREVENT RE-RUN CRASHES)
# =====================================================================
@st.cache_resource
def run_analytics_and_train_engine():
    """Runs data engineering, fits models once on cloud boot, and caches the state."""
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        return None, None, None, None
        
    df = pd.read_csv(data_path)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Target variable engineering
    df['target_mental_health'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)

    # Layer 1: Ordered Logit Background Run
    psy_df = df.copy()
    mapping = {'Yes': 1, 'No': 0, "I don't know": 0}
    psy_df['employer_talk'] = psy_df['mh_employer_discussion'].map(mapping).fillna(0)
    psy_df['coworker_talk'] = psy_df['mh_coworker_discussion'].map(mapping).fillna(0)
    X_psy = sm.add_constant(psy_df[['employer_talk', 'coworker_talk', 'age']])
    y_psy = psy_df['mh_share'].astype(int)
    ol_model = OrderedModel(y_psy, X_psy, distr='logit')
    ol_results = ol_model.fit(disp=False)
    logit_summary_table = ol_results.summary().tables[1].as_html()

    # Layer 2: Machine Learning Matrix (Strict Demographic Blindness enforced)
    features = ['tech_company', 'benefits', 'workplace_resources', 
                'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'mh_share', 'age']
    X_ml = pd.get_dummies(df[features])
    y_ml = df['target_mental_health']

    bayes_model = BayesianRidge()
    bayes_model.fit(np.array(X_ml, dtype=np.float64), np.array(y_ml, dtype=np.float64))
    
    # Generate Feature Weights Chart purely in-memory
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    coef_df = pd.DataFrame({'Feature': X_ml.columns, 'Weight': bayes_model.coef_})
    coef_df['Abs_Weight'] = coef_df['Weight'].abs()
    coef_df = coef_df.sort_values(by='Abs_Weight', ascending=False).head(10)
    sns.barplot(data=coef_df, x='Weight', y='Feature', palette='coolwarm', ax=ax)
    ax.set_title("Bayesian Feature Weight Matrix (Lagging & Protective Vector Signals)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    return bayes_model, list(X_ml.columns), logit_summary_table, fig

# Execute/retrieve the backend engine state from cache
bayes_model, model_columns, logit_table, weights_fig = run_analytics_and_train_engine()

if bayes_model is None:
    st.error("❌ Critical Error: 'data.csv' was not discovered in your repository.")
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

# Build dynamic inference payload
user_input = pd.DataFrame([{
    'age': age, 'mh_share': mh_share, 'tech_company': tech_company, 'benefits': benefits,
    'workplace_resources': resources, 'mh_employer_discussion': emp_discuss,
    'mh_coworker_discussion': cowork_discuss, 'medical_coverage': med_coverage
}])

# Real-time dummy alignment transformation
user_encoded = pd.get_dummies(user_input)
final_features = pd.DataFrame(0, index=[0], columns=model_columns)
for col in user_encoded.columns:
    if col in final_features.columns:
        final_features[col] = user_encoded[col].values

# =====================================================================
# 🔮 REAL-TIME PREDICTIONS & CORRECTIONS
# =====================================================================
prob_mean, prob_std = bayes_model.predict(np.array(final_features, dtype=np.float64), return_std=True)
risk_probability = np.clip(prob_mean[0], 0, 1)
epistemic_uncertainty = prob_std[0]**2

# Apply post-processing calibrated boundaries
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
    st.info("💡 **Methodology Context:** The optimization engine automatically applies group-specific boundaries post-hoc to remove demographic skew while ensuring the model parameters remain completely blind to protected identity data during calculation.")

# =====================================================================
# 📉 INTERACTIVE ANALYTICAL SUBSTRATE PANELS
# =====================================================================
st.markdown("---")
tab1, tab2 = st.tabs(["📊 Model Parameters & Weights", "📈 Latent Psychometric Structures"])

with tab1:
    st.markdown("#### Cloud-Generated Feature Coefficients Matrix")
    st.pyplot(weights_fig)

with tab2:
    st.markdown("#### Layer 1 Ordered Logit Coefficients Matrix")
    st.markdown(logit_table, unsafe_allowed_html=True)
