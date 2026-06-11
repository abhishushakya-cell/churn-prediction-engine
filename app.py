import streamlit as st
import pandas as pd
import numpy as np
import pickle, sys, os

sys.path.insert(0, os.path.dirname(__file__))

from features import engineer_features, FEATURE_COLS
from model import get_top_churn_drivers
from rag.email_generator import generate_retention_email

st.set_page_config(
    page_title="Churn Prediction Engine",
    page_icon="🔮",
    layout="wide",
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("models/xgb_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/shap_explainer.pkl", "rb") as f:
        explainer = pickle.load(f)
    return model, explainer

@st.cache_data
def load_data():
    df = pd.read_csv("data/customers.csv")
    return engineer_features(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔮 AI Churn Prediction + Intervention Engine")
st.caption("XGBoost · SHAP Explainability · RAG-Powered Retention Emails")

try:
    model, explainer = load_artifacts()
    df = load_data()
except FileNotFoundError:
    st.error("Model not trained yet. Run `python model.py` first.")
    st.stop()

# ── Score all customers ───────────────────────────────────────────────────────
X = df[FEATURE_COLS]
df['churn_probability'] = model.predict_proba(X)[:, 1]
df['risk_tier'] = pd.cut(
    df['churn_probability'],
    bins=[0, 0.4, 0.65, 1.0],
    labels=['🟢 Low', '🟡 Medium', '🔴 High']
)

at_risk = df[df['churn_probability'] >= 0.5].sort_values(
    'churn_probability', ascending=False
).reset_index(drop=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers",   f"{len(df):,}")
c2.metric("At-Risk Customers", f"{len(at_risk):,}",
          delta=f"{len(at_risk)/len(df)*100:.1f}% of base", delta_color="inverse")
c3.metric("MRR at Risk",
          f"${at_risk['monthly_spend'].sum():,.0f}",
          delta="monthly recurring revenue")
c4.metric("Avg Churn Score (at-risk)",
          f"{at_risk['churn_probability'].mean():.0%}")

st.divider()

# ── Two-panel layout ──────────────────────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    st.subheader("At-Risk Customers")

    # Filters
    fc1, fc2 = st.columns(2)
    plan_filter = fc1.multiselect("Plan", ['Basic','Pro','Enterprise'],
                                  default=['Basic','Pro','Enterprise'])
    tier_filter = fc2.multiselect("Risk tier",
                                  ['🔴 High','🟡 Medium'],
                                  default=['🔴 High','🟡 Medium'])

    filtered = at_risk[
        at_risk['plan'].isin(plan_filter) &
        at_risk['risk_tier'].isin(tier_filter)
    ]

    display_df = filtered[[
        'customer_id','name','plan','monthly_spend',
        'churn_probability','risk_tier'
    ]].copy()
    display_df['churn_probability'] = display_df['churn_probability'].map("{:.0%}".format)
    display_df['monthly_spend']     = display_df['monthly_spend'].map("${:.0f}".format)
    display_df.columns = ['ID','Name','Plan','MRR','Churn Risk','Tier']

    selected = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        height=420,
    )

with right:
    st.subheader("Customer Detail + Intervention")

    # Get selected row
    sel_rows = selected.selection.rows if hasattr(selected, 'selection') else []
    customer_row = filtered.iloc[sel_rows[0]] if sel_rows else filtered.iloc[0]

    # Profile card
    score = customer_row['churn_probability']
    color = "#ef4444" if score >= 0.65 else "#f59e0b"

    st.markdown(f"""
    <div style="border:1px solid {color};border-radius:12px;padding:16px;margin-bottom:12px">
        <h4 style="margin:0 0 8px">{customer_row['name']}</h4>
        <div style="display:flex;gap:24px;font-size:13px">
            <span>📦 {customer_row['plan']}</span>
            <span>💰 ${customer_row['monthly_spend']:.0f}/mo</span>
            <span>📅 {customer_row['tenure_months']} months</span>
        </div>
        <div style="margin-top:12px;font-size:22px;font-weight:600;color:{color}">
            {score:.0%} churn risk
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Churn drivers
    drivers = get_top_churn_drivers(customer_row, explainer, top_n=3)
    st.markdown("**Top churn drivers:**")
    for d in drivers:
        st.markdown(f"• {d}")

    st.divider()

    # Email generation
    st.markdown("**Retention email**")
    api_key = st.text_input("Anthropic API key (optional — leave blank for template)",
                             type="password", placeholder="sk-ant-...")

    if st.button("✉️ Generate Retention Email", use_container_width=True, type="primary"):
        with st.spinner("Generating personalized email..."):
            result = generate_retention_email(
                customer_row.to_dict(), score, drivers,
                api_key=api_key or None
            )
        st.session_state['email_result'] = result

    if 'email_result' in st.session_state:
        r = st.session_state['email_result']
        badge = "🤖 Claude API" if r['source'] == 'claude' else "📝 Template"
        st.caption(badge)
        st.text_input("Subject", value=r['subject'])
        st.text_area("Email body", value=r['body'], height=220)

# ── Risk distribution chart ───────────────────────────────────────────────────
st.divider()
st.subheader("Churn Score Distribution")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
fig.patch.set_alpha(0)

# Distribution
axes[0].hist(df['churn_probability'], bins=30, color='#6366f1', alpha=0.8, edgecolor='none')
axes[0].axvline(0.5, color='#ef4444', linestyle='--', linewidth=1.5, label='Risk threshold')
axes[0].set_xlabel('Churn probability', color='gray')
axes[0].set_ylabel('Count', color='gray')
axes[0].set_title('All customers', color='white')
axes[0].tick_params(colors='gray')
axes[0].set_facecolor('#1e1e2e')
axes[0].legend(labelcolor='gray', framealpha=0)

# MRR at risk by plan
mrr_risk = df[df['churn_probability'] >= 0.5].groupby('plan')['monthly_spend'].sum()
bars = axes[1].bar(mrr_risk.index, mrr_risk.values, color=['#f59e0b','#6366f1','#10b981'],
                   edgecolor='none', alpha=0.9)
axes[1].set_xlabel('Plan', color='gray')
axes[1].set_ylabel('MRR at risk ($)', color='gray')
axes[1].set_title('MRR at risk by plan', color='white')
axes[1].tick_params(colors='gray')
axes[1].set_facecolor('#1e1e2e')
for bar, val in zip(bars, mrr_risk.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'${val:,.0f}', ha='center', fontsize=9, color='gray')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)

st.caption("Built with XGBoost · SHAP · LangChain · Streamlit · Anthropic Claude")
