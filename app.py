import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, requests
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Churn Prediction Engine",
    page_icon="🔮",
    layout="wide",
)

# ── Feature engineering (inline) ─────────────────────────────────────────────
FEATURE_COLS = [
    'tenure_months', 'days_since_last_login', 'logins_last_30d',
    'features_used', 'support_tickets_open', 'billing_failures',
    'plan_downgrades', 'nps_score', 'contract_months_left',
    'monthly_spend', 'rfm_score', 'engagement_index',
    'risk_score', 'plan_encoded'
]

def engineer_features(df):
    df = df.copy()
    df['recency_score']   = pd.cut(df['days_since_last_login'], bins=[0,7,14,30,60,90],
                                   labels=[5,4,3,2,1]).astype(float)
    df['frequency_score'] = pd.cut(df['logins_last_30d'], bins=[-1,2,5,15,30,100],
                                   labels=[1,2,3,4,5]).astype(float)
    df['monetary_score']  = pd.cut(df['monthly_spend'], bins=[0,29,79,199,10000],
                                   labels=[1,2,3,4]).astype(float)
    df['rfm_score']        = df['recency_score'] + df['frequency_score'] + df['monetary_score']
    df['engagement_index'] = (df['logins_last_30d']*0.4 + df['features_used']*0.4 + df['nps_score']*0.2).round(2)
    df['risk_score']       = (df['billing_failures']*3 + df['plan_downgrades']*4 +
                               df['support_tickets_open']*2 + (df['days_since_last_login']>14).astype(int)*3)
    le = LabelEncoder()
    df['plan_encoded'] = le.fit_transform(df['plan'])
    return df

def get_top_churn_drivers(customer_row, explainer, top_n=3):
    vals = explainer.shap_values(customer_row[FEATURE_COLS].values.reshape(1, -1))[0]
    pairs = sorted(zip(FEATURE_COLS, vals), key=lambda x: abs(x[1]), reverse=True)
    READABLE = {
        "days_since_last_login": "inactive for {} days",
        "billing_failures":      "{} billing failures",
        "support_tickets_open":  "{} open support tickets",
        "plan_downgrades":       "{} plan downgrade(s)",
        "nps_score":             "low NPS score of {}",
        "logins_last_30d":       "only {} logins last 30 days",
        "engagement_index":      "low engagement score ({})",
        "risk_score":            "high risk score ({})",
        "contract_months_left":  "{} months left on contract",
        "tenure_months":         "{} months tenure",
    }
    drivers = []
    for feat, val in pairs[:top_n]:
        if val > 0 and feat in READABLE:
            raw_val = int(round(customer_row[feat])) if feat in customer_row else "N/A"
            drivers.append(READABLE[feat].format(raw_val))
    return drivers if drivers else ["low overall engagement"]

# ── Email generator (inline) ──────────────────────────────────────────────────
PLAYBOOKS = [
    {"trigger": "billing failures payment", "content": "Offer a 1-click payment update link, waive the next month fee if 2+ failures, and assign a CSM to follow up within 24 hours."},
    {"trigger": "inactive low logins engagement", "content": "Highlight 3 features they haven't used, invite them to a personalized onboarding call, and share a success story from a similar customer."},
    {"trigger": "support tickets unresolved", "content": "Escalate to senior support, offer a dedicated support line, and acknowledge the frustration directly."},
    {"trigger": "downgrade cost pricing", "content": "Offer a 20% loyalty discount, show ROI data from similar companies, and offer a free 30-day trial of Enterprise features."},
    {"trigger": "nps unhappy dissatisfied", "content": "Open with a genuine apology, ask one specific question about their pain point, offer a direct line to the VP of Customer Success."},
    {"trigger": "contract expiring renewal", "content": "Offer early renewal at a locked-in rate, present a roadmap of upcoming features, and schedule a business review call."},
]

def build_rag_context(drivers):
    text = " ".join(drivers).lower()
    matched = [pb["content"] for pb in PLAYBOOKS if any(kw in text for kw in pb["trigger"].split())]
    return "\n\n".join(matched[:2]) if matched else PLAYBOOKS[1]["content"]

def generate_retention_email(customer, churn_score, drivers, api_key=None):
    rag_context = build_rag_context(drivers)
    drivers_str = "; ".join(drivers)
    prompt = f"""You are a Customer Success Manager writing a retention email.
CUSTOMER: {customer.get('name')}, {customer.get('plan')} plan, ${customer.get('monthly_spend',79):.0f}/mo, {customer.get('tenure_months',12)} months tenure
CHURN RISK: {churn_score:.0%}
CHURN DRIVERS: {drivers_str}
PLAYBOOK: {rag_context}
Write a warm retention email (120-150 words). Format: Subject: <subject>\n\n<body>"""

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "max_tokens": 400,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=15)
            if resp.status_code == 200:
                text = resp.json()["content"][0]["text"]
                lines = text.strip().split("\n", 1)
                return {"subject": lines[0].replace("Subject:", "").strip(),
                        "body": lines[1].strip() if len(lines) > 1 else text, "source": "claude"}
        except:
            pass

    name = customer.get('name', 'there')
    return {
        "subject": f"We'd love to keep you — a personal note",
        "body": f"""Hi {name},\n\nI noticed some friction on your account recently and wanted to reach out personally.\n\nYour top concerns: {drivers_str}.\n\nHere's what I can offer right now:\n→ Free 30-min onboarding session to unlock unused features\n→ 20% loyalty discount applied immediately\n\nWould you be open to a quick 15-min call this week?\n\nBest,\nAlex from Customer Success""",
        "source": "template"
    }

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
    st.error("Model files not found. Make sure models/ and data/ folders are in the repo.")
    st.stop()

# ── Score customers ───────────────────────────────────────────────────────────
X = df[FEATURE_COLS]
df['churn_probability'] = model.predict_proba(X)[:, 1]
df['risk_tier'] = pd.cut(df['churn_probability'], bins=[0,0.4,0.65,1.0],
                          labels=['🟢 Low','🟡 Medium','🔴 High'])
at_risk = df[df['churn_probability'] >= 0.5].sort_values('churn_probability', ascending=False).reset_index(drop=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Customers",   f"{len(df):,}")
c2.metric("At-Risk Customers", f"{len(at_risk):,}", delta=f"{len(at_risk)/len(df)*100:.1f}% of base", delta_color="inverse")
c3.metric("MRR at Risk",       f"${at_risk['monthly_spend'].sum():,.0f}", delta="monthly recurring revenue")
c4.metric("Avg Churn Score",   f"{at_risk['churn_probability'].mean():.0%}")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1.6, 1], gap="large")

with left:
    st.subheader("At-Risk Customers")
    fc1, fc2 = st.columns(2)
    plan_filter = fc1.multiselect("Plan", ['Basic','Pro','Enterprise'], default=['Basic','Pro','Enterprise'])
    tier_filter = fc2.multiselect("Risk tier", ['🔴 High','🟡 Medium'], default=['🔴 High','🟡 Medium'])
    filtered = at_risk[at_risk['plan'].isin(plan_filter) & at_risk['risk_tier'].isin(tier_filter)]
    display_df = filtered[['customer_id','name','plan','monthly_spend','churn_probability','risk_tier']].copy()
    display_df['churn_probability'] = display_df['churn_probability'].map("{:.0%}".format)
    display_df['monthly_spend']     = display_df['monthly_spend'].map("${:.0f}".format)
    display_df.columns = ['ID','Name','Plan','MRR','Churn Risk','Tier']
    selected = st.dataframe(display_df, use_container_width=True, hide_index=True,
                            on_select="rerun", selection_mode="single-row", height=420)

with right:
    st.subheader("Customer Detail + Intervention")
    sel_rows = selected.selection.rows if hasattr(selected, 'selection') else []
    customer_row = filtered.iloc[sel_rows[0]] if sel_rows else filtered.iloc[0]
    score = customer_row['churn_probability']
    color = "#ef4444" if score >= 0.65 else "#f59e0b"
    st.markdown(f"""<div style="border:1px solid {color};border-radius:12px;padding:16px;margin-bottom:12px">
        <h4 style="margin:0 0 8px">{customer_row['name']}</h4>
        <div style="display:flex;gap:24px;font-size:13px">
            <span>📦 {customer_row['plan']}</span>
            <span>💰 ${customer_row['monthly_spend']:.0f}/mo</span>
            <span>📅 {customer_row['tenure_months']} months</span>
        </div>
        <div style="margin-top:12px;font-size:22px;font-weight:600;color:{color}">{score:.0%} churn risk</div>
    </div>""", unsafe_allow_html=True)
    drivers = get_top_churn_drivers(customer_row, explainer, top_n=3)
    st.markdown("**Top churn drivers:**")
    for d in drivers:
        st.markdown(f"• {d}")
    st.divider()
    st.markdown("**Retention email**")
    api_key = st.text_input("Anthropic API key (optional)", type="password", placeholder="sk-ant-...")
    if st.button("✉️ Generate Retention Email", use_container_width=True, type="primary"):
        with st.spinner("Generating personalized email..."):
            result = generate_retention_email(customer_row.to_dict(), score, drivers, api_key=api_key or None)
        st.session_state['email_result'] = result
    if 'email_result' in st.session_state:
        r = st.session_state['email_result']
        st.caption("🤖 Claude API" if r['source'] == 'claude' else "📝 Template")
        st.text_input("Subject", value=r['subject'])
        st.text_area("Email body", value=r['body'], height=220)

# ── Charts ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Churn Score Distribution")
import matplotlib.pyplot as plt, matplotlib
matplotlib.use("Agg")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
fig.patch.set_alpha(0)
axes[0].hist(df['churn_probability'], bins=30, color='#6366f1', alpha=0.8, edgecolor='none')
axes[0].axvline(0.5, color='#ef4444', linestyle='--', linewidth=1.5, label='Risk threshold')
axes[0].set_xlabel('Churn probability', color='gray'); axes[0].set_ylabel('Count', color='gray')
axes[0].set_title('All customers', color='white'); axes[0].tick_params(colors='gray')
axes[0].set_facecolor('#1e1e2e'); axes[0].legend(labelcolor='gray', framealpha=0)
mrr_risk = df[df['churn_probability'] >= 0.5].groupby('plan')['monthly_spend'].sum()
bars = axes[1].bar(mrr_risk.index, mrr_risk.values, color=['#f59e0b','#6366f1','#10b981'], edgecolor='none', alpha=0.9)
axes[1].set_xlabel('Plan', color='gray'); axes[1].set_ylabel('MRR at risk ($)', color='gray')
axes[1].set_title('MRR at risk by plan', color='white'); axes[1].tick_params(colors='gray')
axes[1].set_facecolor('#1e1e2e')
for bar, val in zip(bars, mrr_risk.values):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, f'${val:,.0f}', ha='center', fontsize=9, color='gray')
plt.tight_layout()
st.pyplot(fig, use_container_width=True)
st.caption("Built with XGBoost · SHAP · Streamlit · Anthropic Claude")
