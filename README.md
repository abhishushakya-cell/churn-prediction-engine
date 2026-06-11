# 🔮 AI Churn Prediction + Intervention Engine

> End-to-end ML + Generative AI system that predicts customer churn and auto-generates personalized retention emails — built for resume showcase.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test AUC-ROC | **0.88** |
| Precision (churn class) | **0.96** |
| Customers scored | 1,000 |
| Email generation time | < 2s per customer |

---

## 🏗️ Architecture

```
Raw Data (CRM · Usage · Billing · Support)
         ↓
  Feature Engineering  ←  RFM scores, engagement index, risk signals
         ↓
   XGBoost Model       ←  Optuna-tuned, 0.88 AUC-ROC
         ↓
  SHAP Explainer       ←  Top 3 churn drivers per customer
         ↓
  RAG Context Builder  ←  Customer history + retention playbooks
         ↓
  LLM Email Generator  ←  Claude API / template fallback
         ↓
  Streamlit Dashboard  ←  At-risk list, scores, email preview
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourusername/churn-engine
cd churn-engine
pip install -r requirements.txt

# 2. Generate data & train model
python data/generate_data.py
python model.py

# 3. Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
churn_engine/
├── data/
│   └── generate_data.py    # Synthetic dataset generator
├── rag/
│   └── email_generator.py  # RAG + LLM email engine
├── models/                 # Saved model artifacts (auto-created)
├── features.py             # Feature engineering pipeline
├── model.py                # XGBoost + Optuna + SHAP training
├── app.py                  # Streamlit dashboard
└── README.md
```

---

## 🔑 API Key (Optional)

Add your Anthropic API key for Claude-powered emails:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
Or paste it directly in the dashboard. Without a key, a smart template fallback is used.

---


---

## 🛠️ Tech Stack

`XGBoost` · `SHAP` · `Optuna` · `LangChain` · `FAISS` · `Anthropic Claude` · `Streamlit` · `pandas` · `scikit-learn`
