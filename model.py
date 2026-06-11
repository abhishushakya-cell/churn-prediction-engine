import pandas as pd
import numpy as np
import pickle, os
import xgboost as xgb
import shap
import optuna
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from features import engineer_features, FEATURE_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_model():
    # Load & engineer features
    df = pd.read_csv("data/customers.csv")
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df['churned']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optuna hyperparameter tuning
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
        return scores.mean()

    print("Tuning hyperparameters with Optuna (30 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_params = study.best_params
    best_params["scale_pos_weight"] = (y_train == 0).sum() / (y_train == 1).sum()
    best_params["eval_metric"] = "auc"
    best_params["use_label_encoder"] = False
    best_params["random_state"] = 42

    print(f"Best AUC (CV): {study.best_value:.4f}")

    # Train final model
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = model.predict(X_test)
    auc          = roc_auc_score(y_test, y_pred_proba)

    print(f"\nTest AUC-ROC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # SHAP explainer
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    with open("models/xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)

    pd.DataFrame(shap_values, columns=FEATURE_COLS).to_csv(
        "models/shap_values.csv", index=False
    )

    print("\nModel + SHAP explainer saved to /models/")
    return model, explainer, auc

def get_top_churn_drivers(customer_row: pd.Series, explainer, top_n=3):
    """Return top N churn drivers for a single customer as human-readable strings."""
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

if __name__ == "__main__":
    train_model()
