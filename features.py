import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RFM features
    df['recency_score']   = pd.cut(df['days_since_last_login'], bins=[0,7,14,30,60,90],
                                   labels=[5,4,3,2,1]).astype(float)
    df['frequency_score'] = pd.cut(df['logins_last_30d'], bins=[-1,2,5,15,30,100],
                                   labels=[1,2,3,4,5]).astype(float)
    df['monetary_score']  = pd.cut(df['monthly_spend'], bins=[0,29,79,199,10000],
                                   labels=[1,2,3,4]).astype(float)
    df['rfm_score'] = df['recency_score'] + df['frequency_score'] + df['monetary_score']

    # Engagement index
    df['engagement_index'] = (
        df['logins_last_30d'] * 0.4 +
        df['features_used']   * 0.4 +
        df['nps_score']        * 0.2
    ).round(2)

    # Risk signals
    df['risk_score'] = (
        df['billing_failures']     * 3 +
        df['plan_downgrades']      * 4 +
        df['support_tickets_open'] * 2 +
        (df['days_since_last_login'] > 14).astype(int) * 3
    )

    # Encode plan
    le = LabelEncoder()
    df['plan_encoded'] = le.fit_transform(df['plan'])

    return df

FEATURE_COLS = [
    'tenure_months', 'days_since_last_login', 'logins_last_30d',
    'features_used', 'support_tickets_open', 'billing_failures',
    'plan_downgrades', 'nps_score', 'contract_months_left',
    'monthly_spend', 'rfm_score', 'engagement_index',
    'risk_score', 'plan_encoded'
]

if __name__ == "__main__":
    df = pd.read_csv("data/customers.csv")
    df = engineer_features(df)
    df.to_csv("data/customers_features.csv", index=False)
    print("Features engineered. Columns added:")
    print([c for c in df.columns if c not in pd.read_csv("data/customers.csv").columns])
