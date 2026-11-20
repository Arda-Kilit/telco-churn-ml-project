import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ActiveServices",
    "AvgMonthlySpend",
    "ChargeRatio",
    "SecuritySupportCount",
    "StreamingCount",
    "SeniorCitizen",
    "IsAutoPay",
    "IsMonthToMonth",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Zero-leakage preprocessing & domain feature engineering transformer.
    All transformations are row-wise (stateless), safe to place inside a
    scikit-learn Pipeline before ColumnTransformer / scaling / encoding.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Drop identifier if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # 1. Fix TotalCharges string/whitespace issue (11 rows with tenure == 0)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # Ensure numeric types
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(float)
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

        # 2. Domain Feature Engineering
        # Active service count (0..8)
        df["ActiveServices"] = (df[SERVICE_COLS] == "Yes").sum(axis=1)

        # Average historical monthly spend (avoids division by zero when tenure == 0)
        df["AvgMonthlySpend"] = np.where(
            df["tenure"] > 0,
            df["TotalCharges"] / df["tenure"],
            df["MonthlyCharges"],
        )

        # Ratio of current monthly charge to historical average (>1 signals recent price hike)
        df["ChargeRatio"] = np.where(
            df["AvgMonthlySpend"] > 0,
            df["MonthlyCharges"] / df["AvgMonthlySpend"],
            1.0,
        )

        # Protective add-ons count (OnlineSecurity + TechSupport + DeviceProtection + OnlineBackup)
        protect_cols = ["OnlineSecurity", "TechSupport", "DeviceProtection", "OnlineBackup"]
        df["SecuritySupportCount"] = (df[protect_cols] == "Yes").sum(axis=1)

        # Entertainment add-ons count
        df["StreamingCount"] = (df[["StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)

        # Automatic payment indicator
        df["IsAutoPay"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)

        # Month-to-month contract risk flag
        df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)

        # Tenure lifecycle bucket
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[-1, 6, 12, 24, 48, 120],
            labels=["0-6_Ay", "7-12_Ay", "13-24_Ay", "25-48_Ay", "49+_Ay"],
        ).astype(str)

        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
