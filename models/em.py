import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder


def run_em():
    df = pd.read_csv("dataset/telco.csv")

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    df.fillna(df.mode().iloc[0], inplace=True)

    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    X = df[["Tenure in Months", "Monthly Charge", "Total Charges"]]

    model = GaussianMixture(
        n_components=4,
        random_state=42
    )

    df["EM_Cluster"] = model.fit_predict(X)

    return df[["Tenure in Months", "Monthly Charge", "Total Charges", "EM_Cluster"]]