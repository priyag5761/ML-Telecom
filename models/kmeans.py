import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


def run_kmeans():
    df = pd.read_csv("dataset/telco.csv")

    # Remove customerID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    if "Total Charge" in df.columns:
        df["Total Charge"] = pd.to_numeric(df["Total Charge"], errors="coerce")

    df.fillna(df.mode().iloc[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = le.fit_transform(df[col])

    # Select important columns for clustering
    X = df[["Tenure in Months", "Monthly Charge", "Total Charges"]]

    model = KMeans(n_clusters=4, random_state=42)

    df["Cluster"] = model.fit_predict(X)

    return df[["Tenure in Months", "Monthly Charge", "Total Charges", "Cluster"]]