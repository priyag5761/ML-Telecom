import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("dataset/telco.csv")

    # Remove customerID if exists
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df.fillna(df.mode().iloc[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = le.fit_transform(df[column])

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test