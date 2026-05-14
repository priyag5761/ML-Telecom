from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from models.preprocessing import load_and_preprocess_data
import pandas as pd


def run_knn():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()
    
    
    report_df.loc["accuracy", "support"] = len(y_test)

    return accuracy, report_df