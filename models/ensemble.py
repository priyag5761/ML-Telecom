from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from models.preprocessing import load_and_preprocess_data
import pandas as pd


def run_ensemble(model_type):
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    dt = DecisionTreeClassifier(random_state=42)
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)

    if model_type == "dt_nb":
        model = VotingClassifier(
            estimators=[
                ('DecisionTree', dt),
                ('NaiveBayes', nb)
            ],
            voting='hard'
        )

    elif model_type == "nb_knn":
        model = VotingClassifier(
            estimators=[
                ('NaiveBayes', nb),
                ('KNN', knn)
            ],
            voting='hard'
        )

    elif model_type == "dt_knn":
        model = VotingClassifier(
            estimators=[
                ('DecisionTree', dt),
                ('KNN', knn)
            ],
            voting='hard'
        )
        
        
    
    elif model_type == "dt_nb_knn":
        model = VotingClassifier(
            estimators=[
                ('DecisionTree', dt),
                ('NaiveBayes', nb),
                ('KNN', knn)
            ],
            voting='hard'
        )
    

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