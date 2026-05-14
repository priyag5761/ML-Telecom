import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.decision_tree import run_decision_tree
from models.naive_bayes import run_naive_bayes
from models.knn import run_knn
from models.kmeans import run_kmeans
from models.em import run_em

from models.ensemble import run_ensemble


st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>

/* Main background */
.main {
    background-color: #0E1117;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    color: white;
}

/* Main title */
h1 {
    color: #4CAF50;
    font-weight: bold;
}

/* Subheaders */
h2, h3 {
    color: #F8FAFC;
}

/* Buttons */
.stButton > button {
    background-color: #16A34A;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 250px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #15803D;
    color: white;
}

/* Success box */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

df = pd.read_csv("dataset/telco.csv")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Overview",
        "ML Models",
        "Customer Segmentation",
        "Dashboard Analysis",
        "Final Conclusion"
    ]
)

# -------------------------------
# HOME PAGE

if page == "Home":
    st.title("📊 Telecom Customer Churn Prediction System")

    st.markdown("""
    ---
    ## 🚀 Project Objective

    This project helps telecom companies identify customers likely to churn
    and segment customers using Machine Learning algorithms.

    ### 🤖 Algorithms Used

    ✔ Decision Tree  
    ✔ Naive Bayes  
    ✔ KNN  
    ✔ K-Means Clustering  
    ✔ EM Algorithm

    ### 🖥 Frontend

    ✔ Streamlit Dashboard

    ### 📈 Business Goal

    Reduce customer churn and improve customer retention.
    ---
    """)




# -------------------------------
# DATASET OVERVIEW
# -------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")

    st.dataframe(df.head(20))

    st.write("Total Rows:", df.shape[0])
    st.write("Total Columns:", df.shape[1])

    st.subheader("Column Names")
    st.write(list(df.columns))



# -------------------------------
# ML MODELS
# -------------------------------
elif page == "ML Models":
    # st.title("Classification Models")
    st.title("📊 Classification Models")

    # Session State Initialization
    if "dt_result" not in st.session_state:
        st.session_state.dt_result = None

    if "nb_result" not in st.session_state:
        st.session_state.nb_result = None

    if "knn_result" not in st.session_state:
        st.session_state.knn_result = None

 
    
    
    st.subheader("🌳 Decision Tree")

    if st.button("Run Decision Tree"):
        st.session_state.dt_result = run_decision_tree()

    if st.session_state.dt_result is not None:
        accuracy, report_df = st.session_state.dt_result

        st.metric("Decision Tree Accuracy", f"{accuracy * 100:.2f}%")

        st.write("### Classification Report")
        st.dataframe(report_df)

    st.markdown("---")
        
    
    
    
    st.subheader("🧠 Naive Bayes")

    if st.button("Run Naive Bayes"):
        st.session_state.nb_result = run_naive_bayes()

    if st.session_state.nb_result is not None:
        accuracy, report_df = st.session_state.nb_result

        st.metric("Naive Bayes Accuracy", f"{accuracy * 100:.2f}%")

        st.write("### Classification Report")
        st.dataframe(report_df)

    st.markdown("---")
    
 
        
        
    st.subheader("📍 KNN")

    if st.button("Run KNN"):
        st.session_state.knn_result = run_knn()

    if st.session_state.knn_result is not None:
        accuracy, report_df = st.session_state.knn_result

        st.metric("KNN Accuracy", f"{accuracy * 100:.2f}%")

        st.write("### Classification Report")
        st.dataframe(report_df)
        
        


    #ENSEMBLE 2
    st.markdown("---")
    st.subheader("🔥 Ensemble Learning Models")

    # Session state initialization
    if "dt_nb_result" not in st.session_state:
        st.session_state.dt_nb_result = None

    if "nb_knn_result" not in st.session_state:
        st.session_state.nb_knn_result = None

    if "dt_knn_result" not in st.session_state:
        st.session_state.dt_knn_result = None


    # -------------------------------
    # Decision Tree + Naive Bayes
    # -------------------------------
    st.subheader("🌳 + 🧠 Decision Tree + Naive Bayes")

    if st.button("Run DT + NB"):
        st.session_state.dt_nb_result = run_ensemble("dt_nb")

    if st.session_state.dt_nb_result is not None:
        accuracy, report_df = st.session_state.dt_nb_result

        st.metric("DT + NB Accuracy", f"{accuracy * 100:.2f}%")
        st.dataframe(report_df)

    st.markdown("---")


    # -------------------------------
    # Naive Bayes + KNN
    # -------------------------------
    st.subheader("🧠 + 📍 Naive Bayes + KNN")

    if st.button("Run NB + KNN"):
        st.session_state.nb_knn_result = run_ensemble("nb_knn")

    if st.session_state.nb_knn_result is not None:
        accuracy, report_df = st.session_state.nb_knn_result

        st.metric("NB + KNN Accuracy", f"{accuracy * 100:.2f}%")
        st.dataframe(report_df)

    st.markdown("---")


    # -------------------------------
    # Decision Tree + KNN
    # -------------------------------
    st.subheader("🌳 + 📍 Decision Tree + KNN")

    if st.button("Run DT + KNN"):
        st.session_state.dt_knn_result = run_ensemble("dt_knn")

    if st.session_state.dt_knn_result is not None:
        accuracy, report_df = st.session_state.dt_knn_result

        st.metric("DT + KNN Accuracy", f"{accuracy * 100:.2f}%")
        st.dataframe(report_df)


    st.markdown("---")

    # -------------------------------
    # Decision Tree + Naive Bayes + KNN
    # -------------------------------
    if "dt_nb_knn_result" not in st.session_state:
        st.session_state.dt_nb_knn_result = None

    st.subheader("🌳 + 🧠 + 📍 DT + NB + KNN")

    if st.button("Run DT + NB + KNN"):
        st.session_state.dt_nb_knn_result = run_ensemble("dt_nb_knn")

    if st.session_state.dt_nb_knn_result is not None:
        accuracy, report_df = st.session_state.dt_nb_knn_result

        st.metric("DT + NB + KNN Accuracy", f"{accuracy * 100:.2f}%")
        st.dataframe(report_df)


# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------
elif page == "Customer Segmentation":
    st.title("Customer Segmentation")

    if "kmeans_result" not in st.session_state:
        st.session_state.kmeans_result = None

    if "em_result" not in st.session_state:
        st.session_state.em_result = None

    # -------------------
    # K-Means
    # -------------------
    st.subheader("K-Means Clustering")

    if st.button("Run K-Means"):
        st.session_state.kmeans_result = run_kmeans()

    if st.session_state.kmeans_result is not None:
        st.success("K-Means Completed")
        st.dataframe(st.session_state.kmeans_result.head(20))

    # -------------------
    # EM Algorithm
    # -------------------
    st.subheader("EM Algorithm")

    if st.button("Run EM Algorithm"):
        st.session_state.em_result = run_em()

    if st.session_state.em_result is not None:
        st.success("EM Completed")
        st.dataframe(st.session_state.em_result.head(20))



# -------------------------------
# DASHBOARD ANALYSIS
# -------------------------------
elif page == "Dashboard Analysis":
    st.title("Dashboard Analysis")

    st.subheader("Churn Distribution")

    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Contract Type Analysis")

    fig, ax = plt.subplots()
    sns.countplot(x="Contract", data=df, ax=ax)
    plt.xticks(rotation=20)
    st.pyplot(fig)

    st.subheader("Monthly Charge Distribution")

    fig, ax = plt.subplots()
    sns.histplot(df["Monthly Charge"], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# FINAL CONCLUSION
# -------------------------------
elif page == "Final Conclusion":
    st.title("Final Conclusion")

    st.markdown("""
    ## Project Result

    - Successfully predicted telecom customer churn
    - Compared multiple ML algorithms
    - Performed customer segmentation
    - Built interactive Streamlit dashboard

    ## Business Impact

    Helps telecom companies:
    - Reduce churn
    - Improve retention
    - Increase customer satisfaction
    - Improve revenue planning
    """)