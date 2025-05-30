# Name: DEBASIS MAJI
# Class Roll No: 30
# Exam Roll No : 97/CSM/241009
# Registration No: D01-1111-0375-24

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#  Page Config
st.set_page_config(page_title="Dataset ML App", page_icon="📊", layout="wide")

#  Title
st.markdown("<h1 style='text-align: center; color: #FF6347;'>🤖 Dataset Preprocessor, Visualizer & ML App</h1>", unsafe_allow_html=True)

#  Upload Dataset
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded Successfully!")

    st.write("### 🔍 Original Dataset Preview")
    st.dataframe(df.head())

    processed_df = df.copy()

    #  Handle Missing Values
    st.sidebar.header("🧼 Handle Missing Values")
    missing_method = st.sidebar.radio("Choose Method", ["None", "Mean", "Median", "Mode"])
    columns_to_fill = st.sidebar.multiselect("Select Columns to Fill Missing Values", df.columns.tolist())

    if missing_method != "None" and columns_to_fill:
        for col in columns_to_fill:
            if missing_method == "Mean" and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col].fillna(processed_df[col].mean(), inplace=True)
            elif missing_method == "Median" and pd.api.types.is_numeric_dtype(processed_df[col]):
                processed_df[col].fillna(processed_df[col].median(), inplace=True)
            elif missing_method == "Mode":
                processed_df[col].fillna(processed_df[col].mode().iloc[0], inplace=True)
            else:
                st.warning(f"⚠️ Cannot apply {missing_method} on non-numeric column '{col}'")

    #  Scaling
    numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
    st.sidebar.header("📐 Scaling")
    scaling_method = st.sidebar.radio("Select Scaling Method", ["None", "Standard Scaling", "Min-Max Scaling"])
    scaling_cols = st.sidebar.multiselect("Select Columns to Scale", numeric_cols)

    if scaling_method != "None" and scaling_cols:
        scaler = StandardScaler() if scaling_method == "Standard Scaling" else MinMaxScaler()
        processed_df[scaling_cols] = scaler.fit_transform(processed_df[scaling_cols])

    #  Display Processed Dataset
    st.write("### ✅ Processed Dataset")
    st.dataframe(processed_df.head())

    #  Visualization
    st.markdown("---")
    st.markdown("<h2 style='color:#4682B4;'>📈 Visualize Dataset</h2>", unsafe_allow_html=True)

    plot_type = st.selectbox("Choose Plot Type", ["Histogram", "Boxplot", "Scatter Plot", "Line Plot", "Heatmap"])

    if plot_type == "Histogram":
        hist_col = st.selectbox("Select column for Histogram", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(processed_df[hist_col], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

    elif plot_type == "Boxplot":
        box_col = st.selectbox("Select column for Boxplot", numeric_cols)
        fig, ax = plt.subplots()
        sns.boxplot(y=processed_df[box_col], ax=ax, color="orange")
        st.pyplot(fig)

    elif plot_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        fig, ax = plt.subplots()
        sns.scatterplot(x=processed_df[x_col], y=processed_df[y_col], ax=ax)
        st.pyplot(fig)

    elif plot_type == "Line Plot":
        x_col = st.selectbox("X-axis", numeric_cols, key="line_x")
        y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
        fig, ax = plt.subplots()
        sns.lineplot(x=processed_df[x_col], y=processed_df[y_col], ax=ax)
        st.pyplot(fig)

    elif plot_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(processed_df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    #  ML Section
    st.markdown("---")
    st.markdown("<h2 style='color:#008080;'>🤖 Apply Machine Learning Model</h2>", unsafe_allow_html=True)

    # Automatically choose the last column as target
    target_col = processed_df.columns[-1]
    st.success(f"Auto-selected Target Column: **{target_col}**")

    model_option = st.selectbox("Choose Model", [
        "Logistic Regression", 
        "Decision Tree", 
        "Random Forest", 
        "K-Nearest Neighbors", 
        "Support Vector Machine"
    ])

    if st.button("Run Model"):
        try:
            X = processed_df.drop(columns=[target_col])
            y = processed_df[target_col]

            # Encode non-numeric columns
            X = pd.get_dummies(X, drop_first=True)
            if y.dtype == 'object':
                y = pd.factorize(y)[0]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Model selection
            model = None
            if model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_option == "Random Forest":
                model = RandomForestClassifier()
            elif model_option == "K-Nearest Neighbors":
                model = KNeighborsClassifier()
            elif model_option == "Support Vector Machine":
                model = SVC()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            #  Display Results
            st.subheader("🎯 Model Evaluation")
            st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 4))
            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred))
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    #  Download Processed Dataset
    st.sidebar.download_button(
        label="⬇️ Download Processed CSV",
        data=processed_df.to_csv(index=False).encode("utf-8"),
        file_name="processed_dataset.csv",
        mime="text/csv"
    )