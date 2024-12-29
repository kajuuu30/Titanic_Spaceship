import streamlit as st
import pandas as pd
import joblib  # For loading saved models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set up Streamlit app
st.title("Spaceship Titanic Competition Interface")
st.markdown("### An interactive app for analyzing and predicting the survival of passengers!")

# Sidebar for navigation
st.sidebar.header("Navigation")
options = ["Home", "Upload Data", "Train Model", "Make Predictions"]
choice = st.sidebar.selectbox("Choose a section", options)

# Persistent state for storing model and encoder
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoder' not in st.session_state:
    st.session_state.encoder = None

# File uploader
def upload_file(label):
    return st.file_uploader(label, type=["csv"])

# Preprocessing function
def preprocess_data(data, encoder=None):
    """Preprocess the data: handle missing values, encode categorical columns."""
    data = data.copy()

    # Drop irrelevant columns
    if 'PassengerId' in data.columns:
        data.drop(columns=['PassengerId', 'Name', 'Cabin'], inplace=True, errors='ignore')

    # Handle missing values
    data.fillna(value={'Age': data['Age'].median(), 'HomePlanet': 'Unknown', 'Destination': 'Unknown'}, inplace=True)
    data.fillna(0, inplace=True)  # Fill numeric columns with 0

    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=['object']).columns
    if encoder is None:
        encoder = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            data[col] = encoder[col].fit_transform(data[col].astype(str))
    else:
        for col in categorical_columns:
            if col in encoder:
                # Handle unseen labels by assigning them a default value
                data[col] = data[col].astype(str).apply(lambda x: x if x in encoder[col].classes_ else 'Unknown')
                encoder[col].classes_ = np.append(encoder[col].classes_, 'Unknown')
                data[col] = encoder[col].transform(data[col].astype(str))

    return data, encoder

# Home Section
if choice == "Home":
    st.write("Welcome to the **Spaceship Titanic** interactive interface! Use the sidebar to navigate.")
    st.image("https://cdn.mos.cms.futurecdn.net/AKbyqTKUkicsYGx3xwe3HA.jpg", caption="Spaceship Titanic", width=500)

# Upload Data Section
elif choice == "Upload Data":
    st.subheader("Upload Training and Test Datasets")
    train_file = upload_file("Upload Training Dataset")
    test_file = upload_file("Upload Test Dataset")

    if train_file is not None:
        train_data = pd.read_csv(train_file)
        st.write("Training Data Preview:")
        st.dataframe(train_data.head())

    if test_file is not None:
        test_data = pd.read_csv(test_file)
        st.write("Test Data Preview:")
        st.dataframe(test_data.head())

# Train Model Section
elif choice == "Train Model":
    st.subheader("Train Your Model")
    uploaded_file = upload_file("Upload Training Dataset for Training")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        # Basic preprocessing
        if 'Transported' in data.columns:
            X = data.drop(columns=['Transported'])
            y = data['Transported']
            X, st.session_state.encoder = preprocess_data(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a model
            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Trained with Accuracy: {accuracy:.2f}")

            # Save the model to session state
            st.session_state.model = model
            st.success("Model trained and saved successfully!")

# Make Predictions Section
elif choice == "Make Predictions":
    st.subheader("Make Predictions with Your Model")

    # Ensure model is loaded
    if st.session_state.model is None:
        st.warning("No model found. Train a model first!")
    else:
        test_file = upload_file("Upload Test Dataset for Predictions")

        if test_file is not None:
            test_data = pd.read_csv(test_file)
            st.write("Test Data Preview:")
            st.dataframe(test_data.head())

            # Preprocess test data
            try:
                test_data_processed, _ = preprocess_data(test_data, encoder=st.session_state.encoder)
                predictions = st.session_state.model.predict(test_data_processed)
                test_data['Transported'] = predictions
                st.write("Predictions:")
                st.dataframe(test_data[['PassengerId', 'Transported']])

                # Download predictions
                csv = test_data[['PassengerId', 'Transported']].to_csv(index=False)
                st.download_button("Download Predictions", data=csv, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
