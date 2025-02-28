import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
housing_data = pd.read_csv("housing.csv")
X = housing_data.drop(columns=["median_house_value"])
y = housing_data["median_house_value"]

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=["float64", "int64"]).columns
cat_features = ["ocean_proximity"]

# Preprocessing steps
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Define and train Random Forest model
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_pipeline.fit(X_train, y_train)

# Streamlit UI Styling
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")
st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .stTextInput > div > div > input {
        border-radius: 8px;
    }
    .stSelectbox > div {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("🏡 House Price Prediction App")
st.write("### Enter house details below to get the predicted price.")

col1, col2 = st.columns(2)

# User input fields
inputs = {}
with col1:
    for feature in num_features[:len(num_features)//2]:
        unique_values = sorted(X[feature].unique())
        inputs[feature] = st.selectbox(f"{feature}", unique_values)

with col2:
    for feature in num_features[len(num_features)//2:]:
        unique_values = sorted(X[feature].unique())
        inputs[feature] = st.selectbox(f"{feature}", unique_values)

cat_values = {}
with st.expander("Select Categorical Features"):
    for feature in cat_features:
        cat_values[feature] = st.selectbox(feature, X[feature].unique())

# Predict button
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([inputs])
    for feature, value in cat_values.items():
        input_data[feature] = value
    
    prediction = rf_pipeline.predict(input_data)
    st.success(f"🏠 Predicted House Price: **${prediction[0]:,.2f}**")
