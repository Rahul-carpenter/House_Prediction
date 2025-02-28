import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Preprocessing pipeline
num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipeline = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Train Random Forest model
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("random_forest", RandomForestRegressor(n_estimators=100, random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_pipeline.fit(X_train, y_train)

# Streamlit UI Styling
st.set_page_config(page_title="🏡 House Price Predictor", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton > button { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 8px; }
    .stTextInput > div > div > input { border-radius: 8px; }
    .stSelectbox > div { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🏡 House Price Prediction App")
st.write("### Enter details below to predict the estimated house price.")

# User Input Layout
col1, col2 = st.columns(2)
inputs = {}

with col1:
    st.subheader("🏗 Structural Features")
    for feature in num_features[:len(num_features)//2]:
        unique_values = sorted(X[feature].unique())
        inputs[feature] = st.selectbox(f"{feature}", unique_values)

with col2:
    st.subheader("📍 Location & Amenities")
    for feature in num_features[len(num_features)//2:]:
        unique_values = sorted(X[feature].unique())
        inputs[feature] = st.selectbox(f"{feature}", unique_values)

cat_values = {}
with st.expander("🌎 Select Location Type"):
    for feature in cat_features:
        cat_values[feature] = st.selectbox(feature, X[feature].unique())

# Predict Button
if st.button("🔍 Predict Price"):
    input_data = pd.DataFrame([inputs])
    for feature, value in cat_values.items():
        input_data[feature] = value
    
    prediction = rf_pipeline.predict(input_data)
    st.success(f"🏠 Estimated House Price: **${prediction[0]:,.2f}**")

    # Feature Importance Visualization
    st.subheader("📊 Feature Importance")
    feature_importance = rf_pipeline.named_steps["random_forest"].feature_importances_
    feature_names = list(num_features) + list(rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(cat_features))
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importance_df["Importance"], y=importance_df["Feature"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Price Distribution Visualization
    st.subheader("💰 House Price Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(y, bins=50, kde=True, color="blue", alpha=0.7)
    plt.axvline(prediction[0], color="red", linestyle="--", label="Predicted Price")
    plt.legend()
    st.pyplot(fig)

st.write("📝 *Powered by Machine Learning & Streamlit* 🚀")
