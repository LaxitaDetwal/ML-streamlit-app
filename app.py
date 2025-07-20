import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ML Predictor", layout="centered")

st.title("🌼 Iris Flower Classifier")
st.write("Input flower features to predict its species")

# Input form
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.3)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.3)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)

    classes = ["Setosa", "Versicolor", "Virginica"]
    st.subheader(f"Prediction: 🌸 {classes[prediction[0]]}")
    st.write("Prediction Probabilities:")
    st.bar_chart(pd.DataFrame(pred_proba, columns=classes))

