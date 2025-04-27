import streamlit as st
import pickle
import pandas as pd

# Load model
with open('titanic_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction 🚢")

# Input widgets
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 100, 25)
sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
parch = st.slider("Parents/Children", 0, 6, 0)
fare = st.slider("Fare", 0, 600, 50)
embarked = st.selectbox("Embarked", ["Cherbourg", "Queenstown", "Southampton"])

# Preprocess inputs
sex = 1 if sex == "Female" else 0
embarked_dict = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
embarked = embarked_dict[embarked]

# Predict
if st.button("Predict"):
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                             columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    survival = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if survival == 1:
        st.success(f"Survived (Probability: {proba:.2%})")
    else:
        st.error(f"Did Not Survive (Probability: {1-proba:.2%})")