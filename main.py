import streamlit as st
import pandas as pd
import numpy as np
import joblib
import featuring

model = joblib.load('model_titanic.pkl')

st.title('Survive or Die?')
Name = st.text_input('Name : ', 'John Smith')
Pclass = st.selectbox('Passenger class', [1,2,3])
Sex = st.selectbox('Sex : ', ['male', 'female'])
Age = st.number_input('Age :', 0, 100)
Sibsp = st.number_input('Number of Sibling and Sprouse', 0, 30)
Parch =st.number_input('Number of Parent and Child', 0, 30)
Fare = st.number_input('Fare : ', 0, 3000)
Embarked = st.selectbox('Embark : (C = Cherbourg, Q = Queenstown, S = Southampton)', ['C', 'Q', 'S'])

def prediction():
    row = np.array([Pclass, Name, Sex, Age, Sibsp, Parch, Fare, Embarked])
    X = pd.DataFrame([row], columns=["Pclass", 'Name', 'Sex', 'Age', 'Sibsp', 'Parch', 'Fare', 'Embarked'])
    X = featuring.featuring(X)
    prediction = model.predict(X)

    if prediction >= 1:
        st.success("you are survived :thumbsup")
    else:
        st.error("you died :thumbsdown")


st.button('Predict', on_click=prediction)