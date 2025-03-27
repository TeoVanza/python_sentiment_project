import os
import sys 
sys.path.append(os.path.abspath('..'))

from src import config
import streamlit as st
import pickle


with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "rb") as file:
        model = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)

st.title("Text Classification")

# text input 

user_input = st.text_area("enter text to classify", "")

if st.button("classify"):
    if user_input.strip() == "":
        st.warning("please enter some text.")
    else: 
        x = vectorizer.transform([user_input])
        prediction = model.predict(x)[0]
        if prediction == "positive":
              st.success(f"Predicted class: {prediction}")
        elif prediction == "negative":
              st.success(f"Predicted class: {prediction}")
              



