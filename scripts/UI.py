import os
import sys 
sys.path.append(os.path.abspath('..'))

from src import config
import streamlit as st
import pickle

# Per far runnare il programma bisogna usare questa linea di codice nel terminal: streamlit run 'nome file'

# random forest
with open(os.path.join(config.MODELS_PATH, "random_forest.pickle"), "rb") as file:
        rf = pickle.load(file)

# logistic 
with open(os.path.join(config.MODELS_PATH, "logistic.pickle"), "rb") as file:
        clf = pickle.load(file)

with open(f"{config.MODELS_PATH}vectorizer.pickle", "rb") as f:
        vectorizer = pickle.load(f)

st.title("Text Classification")

# text input 
scelta = st.selectbox(
    "Scegli il modello:",
    ['Random Forest', 'Logistic']
)

user_input = st.text_area("enter text to classify", "")

if st.button("classify"):
    if user_input.strip() == "":
        st.warning("please enter some text.")
    else: 
        x = vectorizer.transform([user_input])
        if scelta == 'Random Forest':
            prediction = rf.predict(x)[0]
            if prediction == "positive":
                    st.success(f"Predicted class RF: {prediction}")
            elif prediction == "negative":
                    st.success(f"Predicted class RF: {prediction}")
        else:
            prediction = clf.predict(x)[0]
            if prediction == "positive":
                st.success(f"Predicted class CLF: {prediction}")
            elif prediction == "negative":
                st.success(f"Predicted class CLF: {prediction}")

