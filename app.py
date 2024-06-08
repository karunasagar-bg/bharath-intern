import streamlit as st
import pickle

#loading the saved vectorizer and nave model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS spam Classifier")
input_sms =st.text_area("Enter your message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]
    #display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not spam")