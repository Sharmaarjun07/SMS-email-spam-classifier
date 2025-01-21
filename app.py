import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Load pre-trained vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer1.pkl', 'rb'))
    model = pickle.load(open('model1.pkl', 'rb'))

    # Check if the model is already trained
    if not hasattr(model, "predict"):
        raise ValueError("The loaded model is not trained or incompatible.")
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()
except ValueError as e:
    st.error(f"Model error: {e}")
    st.stop()

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Streamlit UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # 1. Preprocess the input
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize the transformed text
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict using the trained model
        try:
            result = model.predict(vector_input)[0]
            # 4. Display the result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
