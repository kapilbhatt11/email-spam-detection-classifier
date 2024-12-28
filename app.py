import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK resources
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')

ps = PorterStemmer()

# Load the saved TF-IDF vectorizer and the model
try:
    tfidf = pickle.load(open('spam_vectorizer.pkl', 'rb'))
    model = pickle.load(open('mnb_spam_model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input field for the message
input_sms = st.text_area("Enter the message")

# Function for text transformation
def text_transformation(input_sms):
    text = input_sms.lower()
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

# Process input if provided
if st.button('Predict'):
    try:
        # Preprocessing
        transformed_text = text_transformation(input_sms)

        # Vectorization
        vector_input = tfidf.transform([transformed_text])

        # Model prediction
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"An error occurred: {e}")
