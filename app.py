import streamlit as st # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import pickle
import time

# Set up the Streamlit app title
st.title("Twitter Sentiment Analysis")

# Load the pre-trained model and vectorizer
try:
    model = pickle.load(open("twitter_sentiment_model.pkl", 'rb'))
    vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please check the file paths and try again.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model or vectorizer: {e}")
    st.stop()

# Mapping of prediction labels to sentiment strings
label_mapping = {
    -1: "Negative",
    1: "Positive",
    0: "Neutral"
}

# Create a text input field for user to enter a tweet
tweet = st.text_input("Enter your comment")

# Create a button for prediction
submit = st.button('Predict')

# When the button is clicked
if submit:
    if tweet:
        # Measure the time taken for prediction
        start = time.time()
        
        # Transform the input text using the loaded vectorizer
        tweet_vector = vectorizer.transform([tweet])
        
        # Make a prediction using the loaded model
        prediction = model.predict(tweet_vector)
        
        # Convert numeric prediction to sentiment string
        sentiment = label_mapping.get(prediction[0], "Unknown")

        # Measure the end time
        end = time.time()
        
        # Display the prediction and time taken
        st.write(f"Prediction sentiment is: **{sentiment}**")
        st.write(f"Prediction time taken: **{round(end - start, 2)}** seconds")
    else:
        st.warning("Please enter a comment before clicking Predict.")
