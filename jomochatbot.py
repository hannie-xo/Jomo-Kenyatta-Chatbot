import streamlit as st
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Load text
with open('Jomo_Kenyatta.txt', "r", encoding="utf-8") as f:
    raw_text = f.read()

from nltk.tokenize import sent_tokenize

# Split text into sentences
def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Convert raw text into a list of sentences
sentences = split_into_sentences(raw_text)

def get_most_relevant_sentence(user_input, sentences):
    documents = sentences + [user_input]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(documents)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity.argmax()
    score = similarity[0][index]
    return sentences[index], score

GREETINGS = ["hello", "hi", "hey"]

def chatbot(user_input):
    user_input_lower = user_input.lower()

    # Greeting
    if any(word in user_input_lower for word in GREETINGS):
        return random.choice([
            "Hello! Ask me anything about Jomo Kenyatta.",
            "Hi there! What would you like to know about Jomo Kenyatta?",
            "Greetings! I can answer questions about Jomo Kenyatta’s life and legacy."
        ])

    # Find best match
    best_sentence, score = get_most_relevant_sentence(user_input, sentences)

    if score < 0.1:
        return "I'm not sure about that. Try asking about his early life, education, politics, or presidency."

    return best_sentence

# Streamlit interface
def main():
    st.title("🇰🇪 Jomo Kenyatta Chatbot")
    st.write("Ask me anything about Jomo Kenyatta!")

    user_input = st.text_input("Your question:")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please type a question.")
        else:
            answer = chatbot(user_input)
            st.success(answer)

if __name__ == "__main__":
    main()


print(len(sentences))
