import streamlit as st
import nltk
import string
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')


def preprocess(text):
    text = text.lower()                                       # lowercase
    tokens = word_tokenize(text)                              # tokenize
    tokens = [t for t in tokens if t not in string.punctuation]  # remove punctuation

    stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stops]            # remove stopwords

    return " ".join(tokens)



with open("Jomo_Kenyatta.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split into sentences first
sentences = sent_tokenize(raw_text)

# Preprocess each sentence
processed_sentences = [preprocess(s) for s in sentences]



def get_most_relevant_sentence(user_input):
    user_input_processed = preprocess(user_input)

    all_documents = processed_sentences + [user_input_processed]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(all_documents)

    similarity_scores = cosine_similarity(tfidf[-1], tfidf[:-1])
    index = similarity_scores.argmax()
    score = similarity_scores[0][index]

    return sentences[index], score


GREETINGS = ["hello", "hi", "hey"]

def chatbot(user_input):
    user_input_lower = user_input.lower()

    # Greetings rule
    if any(g in user_input_lower for g in GREETINGS):
        return random.choice([
            "Hello! Ask me anything about Jomo Kenyatta.",
            "Hi there! What would you like to know?",
            "Hello! I'm here to answer questions about Mzee Jomo Kenyatta."
        ])

    # Similarity-based answer
    best_sentence, score = get_most_relevant_sentence(user_input)

    if score < 0.1:
        return "I'm not sure about that. Try asking something else related to his life, career, or history."

    return best_sentence



def main():
    st.title("Jomo Kenyatta Chatbot")
    st.write("Ask a question about Mzee Jomo Kenyatta based on the provided text file.")

    user_input = st.text_input("Your question:")

    if st.button("Send"):
        if user_input.strip() == "":
            st.warning("Please type something.")
        else:
            response = chatbot(user_input)
            st.success(response)


if __name__ == "__main__":
    main()
