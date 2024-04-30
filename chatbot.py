import pickle
import numpy as np
import pandas as pd
import os
import json
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

vectorizer = TfidfVectorizer()  # Initialize TfidfVectorizer
# Load the sentiment model
model = load_model('sentiment.keras')   # classes: ['happy', 'other', 'sadness']

def main():
    # Load the knowledge base created by web_crawler
    with open("knowledge_base.pickle", 'rb') as f:
        term_dict  = pickle.load(f)

    print("Welcome to Michael Jordan chatbot!")
    print("Before we start, May I have your name and email address? You can type 'exit' to quit.")

    # Get user's info first
    name, email = get_personal_info()

    # Save user's info as a json form
    user_info = {
        'name': name,
        'email': email,
        'likes': [],
        'dislikes': [],
        'neutral': []
    }

    # Check if the user is a returning user
    old_user_info = load_user_info(name)
    if old_user_info and len(old_user_info['name']) > 0:    # existing user
        print(f"Welcome back {old_user_info['name']}! Glad to see you again!")
        user_info = old_user_info
    else:   # new user
        print(f'Hi {name}!')

    # Define keywords associated with Michael Jordan
    jordan_keywords = ["jordan", "him", "who's", "basketball player", "michael", "mj", "air jordan", "chicago bull"]

    # Get all sentences in the knowledge base
    all_sentences = [sentence for sentences in term_dict.values() for sentence in sentences]
    vectorizer.fit(all_sentences)

    # Start to interact with the user
    while(1):
        user_input = input("> ")  # Read user input
        print()
        # If the user inputs "exit", exit the program
        if "exit" in user_input.lower():
            break
        
        # If the user mentioned something about Michael Jordan
        if any(keyword in user_input.lower() for keyword in jordan_keywords):
            # Display available topics to the user  
            print("Bot: Did you mention Jordan? Feel free to ask me anything about Michael Jordan, or any topic related to him, including the following terms: ", end="")
            print(", ".join(term_dict.keys()))  # ("'jordan', 'michael', 'nba', 'game', 'team', 'player', 'basketball','season', 'bull', 'point', 'first', 'chicago', 'career', 'award', 'championship'")
            user_input = input("> ")  # Read user input

            # Generate a response based on the user's input
            response = generate_response(user_input, term_dict)
            print()

            # If there are no known terms in the user's input, ask user to input again
            if response == "":
                print("Bot: Sorry, I don't have the information you want.")
                continue

            print(response)
            print()

            # Ask the user if they like the response and save their preferences
            get_feedback_and_save(user_info, response)
            print(">>> Thanks for your feedback. If you are done talking, type 'exit' to quit.")
        
        # Daily Conversation
        else :
            # print("Bot: conversation")
            predicted_answer = predict_response(user_input)
            print("Bot:", predicted_answer)
        
    # User quit, save the user's information
    save_user_info(user_info)
    print(">>> Thank you for your questions! Hope to chat with you again soon. Have a great day!")
    

# Generate the response by user's question
def generate_response(user_input, term_dict):
    
    # Preprocess the user's input
    cleaned_input = clean_text(user_input)
    
    # Extract terms contained in the user's input
    user_terms = [term for term in term_dict.keys() if term in cleaned_input]

    if "jordan" in user_input or "him" in user_input or "who" in user_input:
        user_terms =['jordan','michael']

    # If there are no known terms in the user's input, return message to ask them to try again
    if not user_terms:
        return ""

    all_sentences = []
    response = []
    # Iterate over each term in the user's input and get the sentences which will be compared later
    for term in user_terms:
        sentences = term_dict.get(term)
        all_sentences.extend(sentences)
        response.append(term_dict[term][0])

    all_sentences = [sentence for sentence in all_sentences if len(sentence.split()) > 5]

    # Convert the user's input into a TF-IDF vector
    user_vector = vectorizer.transform([cleaned_input])

    # Calculate the similarity between the user's input and each sentence
    similarities = cosine_similarity(user_vector, vectorizer.transform(all_sentences))

    # Select the most similar sentences to the user's input as the response
    best_index = np.argsort(similarities.flatten())[-5:][::-1]  # select top 5 most similar sentences to compose a paragraph
    best_response = [all_sentences[i] for i in best_index]

    # Remove leading and trailing whitespace for each sentence
    best_response = [response.strip() for response in best_response if response.strip()]
    response.append(best_response)

    # Make the response more readable before returning
    return clean_response(response)

# Preprocess the user's input so as to do cosine similarity
def clean_text(text):
    # Convert the text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Reassemble the words into text
    cleaned_text = ' '.join(words)

    return cleaned_text

# Make the response sentences cleaner
def clean_response(response):
    # Process each sentence
    cleaned_sentences = []
    for item in response:
        if isinstance(item, list):  # If it's a list, process each sentence in the list
            for sentence in item:
                cleaned_sentence = sentence.strip()
                cleaned_sentences.append(cleaned_sentence)
        else:             
            cleaned_sentence = item.strip()
            cleaned_sentences.append(cleaned_sentence)

    # Reassemble the sentences into a string
    response = ' '.join(cleaned_sentences)

    return response

# Below functions are for user model
def get_personal_info():
    # name
    name = input("Your name: ")
    if "exit" in name:
        exit()
    while name == '':
        print("Please give me your name")
        name = input("Your name: ")
        if "exit" in name:
            exit()

    # email
    email = input("Your email address: ")
    if "email" in email:
        exit()
    while email == '':
        print("Please give me your email")
        email = input("Your email: ")
        if "email" in email:
            exit()

    return name, email

# Save user's preferences
def save_user_preferences(user_info, response, like):
    if like:
        user_info['likes'].append(response)
    else:
        user_info['dislikes'].append(response)

def save_user_info(user_info):
    users_info = []
    if os.path.exists("users_info.json"):
        with open("users_info.json", 'r', encoding='utf-8') as f:
            users_info = json.load(f)

    existing_user_info = next((item for item in users_info if item['name'] == user_info['name']), None)

    if existing_user_info:
        existing_user_info.update(user_info)
    else:
        users_info.append(user_info)

    with open("users_info.json", 'w', encoding='utf-8') as f:
        json.dump(users_info, f, indent=4)

        
# Load user's info
def load_user_info(username):
    user_info = {'name': '', 'email': '', 'likes': [], 'dislikes': []}
    if os.path.exists("users_info.json"):
        with open("users_info.json", 'r', encoding='utf-8') as f:
            users_info = json.load(f)
            for user_data in users_info:
                if user_data['name'] == username:
                    user_info = user_data
                    return user_info  # If existing user information is found
    return user_info

# Sentiment model
def analyze_sentiment(text):
    vocab_size = 25000
    max_seq_length = 64
    emotion_mapping = {
        0: 'happy',
        1: 'other',
        2: 'sadness'
    }
    # Preprocess the text
    tokenizer = Tokenizer(num_words=vocab_size)
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_seq_length, padding='post')

    # Predict sentiment
    pred = model.predict(text_pad)

    # Get the result (Choose the highest probability)
    predicted_label = np.argmax(pred)

    predicted_emotion = emotion_mapping.get(predicted_label, 'happy')
    print("Predicted Emotion:", predicted_emotion)
    return predicted_emotion

def get_feedback_and_save(user_info, response):
    feedback = input(">>> Can you briefly describe how this response made you feel?: ")

    # Analyze sentiment of the feedback
    sentiment = analyze_sentiment(feedback)  
    # print(f"Detected sentiment: {sentiment}")

    # Save feedback based on detected sentiment
    if sentiment == 'happy':
        user_info['likes'].append(response)
    elif sentiment == 'sadness':
        user_info['dislikes'].append(response)
    else:
        user_info['neutral'].append(response)

# Conversation model
# Load the dataset
df = pd.read_csv('Conversation.csv')

# Data preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['question'].values)

# Convert text to sequences
questions_seq = tokenizer.texts_to_sequences(df['question'].values)
answers_seq = tokenizer.texts_to_sequences(df['answer'].values)

# Pad sequences to have the same length
maxlen = max([len(seq) for seq in questions_seq + answers_seq])
questions_seq_padded = pad_sequences(questions_seq, maxlen=maxlen, padding='post')
answers_seq_padded = pad_sequences(answers_seq, maxlen=maxlen, padding='post')

# Split the training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(questions_seq_padded, answers_seq_padded, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))  # bidirectional LSTM to improve context understanding
model.add(Dropout(0.5))                                     # dropout layer to prevent overfitting
model.add(LSTM(256, return_sequences=True))                 # another LSTM layer
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax')) # dense layer

# Compile
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model using the training data
model.fit(X_train, y_train,
          batch_size=32,
          epochs=300,
          validation_data=(X_test, y_test),
          verbose=1,
          validation_split=0.2)

# Use the model to predict response
def predict_response(question):
    # Convert question to sequence
    question_seq = tokenizer.texts_to_sequences([question])
    # Pad sequence
    question_seq_padded = pad_sequences(question_seq, maxlen=maxlen, padding='post')
    # Make prediction
    prediction = model.predict(question_seq_padded)
    # Get predicted answer sequence
    predicted_answer_seq = np.argmax(prediction, axis=-1)
    
    # Convert NumPy array to list
    predicted_answer_seq_list = predicted_answer_seq.tolist()

    # Convert integer sequence to text
    predicted_answer = tokenizer.sequences_to_texts(predicted_answer_seq_list)
    return predicted_answer[0]

###
if __name__ == "__main__":
    main()