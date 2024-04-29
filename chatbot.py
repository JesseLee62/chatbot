import pickle
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import json
vectorizer = TfidfVectorizer()  # Initialize TfidfVectorizer

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
        'dislikes': []
    }

    # Check if the user is a returning user
    old_user_info = load_user_info(name)
    if old_user_info and len(old_user_info['name']) > 0:    # existing user
        print(f"Welcome back {old_user_info['name']}! How can I help you today?")
        user_info = old_user_info
    else:   # new user
        print(f'Hi {name}! How can I help you today?')

    # Display available topics to the user  
    print("Feel free to ask me anything about Michael Jordan, or any topic related to him, including the following terms: ", end="")
    print(", ".join(term_dict.keys()))  # ("'jordan', 'michael', 'nba', 'game', 'team', 'player', 'basketball','season', 'bull', 'point', 'first', 'chicago', 'career', 'award', 'championship'")

    # Get all sentences in the knowledge base
    all_sentences = [sentence for sentences in term_dict.values() for sentence in sentences]
    vectorizer.fit(all_sentences)

    # Start to interact with the user
    while(1):
        user_input = input(": ")  # Read user input
        print()
        # If the user inputs "exit", exit the program
        if "exit" in user_input.lower():
            break

        # Generate a response based on the user's input
        response = generate_response(user_input, term_dict)
        print(response)

        # If there are no known terms in the user's input, ask user to input again
        if response.startswith("I'm sorry"):
            print("You can ask me anything about Michael Jordan, or any topic related to him, including the following terms: ", end="")
            print(", ".join(term_dict.keys()))
            continue
        print()

        # Ask the user if they like the response and save their preferences
        like = input(">>> Do you like this response? (yes/no): ").lower()
        while 1:       
            if like == "yes":
                save_user_preferences(user_info, response, True)
                break
            elif like == "no":
                save_user_preferences(user_info, response, False)
                break
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
                like = input(": ").lower()

        print(">>> Thanks for your feedback. What else would you like to know about Michael Jordan? If you are done talking, type 'exit' to quit.")

    # User quit, save the user's information
    save_user_info(user_info)
    print("Thank you for your questions! Hope to chat with you again soon. Have a great day!")
    

# Generate the response by user's question
def generate_response(user_input, term_dict):
    
    # Preprocess the user's input
    cleaned_input = clean_text(user_input)
    
    # Extract terms contained in the user's input
    user_terms = [term for term in term_dict.keys() if term in cleaned_input]

    # If there are no known terms in the user's input, return message to ask them to try again
    if not user_terms:
        return "I'm sorry, I couldn't find any relevant terms in your input. Please try again with different keywords."
    
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

# Save user's info     
def save_user_info(user_info):
    # Get the existing users' info in json file
    users_info = []
    if os.path.exists("users_info.json"):
        with open("users_info.json", 'r', encoding='utf-8') as f:
            users_info = json.load(f)

    # Check if the user already exists in the list
    existing_user_info = None
    for user_data in users_info:
        if user_data['name'] == user_info['name']:
            existing_user_info = user_data
            break

    # If the user exists, update their information
    if existing_user_info:
        existing_user_info.update(user_info)
    # If the user doesn't exist, add their information to the list
    else:
        users_info.append(user_info)

    # Write the updated user information back to the JSON file
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
###
if __name__ == "__main__":
    main()