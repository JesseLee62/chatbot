from bs4 import BeautifulSoup
import requests
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import os

# Create directories for this program if they don't exist
def create_directories():
    directories = ["data", "cleaned_data", "knowledge_base"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

# Build searchable knowledge base
def build_knowledge_base(terms, num_files):
    # Initialize dictionaries for each term
    term_dict = {term: [] for term in terms}

    # Iterate over each cleaned text file
    for i in range(1, num_files):
        filename = f"data/{i}.txt"
        
        # Read the cleaned text file
        with open(filename, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        
        # Iterate over each sentence in the text file
        for sentence in sentences:
            # Convert the sentence to lowercase
            sentence = sentence.lower()
            # Check if the sentence contains any of the terms
            for term in terms:
                if term in sentence:
                    # If the term is found, store the sentence in the corresponding term dictionary
                    term_dict[term].append(sentence)

    # Write the sentences containing each term to separate files
    for term, sentences in term_dict.items():
        with open(f"knowledge_base/{term}.txt", 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

    # Save term_dict as a pickle file
    with open("knowledge_base.pickle", "wb") as f:
        pickle.dump(term_dict, f)

# Extract at least 25 important terms from the cleaned-up files
def extract_important_terms(files_path):
    # Load cleaned texts
    documents = []
    for file_path in files_path:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    matrix = vectorizer.fit_transform(documents)

    # Get feature names (terms)
    feature_names = vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for each term in the documents
    tfidf_scores = matrix.toarray()

    # Get the average TF-IDF scores for each term
    avg_tfidf_scores = tfidf_scores.mean(axis=0)

    # Combine feature names and average TF-IDF scores
    term_scores = list(zip(feature_names, avg_tfidf_scores))

    # Sort the terms based on TF-IDF scores (from high to low)
    term_scores.sort(key=lambda x: x[1], reverse=True)

    # Output the top 40 important terms
    num_terms = 40
    top_terms = term_scores[:num_terms]  

    return top_terms

# Clean up the text files
def clean_text(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    
    # Remove non-alpha and non-digit words
    words = [word for word in stripped if word.isalpha() or word.isdigit()]
    
    # Convert to lowercase
    words = [word.lower() for word in words]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    
    return cleaned_text

# Read in each raw file and clean it up as much as possible with NLP techniques. If you have x files in, you will have x files out.
def clean_text_files(counter):
    for i in range(1, counter):
        # Load the text from the file
        filename = f"data/{i}.txt"
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
            cleaned_text = clean_text(text)
            print(f"{i}.txt is cleaned")

        # Write the cleaned text to the cleaned file
        cleaned_filename = f"cleaned_data/{i}_cleaned.txt"
        with open(cleaned_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
            print(f"Write cleaned file to {i}_cleaned.txt")

# Scrape text off each page and write it to a file. Write each URL’s text to a separate file.
def save_url_text(url, index):
    res = requests.get(url)
    data = res.text
    soup = BeautifulSoup(data, features="html.parser")

    # Find all <p> tags
    p_tags = soup.find_all('p')

    # Extract the text content from each <p> tag and concatenate them together.
    text_content = [p.get_text() for p in p_tags]
    text_content = '\n'.join(text_content)

    sentences = sent_tokenize(text_content)
    sentences = [re.sub(r'\[\d+\]', '', sentence) for sentence in sentences]
    sentences = [sentence.strip() for sentence in sentences]
    
    # Write the text content into the text file
    filename = f"{index}.txt"
    with open(f'data/{filename}', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            if sentence.endswith('.'):
                f.write(sentence + '\n')
        print(f"Write into {index}.txt")

def main():
    # Step 1: Start with 1-3 URLs that represent a topic (a sport, a celebrity, a place, etc.)
    starter_urls = [
        "https://en.wikipedia.org/wiki/Michael_Jordan",
        "https://www.biography.com/athlete/michael-jordan",
        "https://www.nba.com/history/legends/profiles/michael-jordan"
    ]

    # Create directories will be used later
    create_directories()

    # Put starter_urls into txt first
    with open('urls.txt', 'w') as f:
        for url in starter_urls:
            f.write(str(url) + '\n')   

    # Crawl to find 15 – 25 (or more) relevant URLs
    wiki_counter = 0
    counter = 3  # count the number of urls. Since the starter_urls are counted, start from 3.
    
    with open('urls.txt', 'a') as f:
        for url in starter_urls:
            res = requests.get(url)
            data = res.text
            soup = BeautifulSoup(data, features="html.parser")

            # Iterate over all <a> tags in the HTML content
            for link in soup.find_all('a'):
                link_str = str(link.get('href'))

                # Skip URLs containing 'wiki' after 15 wiki URLs have been encountered, or URLs ending with '.pdf'
                if 'wiki' in link_str and wiki_counter > 15 or link_str.endswith('.pdf'):
                    continue

                # Remove the '/url?q=' prefix from the URL string
                if link_str.startswith('/url?q='):
                    link_str = link_str[7:]

                # Remove any parameters after the '&' character in the URL string
                if '&' in link_str:
                    i = link_str.find('&')
                    link_str = link_str[:i]

                # Check if the URL string starts with 'http' and is not from Google
                if link_str.startswith('http') and 'google' not in link_str:
                    # Exclude URLs from non-English Wikipedia pages
                    if 'wikipedia.org/wiki' in link_str and 'en' not in link_str:
                        continue

                    # Write URL to the file    
                    f.write(link_str + '\n')
                    if 'wiki' in link_str:
                        wiki_counter += 1
                    counter += 1

                    # Exit the loop if the alreay crawl 40 relevant URLs
                    if counter == 40:
                        break

             # Exit the loop if the alreay crawl 40 relevant URLs
            if counter == 40:
                break

    print("Crawling completed")

    # Step 2: 
    save_index = 1
    with open('urls.txt', 'r') as f:
        # Read the URLs
        urls = f.readlines()
        for url in urls:
            url = url.strip()
            # Save the text content of URLs 
            save_url_text(url, save_index)
            save_index += 1

    # Step 3: 
    clean_text_files(counter+1)

    # Step 4:
    files_path = [f'cleaned_data/{i}_cleaned.txt' for i in range(1, counter+1)]
    
    # Get top terms
    top_terms = extract_important_terms(files_path)
    for term, score in top_terms:
        print(term, score)

        # Write top terms into the text file
        with open('top_terms.txt', 'a') as f:
            f.write(str(term) + " " + str(score) + '\n') 

    # Step 5: Build a searchable knowledge base
    terms = ['jordan', 'michael', 'nba', 'game', 'team', 'player', 'basketball', 
            'season', 'bull', 'point', 'first', 'chicago', 'career', 'award', 'championship']
    build_knowledge_base(terms, counter+1)

if __name__ == "__main__":
    main()