spam_words_list = ["#1", "100\%", "100\% free", "100\% satisfied", "50\% off", "Ad", "All New", "Bargain", "Best Price", "Bonus", "Brand New Pager", "Claims not to be selling        anything", "Cost", "Costs ", "Credit", "Discount", "Don’t delete", "Email harvest", "Email marketing", "F r e e", "Fast cash", "For free", "For instant access", "For just $xxx", "For just $yyy", "For only", "For you", "Free", "Free and free", "Free consultation", "Free dvd", "Free gift", "Free sample", "Free trial", "Free website", "Gift certificate", "Give it away", "Giving away", "Giving it away", "Great", "Great offer", "Incredible deal", "Insurance", "Internet market", "Internet marketing", "It’s effective", "Lower interest rate", "Lowest interest rate", "Lowest insurance        rates", "Lowest price", "Luxury", "Luxury car", "Mortgage ", "Mortgage rates", "Name Brand", "New domain extensions", "One hundred percent        free", "Outstanding values", "Please read", "Prize", "Prizes", "Profits", "Promise", "Promise you", "Sale", "Sales", "Sample", "Satisfaction", "Satisfaction        guaranteed", "Stainless Steel", "Stuff on sale", "The best rates", "We hate spam", "Web traffic", "Will not believe your        eyes"]

import nltk
import re

## Tokenize
nltk.download('punkt')
from nltk.tokenize import word_tokenize 
def tokenize_into_words(text):
	tokens = re.split('\W+', text)
	return tokens

## Normalize - Lemmatization (turn nouns/verbs into base dictionary forms)
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatization(tokenized_words):
	lemmatized_text = [lemmatizer.lemmatize(word)for word in tokenized_words]
	return ' '.join(lemmatized_text)

## Filtering Noise (Cleaning) - removing stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stop_words(normalized_words):
    filtered_sentence = []
    for w in normalized_words: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
    return filtered_sentence

## Apply Tokenization, Normalization, Cleaning
def apply_preprocess_all(text):
    # returns string (NOT list)
    tokenized = tokenize_into_words(text)
    normalized = lemmatization(tokenized)
    cleaned_words = remove_stop_words(normalized)
    return " ".join(cleaned_words) # " ".join() joins array elements with ' ' between each one