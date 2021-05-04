import numpy as np
import nltk
from utils import spam_words_list, apply_preprocess_all

class FeatureExtractor():
    def __init__(self, debug=True):
        from nltk.corpus import words
        nltk.download('words')
        self.set_words = set(words.words())
        self.debug = debug

    def _compute_misspelled_words(self, email):
        """
        Computes the number of misspelled words in sentence
        """
        num_misspelled = 0
        word_list = email.split()
        for word in word_list:
            if word in self.set_words:
                num_misspelled += 1
        return num_misspelled
    
    def _compute_spam_phrases_count(self, email):
        """
        Computes the number of known spam words in sentence
        """
        num_spam = 0
        word_list = email.split()
        for word in spam_words_list:
            if word in self.set_words:
                num_spam += 1
        return num_spam
    
    def _compute_exclamation_point_count(self, email):
        """
        Computes the number of exclamation points in sentence
        """
        num_char = 0
        for character in email:
            if character == "!":
                num_char += 1
        return num_char
    
    def extract_features(self, email, debug=False):
        """
        Here is where you will extract your features from the data in 
        the given window.
        
        Make sure that x is a vector of length d matrix, where d is the number of features.
        
        """
        ## PREPROCESS TEXT
        # email = apply_preprocess_all(email) # omit preprocessing due to how long it takes
        
        feature_vector = np.array([
            self._compute_misspelled_words(email),
            self._compute_spam_phrases_count(email),
            self._compute_exclamation_point_count(email)
        ])
        return feature_vector
