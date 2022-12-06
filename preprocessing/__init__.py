import pandas as pd 
import numpy as np 
import re 
import nltk 
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize 
#nltk.download('wordnet') 
 
class CleanerText: 
    def __init__(self)->str: 
        self.special_chars = ",.@?!¬-\''=()" 
        self.regex_dict = {'Tags' : r'@[A-Za-z0-9]+',  
                      '# symbol' : r'#',  
                      'RT' : r'RT',  
                      'Links' : r'https?://\S+', 
                      'Not letters': r'[^A-Za-z\s]+', 
                      'Phone' : r'\+[0-9]{12}'} 
 
    def stemWords(self, string): 
        ps = PorterStemmer() 
        stem = list(map(ps.stem, self.wordTokenize(string))) 
        stemmed = ' '.join(stem) 
        return stemmed 
 
    def removeNoise(self, string): 
        clean_string = string.lower() 
        for char in self.special_chars: 
            clean_string = clean_string.replace(char, "") 
        splitted = self.wordTokenize(clean_string) 
        cleaned = [w.replace(" ", "") for w in splitted if len(w) > 0] 
        clean_string = " ".join(cleaned) 
        return clean_string 
 
    def wordTokenize(self, string): 
        tokenized = word_tokenize(string) 
        return tokenized 
 
    def textNormalization(self, string): 
        for key in self.regex_dict.keys(): 
            string = re.sub(self.regex_dict[key], '', string) 
        normalized =  " ".join(self.wordTokenize(string)) 
        return normalized 
 
    def cleaning(self, data): 
        #text Normalization 
        data = data.apply(self.textNormalization) 
         
        #stem words 
        data = data.apply(self.stemWords) 
         
        #Remove Noise 
        data = data.apply(self.removeNoise) 
         
        return data