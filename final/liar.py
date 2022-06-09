import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

from cleantext import clean
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model, metrics, naive_bayes, svm

# from keras.models import Sequential

import psycopg2
liar_tests = pd.read_csv("./test.tsv", sep='\t', header=None)
liar_trains = pd.read_csv("./train.tsv", sep='\t', header=None)
liar_valid = pd.read_csv("./valid.tsv", sep='\t', header=None)

def truth_mapper(value):
    true_set = {'true', 'half-true', 'barely-true', 'mostly-true'}
    false_set = {'false', 'pants-fire'}
    if value in true_set:
        return True
    elif value in false_set:
        return False
    else:
        raise ValueError

def map_truth(liar_df):
    truths = liar_df[1]
    liar_df[1] = truths.apply(truth_mapper)
    return liar_df


    
