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



def truth_mapper(value):
    true_set = {'true', 'half-true', 'barely-true', 'mostly-true'}
    false_set = {'false', 'pants-fire'}
    if value in true_set:
        return True
    elif value in false_set:
        return False
    else:
        raise ValueError

def map_project_truth(liar_df):
    truths = liar_df[1]
    return truths.apply(truth_mapper)



def get_liar():
    liar_tests = pd.read_csv("./test.tsv", sep='\t', header=None)
    liar_trains = pd.read_csv("./train.tsv", sep='\t', header=None)
    liar_valid = pd.read_csv("./valid.tsv", sep='\t', header=None)
    liar = pd.concat([liar_tests, liar_trains, liar_valid], ignore_index = True)
    return liar[2], map_project_truth(liar)

