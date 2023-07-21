import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.corpus import wordnet
import seaborn as sns
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter
import plotly.express as px
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC ,SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

plt.style.use('fivethirtyeight')


#get the data
df =  pd.read_csv('dataset/cyberbullying_tweets.csv')

#show the top 5 data
df.head()
df.info()

stemmer = SnowballStemmer("english")
lematizer=WordNetLemmatizer()

from wordcloud import STOPWORDS
STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                  'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                  'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                  'de', 're', 'amp', 'will'])

def lower(text):
    return text.lower()

def remove_hashtag(text):
    return re.sub("#[A-Za-z0-9_]+", ' ', text)

def remove_twitter(text):
    return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

def remove_stopwords(text):
    return " ".join([word for word in 
                     str(text).split() if word not in STOPWORDS])

def stemming(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

def lemmatizer_words(text):
    return " ".join([lematizer.lemmatize(word) for word in text.split()])

def cleanTxt(text):
    text = lower(text)
    text = remove_hashtag(text)
    text = remove_twitter(text)
    text = remove_stopwords(text)
    text = stemming(text)
    text = lemmatizer_words(text)
    return text

#cleaning the text
df['tweet_clean'] = df['tweet_text'].apply(cleanTxt)

#show the clean text
df.head()
