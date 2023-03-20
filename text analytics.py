# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:39:21 2023

@author: asus
"""

import requests
from bs4 import BeautifulSoup
import nltk
import json
from monkeylearn import MonkeyLearn

# Step 1: Scraping the article text
url = 'https://monkeylearn.com/sentiment-analysis/'
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
article = soup.find('div', class_='article-body').get_text()

# Step 2: Text preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Tokenization
tokens = word_tokenize(article.lower())

# Remove stop words and lemmatize
cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

# Step 3: Sentiment analysis using MonkeyLearn API
ml = MonkeyLearn('YOUR_API_KEY')

model_id = 'YOUR_MODEL_ID'
response = ml.classifiers.classify(model_id, cleaned_tokens)

result = response.body

# Extract sentiment score from the API response
sentiment_score = result[0]['classifications'][0]['confidence']

# Step 4: Interpret the sentiment score
if sentiment_score >= 0.75:
    sentiment = 'Positive'
elif sentiment_score >= 0.45 and sentiment_score < 0.75:
    sentiment = 'Neutral'
else:
    sentiment = 'Negative'

# Step 5: Print the result
print('Sentiment: ' + sentiment)