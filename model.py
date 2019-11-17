#data processing
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(reduce_len = True)
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from joblib import dump, load
import os.path
from os import path

#get pd dataframe for training data
df_data = pd.read_csv("twitter-sentiment-analysis-hatred-speech/train.csv",names=('id','label','tweet'),header=None)
tweetData = df_data.to_numpy().T[2]

def processTweet(tweet):
	tokens = tknzr.tokenize(tweet[0:-1])
	tokens = [x for x in tokens if x not in stopwords.words('english')]
	tokens = [x for x in tokens if x not in string.punctuation]
	return tokens

#return tuple of processed tweet string, label
def get_tweet_and_label(tweet_number):
	tmp = df_data.iloc[tweet_number].to_numpy()
	return processTweet(tmp[2]), tmp[1]

#train tf-idf vectorizer or get existing model
if (path.exists('vectorizer.joblib') == False):
	vectorizer = TfidfVectorizer(tokenizer=processTweet, max_features=2000)
	vectorizer.fit(tweetData)
	dump(vectorizer, 'vectorizer.joblib')
else:
	vectorizer = load('vectorizer.joblib')

#model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
data = vectorizer.transform(tweetData)
labels = df_data.to_numpy().T[1].astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33)
classifier = SVC()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))