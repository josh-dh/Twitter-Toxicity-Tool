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

#get pd dataframe for training data
df_data = pd.read_csv("twitter-sentiment-analysis-hatred-speech/train.csv",names=('id','label','tweet'),header=None)

def processTweet(tweet):
	tokens = tknzr.tokenize(tweet[0:-1])
	tokens = [x for x in tokens if x not in stopwords.words('english')]
	tokens = [x for x in tokens if x not in string.punctuation]
	return tokens

#return tuple of processed tweet string, label
def get_tweet_and_label(tweet_number):
	tmp = df_data.iloc[tweet_number].to_numpy()
	return processTweet(tmp[2]), tmp[1]

print(get_tweet_and_label(0))