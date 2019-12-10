#TODO:
	#test dimensionality reduction on tf-idf data to see if accuracy stays the same
	#also test different sizes of tf-idf to see if accuracy is still high at lower dimensions
	#data-viz: represent data with dimensionality reduction
	#cross-validation: plot the number of dimensions the SVM uses against the validation error on the test set
	#try another classifier (naive bayes classifier is in sklearn) and look at validation error
		#maybe use GlovE and convnet (Keras)

	#potential extension of project:
		#apply classifier to twitter API
		#unsupervised learning:
			#see if clusters exist between different types of hate speech
				#use tf-idf, dimensionality reduction, then clustering algorithm to separate
				#if clusters of hate speech don't exist, try and find other clusters with other datasets

	#priorities:
		#1: unsupervised learning and dimensionality reduction (try TSNE https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
		#2: PCA, cross-validation error, other classifiers and encodings

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

def processTweet(tweet):
	tokens = tknzr.tokenize(tweet[0:-1])
	tokens = [x for x in tokens if x not in stopwords.words('english')]
	tokens = [x for x in tokens if x not in string.punctuation]
	return tokens


#return tuple of processed tweet string, label
def get_tweet_and_label(tweet_number):
	tmp = df_data.iloc[tweet_number].to_numpy()
	return processTweet(tmp[2]), tmp[1]


#return random sample of size n from data ndarray
def sample(data, n):
	print("sampling with %d samples" % n)
	return data[np.random.choice(data.shape[0], n, replace=False)]


#train tf-idf vectorizer or get existing model
def get_vectorizer(search=True, vectorizer_name='vectorizer.joblib'):
	if (search and path.exists(vectorizer_name)):
		print("found vectorizer")
		vectorizer = load('vectorizer.joblib')
	else:
		print("did not find trained tf-idf vectorizer, commencing training")
		vectorizer = TfidfVectorizer(tokenizer=processTweet, max_features=20000)
		df_data = pd.read_csv('twitter-sentiment-analysis-hatred-speech/combined.csv')
		vectorizer.fit(df_data.to_numpy().T[0])
		dump(vectorizer, 'vectorizer.joblib')
	return vectorizer


#vectorize tweets in a given file
#optionally save the vector representation of them
def vectorize_tweets(tweet_file, save_path='null'):
	vectorizer = get_vectorizer()
	df_data = pd.read_csv(tweet_file, names=('id','label','tweet'),header=None)
	print(df_data.to_numpy().shape)
	tweets = vectorizer.transform(df_data.to_numpy().T[2])
	if (save_path != 'null'):
		np.save(save_path, tweets)
	return tweets

def get_vectorized_tweets(vector_file_path):
	return np.load(vector_file_path, allow_pickle=True).tolist()
