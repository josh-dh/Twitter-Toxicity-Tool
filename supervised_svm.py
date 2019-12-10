#SUPERVISED MODEL
from model import *
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
df_data = pd.read_csv("twitter-sentiment-analysis-hatred-speech/train.csv",names=('id','label','tweet'),header=None)
data_ = df_data.to_numpy()

SVC NO PCA
data = get_vectorized_tweets('training_vecs.npy')
labels = data_.T[1].astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33)
classifier = LinearSVC(C=10.0, loss='squared_hinge', penalty='l2')
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

PCA TRANSFORM
data = get_vectorized_tweets('training_vecs.npy').toarray()
pca = PCA(200, svd_solver='randomized')
print("start pca")
print(data)
data = pca.fit_transform(data)
np.save('p_training_vecs.npy', data)
print("finish pca")

SVC PCA
data = np.asarray(get_vectorized_tweets('pca_training_vecs.npy'))
labels = data_.T[1].astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33)
classifier = LinearSVC(C=10.0, loss='squared_hinge', penalty='l2')
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
print(y_train)
data_ = sample(data, 500)
classified_labels = classifier.predict(data)
print(classified_labels)

print("start dimensionality reduction")
svd_model = TruncatedSVD(n_components=50)
data_svd = svd_model.fit_transform(data)
print("start TSNE")
tsne_model = TSNE(n_components = 2)
data_tsne = tsne_model.fit_transform(data_svd)
print(data_tsne.shape)

import matplotlib.pyplot as plt
print("scatter:")
plt.scatter(data_tsne[:,0], data_tsne[:,1], c = classified_labels)
plt.show()

#CONVNET MODEL
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D

model = Sequential()
model.add(Dense(200, activation='sigmoid'))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

data = np.asarray(get_vectorized_tweets('pca_training_vecs.npy'))
labels = data_.T[1].astype(np.int_)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.33)

model.fit(X_train, y_train,
          batch_size=128,
          epochs=100,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=1024)
print('Test score:', score)
print('Test accuracy:', acc)