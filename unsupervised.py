#UNSUPERVISED MODEL
from model import *
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


Sparse SVD on tf-idf to reduce features to 50
print("start dimensionality reduction")
data = get_vectorized_tweets('training_vecs.npy').toarray()
svd_model = TruncatedSVD(n_components=50)
data_svd = svd_model.fit_transform(data)
print("start TSNE")
tsne_model = TSNE(n_components = 2)
data_tsne = tsne_model.fit_transform(data_svd)
np.save('tsne_training_data.npy', data_tsne)
data_tsne = sample(np.asarray(get_vectorized_tweets('tsne_training_data.npy')), 500)
print(data_tsne.shape)
cluster_labels = KMeans(n_clusters = 5).fit(data_tsne).labels_

import matplotlib.pyplot as plt
print("scatter:")
plt.scatter(data_tsne[:,0], data_tsne[:,1], c = cluster_labels)
plt.show()

#UNSUPERVISED MODEL ONLY TOXIC SPEECH
#select only toxic speech
df_data = pd.read_csv("twitter-sentiment-analysis-hatred-speech/train.csv",names=('id','label','tweet'),header=None)
labels = df_data.to_numpy().T[1]
data_tsne = np.asarray(get_vectorized_tweets('tsne_training_data.npy'))

print(labels.shape)
print(data_tsne.shape)

labels_data_list = []
for i in range(labels.shape[0]):
	if labels[i] == 1:
		labels_data_list.append(data_tsne[i])

data = sample(np.asarray(labels_data_list), 500)
cluster_labels = KMeans(n_clusters = 5).fit(data).labels_

import matplotlib.pyplot as plt
print("scatter:")
print(cluster_labels)
plt.scatter(data[:,0], data[:,1], c = cluster_labels)
plt.show()