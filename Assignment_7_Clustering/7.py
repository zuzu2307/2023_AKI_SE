from matplotlib import pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X).toarray()

linkage_type = 'ward'


print(f"Using linkage type: {linkage_type}")
linked = linkage(X_vectorized, method=linkage_type)

max_d = 50  
cluster_labels = fcluster(linked, max_d, criterion='distance')
assert len(cluster_labels) == len(y), "Mismatch in number of cluster labels and original labels"

ari_score = adjusted_rand_score(y, cluster_labels)
print(f"Adjusted Rand Index for {linkage_type} linkage: {ari_score}")

# plotting graph
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=3) 
plt.title(f"Dendrogram ({linkage_type})")
plt.show()

