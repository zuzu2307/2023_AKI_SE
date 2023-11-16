from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'))

X = newsgroups.data
y = newsgroups.target[:1000]

vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vectorized = vectorizer.fit_transform(X).toarray()[:1000]

km = KMeans(n_clusters=20)  
y_pred = km.fit_predict(X_vectorized)

linkage_types = ['ward', 'complete', 'average', 'single']

for linkage_type in linkage_types:
    
    print(f"\nUsing linkage type: {linkage_type}")
    model = AgglomerativeClustering(
        n_clusters = 20,  
        linkage = linkage_type  
    )
    clustering = model.fit(X_vectorized)

    # use adjusted_rand_score to compare the clustering labels to the ground truth class labels
    ari_score = adjusted_rand_score(y, clustering.labels_)
    print(f"Adjusted Rand Index for {linkage_type} linkage: {ari_score}")
