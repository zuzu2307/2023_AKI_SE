from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer


data = fetch_20newsgroups(subset="all")

texts = ["文字列のリスト"]
query = ["文字列"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
q = vectorizer.transform(query)

print(X,q)

