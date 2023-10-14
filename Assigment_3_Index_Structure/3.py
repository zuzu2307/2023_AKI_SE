from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict

news = fetch_20newsgroups(subset='all')

news_data = news.data


def simplify(data):
    return data.lower().split()


document = [simplify(doc) for doc in news_data]

index_set = defaultdict(set)
for doc_id, words in enumerate(document):
    for word in words:
        index_set[word].add(doc_id)


def search(keyword, data_set):
    return data_set[keyword]


query = 'university'
doc_ids = search(query, index_set)

print(doc_ids)
print(f"Found {query} in {len(doc_ids)} documents.")
