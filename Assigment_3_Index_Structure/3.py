from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict

news = fetch_20newsgroups(subset="all")

news_data = news.data

# split all data in to a word and make all lower for searching easily


def simplify(data):
    return data.lower().split()


# simplify data
document = [simplify(doc) for doc in news_data]

# making a dict set to match number and word then use it to make a set of word from datas
index_set = defaultdict(set)
for doc_id, words in enumerate(document):
    for word in words:
        index_set[word].add(doc_id)


# searching a word in data_set
def search(keyword, data_set):
    return data_set[keyword]


query = "university"
doc_ids = search(query, index_set)

print(doc_ids)
print(f"Found {query} in {len(doc_ids)} documents.")
