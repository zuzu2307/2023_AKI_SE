from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


news = fetch_20newsgroups(subset="all")

print(f"{len(news.data)} news")

# create a tf-idf vectorizer object and transform the news
vectorizer = TfidfVectorizer(stop_words='english')
doc_vectors = vectorizer.fit_transform(news.data)

# using first document to find a similar
input_index = 0
input_doc_vector = doc_vectors[input_index]

# checking similarity between the input document vector and all document vectors
cosine_similarities = cosine_similarity(
    input_doc_vector, doc_vectors).flatten()

# sorting documents based on their cosine similarity to the input document and get the most similar documents
related_docs_indices = cosine_similarities.argsort()[
    ::-1]

# limit showing document
limit = 5

print(f"\nMost {limit} similar documents to document #{input_index}:")
# skipping the first result since it's the input document itself
for i, index in enumerate(related_docs_indices[1:], start=1):
    if i > limit:  # let's show top limited matches
        break
    print("---------------------------------------------------------------------------------")
    print(
        f"Document #{index} with similarity score: {cosine_similarities[index]:.2f}")
    print("\nContent:\n", news.data[index], "\n")
