import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score


newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'))

# Data and categories
data = newsgroups.data
categories = newsgroups.target

# create a tf-idf vectorizer object and transform the news
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data)


def find_similar_articles(query, X, vectorizer, top_n=5):
    # vectorizing the query (transform to the same vector space)
    query_vec = vectorizer.transform([query])

    # calculating cosine similarity between the query and all other articles
    cosine_sim = cosine_similarity(query_vec, X).flatten()

    # getting indices of top similar articles
    related_docs_indices = np.argsort(cosine_sim)[-top_n:]

    # getting similarity scores
    scores = cosine_sim[related_docs_indices]

    # reversed because argsort gives ascending order
    return related_docs_indices[::-1], scores[::-1]


# select an article to use as a query for demonstration purposes
input_index = 10
input_article = data[input_index]

print(
    f"Selected article category: {newsgroups.target_names[categories[input_index]]}\n")

# finding similar articles
indices, scores = find_similar_articles(input_article, X, vectorizer, top_n=5)

# displaying similar articles
matched_categories = 0

for rank, (idx, score) in enumerate(zip(indices, scores), 1):
    print(f"Rank: {rank}, Similarity Score: {score:.2f}")
    print(f"Content: {data[idx][:100]}...") 
    print(f"Category: {newsgroups.target_names[categories[idx]]}\n")

    if categories[idx] == categories[input_index]:
        matched_categories += 1

