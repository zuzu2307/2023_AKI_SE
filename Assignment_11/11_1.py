from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
X_news = newsgroups.data
y_news = newsgroups.target

print("------------ fetched -----------------")

# Create a pipeline that first vectorizes the text and then applies the SVC model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svc', SVC())
])

# Define a smaller parameter grid due to the computational intensity
param_grid_news = {
    'svc__C': [1, 10],
    'svc__gamma': [0.1, 0.01],
    'svc__kernel': ['rbf', 'sigmoid']
}

# Set up the grid search with cross-validation
grid_search_news = GridSearchCV(pipeline, param_grid_news, cv=3, scoring='accuracy')

print("------------ searching -----------------")

# Fit the grid search model - Note: This might take a considerable amount of time
grid_search_news.fit(X_news, y_news)

print("------------ searched -----------------")

# Best parameters and score
best_parameters_news = grid_search_news.best_params_
best_score_news = grid_search_news.best_score_

print("Best Parameters:", best_parameters_news)
print("Best Score:", best_score_news)