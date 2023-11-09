import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# use SVC classifier
svm_classifier = SVC(kernel='linear')

# Perform 5-fold cross-validation and evaluate
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# create a pipeline that vectorizes and then runs the classifier
pipeline = make_pipeline(TfidfVectorizer(stop_words='english'), SVC(kernel='linear'))

# To store metrics for each fold
accuracies, precisions, recalls, f1s = [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # train the classifier
    pipeline.fit(X_train, y_train)
    
    # make predictions
    predictions = pipeline.predict(X_test)
    
    # print the classification report for this fold
    report = classification_report(y_test, predictions, target_names=newsgroups.target_names, zero_division=0)
    print(report)
    
    # calculate metrics
    accuracies.append(cross_val_score(pipeline, X_test, y_test, cv=kf, scoring='accuracy').mean())
    precisions.append(cross_val_score(pipeline, X_test, y_test, cv=kf, scoring='precision_macro').mean())
    recalls.append(cross_val_score(pipeline, X_test, y_test, cv=kf, scoring='recall_macro').mean())
    f1s.append(cross_val_score(pipeline, X_test, y_test, cv=kf, scoring='f1_macro').mean())

# print the average of the metrics
print(f"Average Accuracy: {np.mean(accuracies):.2f}")
print(f"Average Precision (Macro): {np.mean(precisions):.2f}")
print(f"Average Recall (Macro): {np.mean(recalls):.2f}")
print(f"Average F1-Score (Macro): {np.mean(f1s):.2f}")
