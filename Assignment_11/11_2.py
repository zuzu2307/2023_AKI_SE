from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Support Vector Machine (SVM) classifier
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict_proba(X_test)[:, 1]

# Train a Random Forest classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]

# Calculate ROC curves
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_svm)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

# Plot ROC curves in a single figure for comparison
plt.figure(figsize=(8, 6))
plt.plot(fpr_svm, tpr_svm, label='SVM')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison: SVM vs Random Forest')
plt.legend()
plt.show()


