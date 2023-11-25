# 必要なライブラリのインポート
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Breast Cancerデータの読み込み
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# トレーニングセットとテストセットにデータを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 決定木
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt_clf.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_clf.predict(X_test))

# ランダムフォレスト
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))

# 勾配ブースティング回帰木
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
gb_train_acc = accuracy_score(y_train, gb_clf.predict(X_train))
gb_test_acc = accuracy_score(y_test, gb_clf.predict(X_test))

# サポートベクタマシン
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
svm_train_acc = accuracy_score(y_train, svm_clf.predict(X_train))
svm_test_acc = accuracy_score(y_test, svm_clf.predict(X_test))

# 多層パーセプトロン
mlp_clf = MLPClassifier()
mlp_clf.fit(X_train, y_train)
mlp_train_acc = accuracy_score(y_train, mlp_clf.predict(X_train))
mlp_test_acc = accuracy_score(y_test, mlp_clf.predict(X_test))

# 結果の表示
print("Decision Tree - Training Accuracy:", dt_train_acc, "Testing Accuracy:", dt_test_acc)
print("Random Forest - Training Accuracy:", rf_train_acc, "Testing Accuracy:", rf_test_acc)
print("Gradient Boosting - Training Accuracy:", gb_train_acc, "Testing Accuracy:", gb_test_acc)
print("Support Vector Machine - Training Accuracy:", svm_train_acc, "Testing Accuracy:", svm_test_acc)
print("Multi-layer Perceptron - Training Accuracy:", mlp_train_acc, "Testing Accuracy:", mlp_test_acc)
