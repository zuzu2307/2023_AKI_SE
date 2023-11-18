import mglearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: ",X.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
ridge = Ridge(alpha=1).fit(X_train,y_train)
print("Training set score: {:2f}".format(ridge.score(X_train,y_train)))
print("Test set score: {:2f}".format(ridge.score(X_test,y_test)))