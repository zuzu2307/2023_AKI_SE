from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Load the Iris dataset
iris = load_iris()

# Define the pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('classifier', SVC())
])


# Create a parameter grid
param_grid = [
    {
        'classifier': [SVC()],
        'classifier__C': [1, 10, 100],
        'classifier__gamma': [0.001, 0.01, 0.1],
        'scaler': [StandardScaler(), None]
    },
    {
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7],
        'scaler': [StandardScaler(), None]
    }
]


X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, random_state=0)


# Instantiate GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5)

# Fit the GridSearchCV Model
grid_search.fit(X_train, y_train)


print("Best params:\n{}\n".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Test-set score: {:.2f}".format(grid_search.score(X_test, y_test)))

