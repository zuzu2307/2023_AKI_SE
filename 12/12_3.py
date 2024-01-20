from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Define the pipeline
pipe = Pipeline([
    ('preprocessing', StandardScaler()), 
    ('classifier', SVC())
])

# Define the parameter grid
param_grid = [
    {
        'preprocessing': [StandardScaler(), None],
        'classifier': [SVC()],
        'classifier__C': [1, 10, 100],
        'classifier__gamma': [0.001, 0.01, 0.1]
    },
    {
        'preprocessing': [StandardScaler(), None],
        'classifier': [KNeighborsClassifier()],
        'classifier__n_neighbors': [3, 5, 7],
    }
]

X_train, X_test, y_train, y_test = train_test_split(
 iris.data, iris.target, random_state=0)

# Create and fit the GridSearchCV
grid_search = GridSearchCV(pipe, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best params:\n{}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Test-set score: {:.2f}".format(grid_search.score(X_test, y_test)))

