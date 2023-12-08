from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

california_housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(california_housing.data,california_housing.target,random_state= 0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

poly = PolynomialFeatures(degree = 2).fit(X_train_scaled)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.fit_transform(X_test_scaled)


ride = Ridge().fit(X_train_scaled,y_train)
print("Ridge regression score without interaction:{:.3f}".format(ride.score(X_test_scaled,y_test)))
ride = Ridge().fit(X_train_poly,y_train)
print("Ridge regression score with interaction:{:.3f}".format(ride.score(X_test_poly,y_test)))

rf = RandomForestRegressor(n_estimators = 100).fit(X_train_scaled,y_train)
print("Random Forest score without interaction:{:.3f}".format(rf.score(X_test_scaled,y_test)))
rf = RandomForestRegressor(n_estimators = 100).fit(X_train_poly,y_train)
print("Ridge regression score with interaction:{:.3f}".format(rf.score(X_test_poly,y_test)))