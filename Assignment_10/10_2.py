import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import mglearn
import pandas as pd

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]

    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("Test-set R^2: {:.2f}".format(regressor.score(X_test,y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10,3)) 
    xticks = pd.date_range(start=citibike.index.min(),end=citibike.index.max(),freq="D")
    plt.xticks(xticks,xticks.strftime("%a %m-%d"), rotation = 90, ha = "left")

    plt.plot(range(n_train), y_train, label="train")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
    plt.plot(range(n_train), y_pred_train, '--', label="prediction train")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="prediction test")

    plt.legend(loc=(1.01,0))
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()


citibike = mglearn.datasets.load_citibike().resample("1h").sum().fillna(0)
y = citibike.values
regressor = RandomForestRegressor(n_estimators=100,random_state=0)
X_hour = citibike.index.hour.values.reshape(-1, 1)
n_train = 184


eval_on_features(X_hour,y,regressor)