import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import mglearn
import pandas as pd

citibike = mglearn.datasets.load_citibike().resample("1h").sum().fillna(0)
print("City Bike :\n{}".format(citibike))