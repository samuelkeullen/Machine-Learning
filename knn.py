import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import BaggingRegressor

url = (
     "https://archive.ics.uci.edu/ml/machine-learning-databases"
     "/abalone/abalone.data"
)

abalone = pd.read_csv(url, header=None)

abalone.columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole weight",
    "Shucked weight",
    "Viscera weight",
    "Shell weight",
    "Rings",
]

abalone = abalone.drop("Sex", axis=1)

print(abalone.head())

abalone["Rings"].hist(bins=15)

plt.savefig("plt1 knn.png")

correlation_matrix = abalone.corr()
correlation_matrix["Rings"]

print("\nCorrelation Matrix(Rings):\n{}\n".format(correlation_matrix["Rings"]))

#FOR CALCULATE DISTANCE MATEMATICALLY WITH NUMPY

# a = np.array([2, 2])
# b = np.array([4, 4])
# np.linalg.norm(a - b)


X = abalone.drop("Rings", axis=1)
X = X.values
y = abalone["Rings"]
y = y.values

new_data_point = np.array([
    0.569552,
    0.446407,
    0.154437,
    1.016849,
    0.439051,
    0.222526,
    0.291208,
])

distances = np.linalg.norm(X - new_data_point, axis=1)

k = 3
nearest_neighbor_ids = distances.argsort()[:k]
print("Nearest neighbor IDs: {}".format(nearest_neighbor_ids))

nearest_neighbor_rings = y[nearest_neighbor_ids]
print("Nearest neighbor rings: {}".format(nearest_neighbor_rings))

prediction = nearest_neighbor_rings.mean()

print("Mean of ring more near: {}".format(prediction))


#EVALUATE MODEL ABOVE

print("\nNow I'll evaluate the Model.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

knn_model = KNeighborsRegressor(n_neighbors=3)

knn_model.fit(X_train, y_train)

train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print("\nRMSE of Train: {}".format(rmse))

test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print("\nRMSE of Test: {}".format(rmse))


cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()

points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=test_preds, s=50, cmap=cmap
)

f.colorbar(points)
plt.savefig("plt2 knn.png")


cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cmap
)
f.colorbar(points)
plt.savefig("plt3 knn.png")

from sklearn.model_selection import GridSearchCV
parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)

print("\nBest params in Gridsearch: {}\n".format(gridsearch.best_params_))

train_preds_grid = gridsearch.predict(X_train)
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)


print("RMSE Train: {}\n".format(train_rmse))
print("RMSE Test: {}\n".format(test_rmse))

#NOW I'LL USE MORE PARAMETERS

print("Now I'll use more parameters.\n")

parameters = {
    "n_neighbors": range(1, 50),
    "weights": ["uniform", "distance"],
}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train)


print("Best params: {}\n".format(gridsearch.best_params_))

test_preds_grid = gridsearch.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)

print("RMSE Test: {}\n".format(test_rmse))

print("Now i'll use bagging to improve\n")

best_k = gridsearch.best_params_["n_neighbors"]
best_weights = gridsearch.best_params_["weights"]

bagged_knn = KNeighborsRegressor(
    n_neighbors=best_k, weights=best_weights
)

from sklearn.ensemble import BaggingRegressor
bagging_model = BaggingRegressor(bagged_knn, n_estimators=100)

bagging_model.fit(X_train, y_train)

test_preds_grid = bagging_model.predict(X_test)
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
#test_rmse

print("RMSE Test: {}\n".format(test_rmse))

print("this script have 4 models: 'Arbitrary K', 'GridSearchCV for K', 'GridSearchCV for k and weights' and 'Bagging and GridSearchCV'" )