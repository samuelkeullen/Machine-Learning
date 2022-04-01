import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

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
#plt.show()

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)

