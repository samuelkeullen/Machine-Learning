# Import dataset and classes needed in this example:
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Import Gaussian Naive Bayes classifier:
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset: 
data = load_iris()

# Organize data:
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Print data:
print("Labels:\n{}\n".format(label_names))
print('Class label = {}\n'.format(labels[0]))
print("Features: \n{}\n".format(feature_names))
print("Feature: {}\n".format(features[0]))

# Split dataset into random train and test subsets:
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# Initialize classifier:
gnb = GaussianNB()

# Train the classifier:
model = gnb.fit(train, train_labels)
# Make predictions with the classifier:
predictive_labels = gnb.predict(test)
print("Predictive labels:\n{}\n".format(predictive_labels))

# Evaluate label (subsets) accuracy:
print("Accuracy: {}".format(accuracy_score(test_labels, predictive_labels)))