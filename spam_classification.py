
from sklearn.tree import DecisionTreeClassifier

# Input features
# First value: Suspicious keyword score
# Second value: Suspicious link (1 = Yes, 0 = No)

features = [
    [5000, 1],
    [10000, 1],
    [2000, 1],
    [7000, 1],
    [100, 0],
    [200, 0],
    [300, 0],
    [150, 0]
]

# Output labels
labels = [
    "spam",
    "spam",
    "spam",
    "spam",
    "not spam",
    "not spam",
    "not spam",
    "not spam"
]

# Create model
clf = DecisionTreeClassifier()

# Train model
clf.fit(features, labels)

# Make predictions
print(clf.predict([[6000, 1]]))
print(clf.predict([[180, 0]]))
