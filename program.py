import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Sample dataset: [weight, color] and labels (0: apple, 1: orange)
X_train = np.array([[100, 0], [130, 0], [150, 1], [200, 1]])
y_train = np.array([0, 0, 1, 1])

# Feature scaling (normalize the data)
X_train_normalized = X_train / np.max(X_train, axis=0)

# Create KNN classifier with K=3
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn_classifier.fit(X_train_normalized, y_train)

# New fruit to classify: [weight, color]
new_fruit = np.array([[160, 1]])

# Normalize the new fruit's features
new_fruit_normalized = new_fruit / np.max(X_train, axis=0)

# Predict the class of the new fruit
predicted_class = knn_classifier.predict(new_fruit_normalized)

if predicted_class == 0:
    print("The new fruit is predicted to be an apple.")
else:
    print("The new fruit is predicted to be an orange.")
