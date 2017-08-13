import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from __future__ import division
import os
import json

# Importing the dataset
with open(os.path.join('data', 'beacon-dataset.json'), 'r') as infile:
    data = json.load(infile)

X = data['x']  # input: signal strengths
y = data['y']  # output: area id

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling(needed bcos SVM lib doesnt have feature scaling feature by default)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
predictions = classifier.predict(X_test)

# Making the Confusion Matrix and check if there is any false prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)

num_correct_predictions = 0
for i, y_predicted in enumerate(predictions):
    print(i,y_predicted)
    predicted_category_id = y_predicted
    true_category_id = y_test[i]
    is_prediction_correct = predicted_category_id == true_category_id
    print(predicted_category_id, true_category_id, is_prediction_correct)
    num_correct_predictions += 1 if is_prediction_correct else 0

final_accuracy = num_correct_predictions / len(y_test)

print('Final training accuracy: {}'.format(final_accuracy))
