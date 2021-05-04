from features import FeatureExtractor
import os
import sys
import pickle
import json
import csv
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

# Disable CSV field limit to load long strings in spam_or_not_spam.csv - https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# CONSTANTS
data_file = '.\\spam_or_not_spam.csv'
output_dir = 'training_output'  # directory where the classifier(s) are stored
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# LOAD DATA
with open(data_file, 'r', encoding='Latin1') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    data = np.array(list(reader)).astype(str)

print(headers)
print(data.shape)
print(data[:3])

# FEATURE EXTRACTION

n_features = 3
X = np.zeros((0, n_features))
y = np.zeros(0,)
feature_extractor = FeatureExtractor(debug=False)

for i, datapoint in enumerate(data):
    print("Index: {0}".format(i))
    text_data = datapoint[0]
    label = int(datapoint[-1])  # 0 or 1 for spam - confusion matrix needs integers
    x = feature_extractor.extract_features(text_data)
    if (len(x) != X.shape[1]):
        print("Received feature vector of length {}. Expected feature vector of length {}.".format(
            len(x), X.shape[1]))
    X = np.append(X, np.reshape(x, (1, -1)), axis=0)
    y = np.append(y, label)

print("FINISHED feature extraction")

# TRAIN & EVALUATE CLASSIFIERS
confusion_matrix_labels = [0, 1]
print("---------------------- Decision Tree -------------------------")

total_accuracy = 0.0
total_precision = np.zeros(len(confusion_matrix_labels))
total_recall = np.zeros(len(confusion_matrix_labels))

cv = KFold(n_splits=10, shuffle=True, random_state=None)
for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    print("Fold {} : Training decision tree classifier over {} points...".format(
        i, len(y_train)))
    sys.stdout.flush()
    tree.fit(X_train, y_train)
    print("Evaluating classifier over {} points...".format(len(y_test)))

    # predict the labels on the test data
    y_pred = tree.predict(X_test)
    print(y_pred)
    print(y_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=confusion_matrix_labels)

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  
print("\n")

print("---------------------- Random Forest Classifier -------------------------")
total_accuracy = 0.0
total_precision = np.zeros(len(confusion_matrix_labels))
total_recall = np.zeros(len(confusion_matrix_labels))

for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    print("Evaluating classifier over {} points...".format(len(y_test)))
    # predict the labels on the test data
    y_pred = clf.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=confusion_matrix_labels)

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))  
print("\n")

print("---------------------- Gradient Boosting Classifier -------------------------")
total_accuracy = 0.0
total_precision = np.zeros(len(confusion_matrix_labels))
total_recall = np.zeros(len(confusion_matrix_labels))

for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Fold {} : Training Gradient Boosting Classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

    clf.fit(X_train, y_train)

    print("Evaluating classifier over {} points...".format(len(y_test)))
    # predict the labels on the test data
    y_pred = clf.predict(X_test)

    # show the comparison between the predicted and ground-truth labels
    conf = confusion_matrix(y_test, y_pred, labels=confusion_matrix_labels)

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))

    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
   
print("The average accuracy is {}".format(total_accuracy/10.0))  
print("The average precision is {}".format(total_precision/10.0))    
print("The average recall is {}".format(total_recall/10.0))

# SAVE BEST CLASSIFIER
best_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
best_classifier.fit(X,y) 

classifier_filename = 'classifier.pickle'
print("Saving best classifier to {}...".format(os.path.join(output_dir, classifier_filename)))
with open(os.path.join(output_dir, classifier_filename), 'wb') as f: # 'wb' stands for 'write bytes'
    pickle.dump(best_classifier, f)
