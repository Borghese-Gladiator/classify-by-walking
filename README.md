# Classify By Walking
CS 328 Final Project (Spring 2021)

## Methodology
1. [Gathering Data](#gathering-data)
2. Load CSV into numpy array 
3. [Filtering Walking Data](#filtering-walking-data)
4. [Feature Extraction from Inertial Data](#feature-extraction-from-inertial-data)
5. [Trained & Evaluated Classifiers](#trained--evaluated-classifiers)
6. Wrote best_classifier to /training_output/
7. [Conclusion](#conclusion)

## Gathering Data
Gathered 5 minutes of walking data per subject through accelerometer signal by using Physics Toolbox app.
- Daniel (male)
- Tim (male)
- Wen-Ling (female)

## Filtering Walking Data
- Combined X, Y, Z accelerations into one magnitude signal
- Applied butterworth filter to filter out noise from walking data

## Feature Extraction from Inertial Data
Split dataset into time windows and extracted features on given window
- time-domain features
  - mean
  - variance
  - standard deviation
- frequency-domain features
  - dominant frequency
  - entropy

## Trained \& Evaluated Classifiers
- For every model (Decision Tree, Gradient Boost, and Random Forest Classifier)
  - ran 10-fold cross validation
  - calculated total accuracy, precision, and recall

## Conclusions
There is no distinctive difference between genders in gait (manner of walking)