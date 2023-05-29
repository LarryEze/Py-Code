''' Basic Modeling in scikit-learn '''

'''
Basic Modeling in scikit-learn
What is model validation?
- Model validation consist of:
* various steps and processes that ensure your model performs as expected  on new data
* The most common way to do this is to test the model's accuracy on data it has never seen before (called a holdout set).
- It also involves choosing the best model, parameters and accuracy metrics
* Its goal is to have the best performing model possible, that achieves high accuracy for the data given.

scikit-learn modeling review
Basic modeling steps:
model = RandomForestRegressor(n_estimators = 500, random_state = 1111)
model.fit(X=X_train, y=y_train) -> in

RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1, oob_score=False, random_state=1111, verbose=0, warm_start=False) -> out

predictions = model.predict(X_test)
print('{0:.2f}'.format(mae(y_true=y_test, y_pred=predictions))) -> in

10.84 -> out

- mae (Mean Absolute Error)

Seen vs. unseen data
Training data = seen data
model = RandomForestRegressor(n_estimators=500, random_state=1111)
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)

Testing data = unseen data
model = RandomForestRegressor(n_estimators=500, random_state=1111)
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
'''

# The model is fit using X_train and y_train
from sklearn.metrics import precision_score, make_scorer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
model.fit(X_train, y_train)

# Create vectors of predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Train/Test Errors
train_error = mae(y_true=y_train, y_pred=train_predictions)
test_error = mae(y_true=y_test, y_pred=test_predictions)

# Print the accuracy for seen and unseen data
print("Model error on seen data: {0:.2f}.".format(train_error))
print("Model error on unseen data: {0:.2f}.".format(test_error))


'''
Regression models
- There are 2 types of predictive models
* Models built for continuous variables, or Regression models
* Models built for categorical variables, or Classifiction models

Random forests in scikit-learn
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

rfr = RandomForestRegressor(random_state = 1111)
rfc = RandomForestClassifier(random_state = 1111)

Decision Tree Example: Predicted Amount of Debt
                                Are you left-handed?
                            Y /                     \ N
            Are you older than 20?                  Do you have blue eyes?
                Y /     \ N                             Y /     \ N
Do you have siblings?   Do you like onions?             $14k    $15k
    Y /     \ N                 Y /     \ N
    $10k    $8k                 $4k      $3k

- A new observation will follow the tree based on its own data values until it reaches an end-node (called a leaf).

Decision Tree #1: $4k   \
Decision Tree #2: $4k   \
Decision Tree #3: $3k   -   (4 + 4 + 3 + 5 + 5) / 5 = 4.2
Decision Tree #4: $5k   /
Decision Tree #5: $5k   /

Random forest parameters
- n_estimators: The number of trees to create for the forest
- max_depth: The maximum depth of the trees, or how many times we can split the data
- random_state: random seed

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 50, max_depth = 10)

rfr = RandomForestRegressor(random_State = 1111)
rfr.n_estimators = 50
rfr.max_depth = 10

Feature importance
Print how important each column is to the model

for i , item in enumerate(rfr.feature_importances_):
    print('{0:s}: {1:.2f}'.format(X.columns[i], item))

weight: 0.50
height: 0.39
left_handed: 0.72
union_preference: 0.05
eye_color: 0.03 -> out
'''

# Set the number of trees
rfr.n_estimators = 100

# Add a maximum depth
rfr.max_depth = 6

# Set the random state
rfr.random_state = 1111

# Fit the model
rfr.fit(X_train, y_train)


# Fit the model using X and y
rfr.fit(X_train, y_train)

# Print how important each column is to the model
for i, item in enumerate(rfr.feature_importances_):
    # Use i and item to print out the feature importance of each column
    print("{0:s}: {1:.2f}".format(X_train.columns[i], item))


'''
Classification models
- Categorical Responses:
* Newborn's hair color
* Winner of a basketball game
* Genre of the next song on the radio

The Tic-Tac-Toe dataset
. . .   Bottom-Left   Bottom-Middle   Bottom-Right       Class
. . .   X             O               O               positive
. . .   O             X               O               positive
. . .   O             O               X               positive
. . .   X             X               O               negative
. . .         . . .           . . .           . . .      . . . 

Using .predict() for classification
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier
(random_state = 1111)
rfc.fit(X_train, y_train)
rfc.predict(X_test) -> in

array([1, 1, 1, 1, 0, 1, . . . ]) -> out

pd.Series(rfc.predict(X_test)).value_counts() -> in

1   627
0   331 -> out

Predicting probabilities
rfc.predict_proba(X_test) -> in

array([[0. , 1. ], [0.1, 0.9], [0.1, 0.9], . . .]) -> out

rfc = RandomForestClassifier
(random_state = 1111)
rfc.get_params() -> in

{'bootstrap': True, 'class_weight': None, 'criterion': 'gini', . . .} -> out

rfc.fit(X_train, y_train)
rfc.score(X_test, y_test) -> in

0.8989 -> out

- .score() is a quick way to look at the overall accuracy of the classification model.
'''

# Fit the rfc model.
rfc.fit(X_train, y_train)

# Create arrays of predictions
classification_predictions = rfc.predict(X_test)
probability_predictions = rfc.predict_proba(X_test)

# Print out count of binary predictions
print(pd.Series(classification_predictions).value_counts())

# Print the first value from probability_predictions
print('The first predicted probabilities are: {}'.format(
    probability_predictions[0]))


rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Print the classification model
print(rfc)

# Print the classification model's random state parameter
print('The random state is: {}'.format(rfc.random_state))

# Print all parameters
print('Printing the parameters dictionary: {}'.format(rfc.get_params()))


# Create a random forest classifier
rfc = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=1111)

# Fit rfc using X_train and y_train
rfc.fit(X_train, y_train)

# Create predictions on X_test
predictions = rfc.predict(X_test)
print(predictions[0:5])

# Print model accuracy using score() and the testing data
print(rfc.score(X_test, y_test))


''' Validation Basics '''

'''
Creating train, test, and validation datasets
Traditional train / test split
- Seen data (used for training)
- Unseen data (unavailable for training)
- Testing (holdout sample): It is any data that is not used for training and is only used to asses model performance

Dataset definitions and ratios
Dataset                     Definition
Train                       The sample of datat used when fitting models
Test (holdout sample)       The sample of data used to assess model performance

- Ratio Examples
* 80 : 20
* 90 : 10 (used when we have little data)
* 70 : 30 (used when model is computationally expensive)

The X and y datasets
import pandas as pd

tic_tac_toe = pd.read_csv('tic-tac-toe.csv')
X = pd.get_dummies(tic_tac_toe.iloc[:, 0:9])
y = tic_tac_toe.iloc[:, 9]

Creating holdout samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

- Parameters:
* test_size
* train_size
* random_state

Dataset for preliminary testing?
What do we do when testing different model parameters?
- 100 versus 1000 trees

            Available Data
            /            \
        Training    Testing (holdout sample)

                        Available Data
            /               |                      \
    Training            Validation                  Testing 
                (validation holdout sample) ( testing holdout sample)

- Validation holdout sample are used to assess our model's performance when using different parameter values.

Train, validation, test continued
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=1111)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=1111)

- NB*: Validation holdout sample is only useful when testing parameters, tuning hyper-parameters, or anytime you are frequently evaluating model performance.
'''

# Create dummy variables using pandas
X = pd.get_dummies(tic_tac_toe.iloc[:, 0:9])
y = tic_tac_toe.iloc[:, 9]

# Create training and testing datasets. Use 10% for the test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=1111)


# Create temporary training and final testing datasets
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1111)

# Create the final training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=1111)


'''
Accuracy metrics: regression models
Mean absolute error (MAE)
- It is used to assess the performance of a regression model.
- It is the simplest and most intuitive error metric and is the average absolute difference between the prdictions (y(i) and the actual values (y(i) hat).
- This metric treats all points equally and is not sensitive to outliers.
- It is used when dealing with applications where we don't want large errors to have a major impact

Mean squared error (MSE)
- It is the most widely used regression error metric for regression models
- It is calculated similarly to the mean absolute error, but this tim, we square the difference term.
- It allows larger errors (outliers) to have a larger impact on the model

MAE vs. MSE
- Accuracy metrics are always application specific
- MAE and MSE error terms are in different units and should not be compared

Mean absolute error
rfr = RandomForestRegressor(n_estimators=500, random_state=1111)
rfr.fit(X_train, y_train)
test_predictions = rfr.predict(X_test)
sum(abs(y_test - test-predictions)) / len(test_predictions) -> in

9.99 -> out

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, test_predictions) -> in

9.99 -> out

Mean squared error
sum(abs(y_test - test-predictions) ** 2) / len(test_predictions) -> in

141.4 -> out

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, test_predictions) -> in

141.4 -> out

Accuracy for a subset of data
chocolate_preds = rfr.predict(X_test[X_test[:, 1] == 1])
mean_absolute_error(y_test[X_test[:, 1] == 1], chocolate_preds) -> in

8.79 -> out

nonchocolate_preds = rfr.predict(X_test[X_test[:, 1] == 0])
mean_absolute_error(y_test[X_test[:, 1] == 0], nonchocolate_preds) -> in

10.99 -> out
'''


# Manually calculate the MAE
n = len(predictions)
mae_one = sum(abs(y_test - predictions)) / n
print('With a manual calculation, the error is {}'.format(mae_one))

# Use scikit-learn to calculate the MAE
mae_two = mean_absolute_error(y_test, predictions)
print('Using scikit-learn, the error is {}'.format(mae_two))


n = len(predictions)
# Finish the manual calculation of the MSE
mse_one = sum((y_test - predictions)**2) / n
print('With a manual calculation, the error is {}'.format(mse_one))

# Use the scikit-learn function to calculate MSE
mse_two = mean_squared_error(y_test, predictions)
print('Using scikit-learn, the error is {}'.format(mse_two))


# Find the East conference teams
east_teams = labels == "E"

# Create arrays for the true and predicted values
true_east = y_test[east_teams]
preds_east = predictions[east_teams]

# Print the accuracy metrics
print('The MAE for East teams is {}'.format(mae(true_east, preds_east)))

# Print the West accuracy
print('The MAE for West conference is {}'.format(west_error))


'''
Classification metrics
- Precision
- Recall (also called sensitivity)
- Accuracy
- Specificity
- F1-Score, and its variations
- . . . 

Confusion matrix
                        Predicted Values
                        0         1
                0     23 (TN)    7 (FP)
Actual Values   1      8 (FN)   62 (TP)

True Positive: Predict / Actual are both 1
True Negative: Predict / Actual are both 0
False Positive: Predicted 1, actual 0
False Negative: Predicted 0, actual 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm) -> in

array([ [ 23,  7],
        [  8, 62]   ]) -> out

cm[<true_category_index>, < predicted_category_index>]
cm[1, 0] -> in

8 -> out

Accuracy
- It is the overall ability of the model to correctly predict the correct classification.

                Predicted Values
                    0     1
                0  23     7
Actual Values   1   8    62

Accuracy = ( 23 (TN) + 62 (TP) ) / ( 23 + 7 + 8 + 62 ) = 0.85

Precision
- It is the number of TP (true positives) out of all predicted positive values.
- It is used when we don't want to overpredict positive values.

                Predicted Values
                    0     1
                0  23     7
Actual Values   1   8    62

Precision = 62 (TP) / ( 62 (TP) + 7 (FP) ) = 0.90

Recall
- It is about finding all positive values.
- It is used when we can't afford to miss any positive values

                Predicted Values
                    0     1
                0  23     7
Actual Values   1   8    62

Recall = 62 (TP) / ( 62 (TP) + 8 (FN) ) = 0.885 

Accuracy, Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy_score(y_test, test_predictions) -> in

0.85 -> out

precision_score(y_test, test_predictions) -> in

0.8986 -> out

recall_score(y_test, test_predictions) -> in

0.8857 -> out
'''

# Calculate and print the accuracy
accuracy = (324 + 491) / (953)
print("The overall accuracy is {0: 0.2f}".format(accuracy))

# Calculate and print the precision
precision = (491) / (15 + 491)
print("The precision is {0: 0.2f}".format(precision))

# Calculate and print the recall
recall = (491) / (123 + 491)
print("The recall is {0: 0.2f}".format(recall))


# Create predictions
test_predictions = rfc.predict(X_test)

# Create and print the confusion matrix
cm = confusion_matrix(y_test, test_predictions)
print(cm)

# Print the true positives (actual 1s that were predicted 1s)
print("The number of true positives is: {}".format(cm[1, 1]))


test_predictions = rfc.predict(X_test)

# Create precision or recall score based on the metric you imported
score = precision_score(y_test, test_predictions)

# Print the final result
print("The precision value is {0:.2f}".format(score))


'''
The bias-variance tradeoff
Variance
- It occurs when a model pays too close attention to the training data and fails to generalize to the testing data.
- The models perform well (low error) on only the training data, but not the testing data
- It occurs when models are overfit and have high complexity
* Overfitting occurs when our model starts to attach meaning to the noise in the training data

Bias
- It occurs when the model fails to find the relationships between the data and the response value
- It leads to high errors on both the training and testing datasets and is associated with an underfit model
* Underfitting occurs when the model could not find the underlying patterns available in the data.

Optimal performance
- It is when the model is getting the most out of the training data, while still performing on the testing data.

Parameters causing over / under fitting
rfc = RandomForestClassifier(n_estimators=100, max_depth=4)
rfc.fit(X_train, y_train)

print('Training: {0:.2f}'.format(accuracy_score(y_train, train_predictions))) -> in

Training: 0.84 -> out

print('Testing: {0:.2f}'.format(accuracy_score(y_test, test_predictions))) -> in

Testing: 0.77 -> out

rfc = RandomForestClassifier(n_estimators=100, max_depth=14)
rfc.fit(X_train, y_train)

print('Training: {0:.2f}'.format(accuracy_score(y_train, train_predictions))) -> in

Training: 1.0 -> out

print('Testing: {0:.2f}'.format(accuracy_score(y_test, test_predictions))) -> in

Testing: 0.83 -> out

rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
rfc.fit(X_train, y_train)

print('Training: {0:.2f}'.format(accuracy_score(y_train, train_predictions))) -> in

Training: 0.89 -> out

print('Testing: {0:.2f}'.format(accuracy_score(y_test, test_predictions))) -> in

Testing: 0.86 -> out
'''

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25, random_state=1111, max_features=2)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
    mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(mae(y_test, rfr.predict(X_test))))

# Update the rfr model
rfr = RandomForestRegressor(
    n_estimators=25, random_state=1111, max_features=11)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
    mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(mae(y_test, rfr.predict(X_test))))

# Update the rfr model
rfr = RandomForestRegressor(n_estimators=25, random_state=1111, max_features=4)
rfr.fit(X_train, y_train)

# Print the training and testing accuracies
print('The training error is {0:.2f}'.format(
    mae(y_train, rfr.predict(X_train))))
print('The testing error is {0:.2f}'.format(mae(y_test, rfr.predict(X_test))))


test_scores, train_scores = [], []
for i in [1, 2, 3, 4, 5, 10, 20, 50]:
    rfc = RandomForestClassifier(n_estimators=i, random_state=1111)
    rfc.fit(X_train, y_train)
    # Create predictions for the X_train and X_test datasets.
    train_predictions = rfc.predict(X_train)
    test_predictions = rfc.predict(X_test)
    # Append the accuracy score for the test and train predictions.
    train_scores.append(round(accuracy_score(y_train, train_predictions), 2))
    test_scores.append(round(accuracy_score(y_test, test_predictions), 2))
# Print the train and test scores.
print("The training scores were: {}".format(train_scores))
print("The testing scores were: {}".format(test_scores))


''' Cross Validation '''

'''
The problems with holdout sets
Transition validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
out_of_sample = rf.predict(X_test)
print(mae(y_test, out_of_sample)) -> in

10.24 -> out 

Traditional training splits
cd = pd.read_csv('candy-data.csv')
s1 = cd.sample(60, random_state = 1111)
s2 = cd.sample(60, random_state = 1112)

Overlapping candies:
print(len([i for i in s1.index if i in s2.index])) -> in

39 -> out

- Only 39 of the 60 candies overlap between the two datasets

Chocolate candies:
print(s1.chocolate.value_counts()[0])
print(s2.chocolate.value_counts()[0]) -> in

34
30 -> out

- The s1 sample contains 34 chocolate candies, and the s2 sample only contains 30.

The split matters
Sample 1 Testing Error
print('Testing error: {0:.2f}'.format(mae(s1_y_test, rfr.predict(s1_X_test)))) -> in

10.32 -> out

Sample 2 Testing Error
print('Testing error: {0:.2f}'.format(mae(s2_y_test, rfr.predict(s2_X_test)))) -> in

11.56 -> out

- The s2 testing accuracy is over 12% worse. 

Train, validation, test
X_temp, X_val, y_temp, y_val = train_test_split(. . ., random_state=1111)
X_train, X_test, y_train, y_test = train_test_split(. . ., random_state=1111)

rfr = RandomForestRegressor(n_estimators=25, random_State=1111, max_features=4)
rfr.fit(X_train, y_train)
print('Validation error: {0:.2f}'.format(mae(y_test, rfr.predict(X_test)))) -> in

9.18 -> out 

print('Testing error: {0:.2f}'.format(mae(y_val, rfr.predict(X_val)))) -> in

8.98 -> out

Round 2
X_temp, X_val, y_temp, y_val = train_test_split(. . ., random_state=1171)
X_train, X_test, y_train, y_test = train_test_split(. . ., random_state=1171)

rfr = RandomForestRegressor(n_estimators=25, random_State=1111, max_features=4)
rfr.fit(X_train, y_train)
print('Validation error: {0:.2f}'.format(mae(y_test, rfr.predict(X_test)))) -> in

8.73 -> out 

print('Testing error: {0:.2f}'.format(mae(y_val, rfr.predict(X_val)))) -> in

10.91 -> out

- To overcome this limitation of holdout sets, we use Cross-Validation, which is the Gold-standard for model validation.
'''

# Create two different samples of 200 observations
sample1 = tic_tac_toe.sample(200, random_state=1111)
sample2 = tic_tac_toe.sample(200, random_state=1171)

# Print the number of common observations
print(len([index for index in sample1.index if index in sample2.index]))

# Print the number of observations in the Class column for both samples
print(sample1['Class'].value_counts())
print(sample2['Class'].value_counts())


'''
Cross-validation
- This methods run a single model on various training / validation combinations and gives us a lot more confidence in the final metrics

- KFold():
* n_splits: umber of cross-validation splits
* shuffle: boolean indicating to shuffle data before splitting
* random_State: random seed

from sklearn.model_selection import KFold

X = np.array(range(40))
y = np.array([0] * 20 + [1] * 20)

kf = KFold(n_splits = 5)
splits = kf.split(X)
for train_index, test_index in splits:
    print(len(train_index), len(test_index)) -> in

32 8 32 8 32 8 32 8 32 8 -> out

# Print one of the index sets:
print(train_index, test_index) -> in

[ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 . . .]
[32 33 34 35 36 37 38 39] -> out

rfr = RandomForestRegressor(n_estimators=25, random_state=1111)
errors = []
for train_index, val_index in splits:
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]

    rfr.fit(X_train, y_train)
    predictions = rfr.predict(X_val)
    errors.append(<some_accuracy_metric>)
print(np.mean(errors)) -> in

4.25 -> out
'''


# Use KFold
kf = KFold(n_splits=5, shuffle=True, random_state=1111)

# Create splits
splits = kf.split(X)

# Print the number of indices
for train_index, val_index in splits:
    print("Number of training indices: %s" % len(train_index))
    print("Number of validation indices: %s" % len(val_index))


rfc = RandomForestRegressor(n_estimators=25, random_state=1111)

# Access the training and validation indices of splits
for train_index, val_index in splits:
    # Setup the training and validation data
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    # Fit the random forest model
    rfc.fit(X_train, y_train)
    # Make predictions, and print the accuracy
    predictions = rfc.predict(X_val)
    print("Split accuracy: " + str(mean_squared_error(y_val, predictions)))


'''
sklearn's cross_val_score()
cross_val_score()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

- cross_val_score():
* estimator: the model to use
* X: the predictor dataset
* y: the response array
* cv: the number of cross-validation splits (or folds)

cross_val_score(estimator=rfc, X=X, y=y, cv=5)

Using scoring and make_scorer
The cross_val_score scoring parameter:

# Load the Methods
from sklearn.metrics import mean_absolute_error, make_scorer

# Create a scorer
mae_scorer = make_scorer(mean_absolute_error)

# Use the scorer
cross_val_score(,estimator>, <X>, <y>, cv=5, scoring=mae_scorer)

Example:
Load all of the sklearn methods

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

Create a model and a scorer
rfc = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=1111)
mse = make_scorer(mean_squared_error)

Run cross_val_score()
cv_results = cross_val_score(rfc, X, y, cv=5, scoring=mse)

Accessing the results
print(cv_results) -> in

[196.765, 108.563, 85.963, 222.594, 140.942] -> out

Report the mean and standard deviation:
print('The mean: {}'.format(cv_results.mean()))
print('The std: {}'.format(cv_results.std())) -> in

The mean:   150.965
The std:    51.676 -> out
'''

# Instruction 1: Load the cross-validation method

# Instruction 2: Load the random forest regression model

# Instruction 3: Load the mean squared error method
# Instruction 4: Load the function for creating a scorer


rfc = RandomForestRegressor(n_estimators=25, random_state=1111)
mse = make_scorer(mean_squared_error)

# Set up cross_val_score
cv = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10, scoring=mse)

# Print the mean error
print(cv.mean())


'''
Leave-one-out-cross-validation (LOOCV)
- In LOOCV we are going to implement KFold cross-validation, where k is equal to n, the number of observations in the data.
* i.e every single point will be used in a validation set, completely by itself.

When to use LOOCV?
Use when:
- The data is limited and want to use as much training data as possible when fitting the model.
- You want the absolute best error estimate for a single new point (new data).

Be cautious when:
- Computational resources are limited (method is very computationally expensive).
- You have a lot of data
- You have a lot of parameters to test

- The best way to judge if this method is even possible is to run KFold cross-validation with a large K, maybe 25 or 50, and gauge how long it would take to actually run LOOCV with the n-observations in the data.

LOOCV Example
n = X.shape[0]
mse = make_scorer(mean_squared_error)
cv_results = cross_val_score(estimator, X, y, scoring=mse, cv=n)
print(cv_results)-> in

[5.45, 10.52, 6.23, 1.98, 11.27, 9.21, 4.65, . . . ] -> out

print(cv_results.mean()) -> in

6.32 -> out
'''


# Create scorer
mae_scorer = make_scorer(mean_absolute_error)

rfr = RandomForestRegressor(n_estimators=15, random_state=1111)

# Implement LOOCV
scores = cross_val_score(rfr, X=X, y=y, cv=X.shape[0], scoring=mae_scorer)

# Print the mean and standard deviation
print("The mean of the errors is: %s." % np.mean(scores))
print("The standard deviation of the errors is: %s." % np.std(scores))


''' Selecting the best model with Hyperparameter tuning. '''

'''
Introduction to hyperparameter tuning
Model parameters:
Parameter are:
* It is created as the result of fitting a model 
* Learned or estimated from the input data.
* Used when making future predictions
* Not manually set by the modeler.

Linear regression parameters
Parameters are created by fitting a model:

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
print(lr.coef_, lr.intercept_) -> in

[[0.798, 0.452]]  [1.786] -> out

Parameters do not exist before the model is fit:

lr = LinearRegression()
print(lr.coef_, lr.intercept_) -> in

AttributeError: 'LinearRegression' object has no attribute 'coef_' -> out

Model hyperparameters
Hyperparameters:
- Manually set before the training occurs
- Specify how the training is supposed to happen

Random forest hyperparameters
Hyperparameter          Description                                                 Possible Values (default)
n_estimators            Number of decision trees in the forest                      2+ (10)
max_depth               Maximum depth of the decision trees                         2+ (None)
max_features            Number of features to consider when making a split          See documentation
min_samples_split       The minimum number of samples required to make a split      2+ (2)

What is hyperparameter tuning? 
Hyperparameter tuning?
- It consists of selecting hyperparameters to test 
- Running a single model type with various values for these hyperparameters.
- Create ranges of possible values to select from
- Specify a single accuracy metric

Specifying ranges
depth = [4, 6, 8, 10, 12]
samples = [2, 4, 6, 8]
features = [2, 4, 6, 8, 10]

# Specify hyperparameters
rfr = RandomForestRegressor(n_estimators=100, max_depth=depth[0], min_samples_split=samples[3], max_features=features[1])
rfr.get_params() -> in

{'bootstrap': True, 'criterion': 'mse', 'max_depth': 4, 'max_features': 4, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 8, . . . } -> out 

General guidelines
- Start with the basics
- Read through the documentation
- Test practical ranges
'''

# Review the parameters of rfr
print(rfr.get_params())

# Maximum Depth
max_depth = [4, 8, 12]

# Minimum samples for a split
min_samples_split = [2, 5, 10]

# Max features
max_features = [4, 6, 8, 10]


# Fill in rfr using your variables
rfr = RandomForestRegressor(n_estimators=100, max_depth=random.choice(
    max_depth), min_samples_split=random.choice(min_samples_split), max_features=random.choice(max_features))

# Print out the parameters
print(rfr.get_params())


'''
RandomizedSearchCV
Grid Searching
Benefits:
- Tests every possible combination of values

Drawbacks:
- Additional hyperparameters increase training time exponentially
* Therefore, grid searching is only possible with a limited number of parameters, and a limited number of ranges.

Better methods
- Random searching
* It consists of randomly selecting from all hyperparameter values from the list of possible ranges
- Bayesian optimization
* It uses the past results of each test to update te hyperparameters for the next run.

Random search
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV()
Parameter Distribution:
param_dist = {'max_depth': [4, 6, 8, None], 'max_features': range(2, 11), 'min_samples_split': range(2, 11)}

Random search parameters
Parameters:
- estimator: the model to use
- param_distributions: dictionary containing hyperparameters and possible values
- n_iter: number of iterations
- scoring: scoring method to use

Setting RandomizedSearchCV parameters
param_dist = {'max_depth': [4, 6, 8, None], 'max_features': range(2, 11), 'min_samples_split': range(2, 11)}

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error

rfr = RandomForestRegressor(n_estimators = 20, random_state = 1111)
scorer = make_scorer(mean_absolute_error)

RandomizedSearchCV implemented
Setting up the random search:
random_search = RandomizedSearchCV(estimator=rfr, param_distributions=param_dist, n_iter=40, cv=5)

Complete the random search:
random_search.fit(X, y)

- We cannot do hyperparameter tuning without understanding model validation
- Model validation allows us to compare multiple models and parameter sets
'''


# Finish the dictionary by adding the max_depth parameter
param_dist = {"max_depth": [2, 4, 6, 8], "max_features": [
    2, 4, 6, 8, 10], "min_samples_split": [2, 4, 8, 16]}

# Create a random forest regression model
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)

# Create a scorer to use (use the mean squared error)
scorer = make_scorer(mean_squared_error)


# Import the method for random search

# Build a random search using param_dist, rfr, and scorer
random_search = RandomizedSearchCV(
    estimator=rfr, param_distributions=param_dist, n_iter=10, cv=5, scoring=scorer)


'''
Selecting your final model
# Best Score
rs.best_score_ -> in

5.45 -> out

# Best Parameters
rs.best_params_ -> in

{'max_depth': 4, 'max_features': 8, 'min_samples_split': 4} 

# Best Estimator
rs.best_estimator_ 

Other attributes
rs.cv_results_
rs.cv_results_['mean_test_score'] -> in

array([5.45, 6.23, 5.87, 5.91, 5.67]) -> out

# Selected Parameters:
rs.cv_results_['params'] -> in

[   {'max_depth': 10, 'min_samples_split': 8, 'n_estimators': 25},
    {'max_depth': 4, 'min_samples_split': 8, 'n_estimators': 50},
    . . .] -> out

Using .cv_results_
Group the max depths:
max_depth = [item['max_depth'] for item in rs.cv_results_['params']]
scores = list(rs.cv_results_['mean_test_score'])
d = pd.DataFrame([max_depth, scores]).T
d.columns = ['Max Depth', 'Score']
d.groupby(['Max Depth']).mean() -> in

Max Depth   Score
2.0         0.677928
4.0         0.753021
6.0         0.817219
8.0         0.879136
10.0        0.896821 -> out

Uses of the output:
- Visualize the effect of each parameter
- Make inferences on which parameters have big impacts on the results

Selecting the best model
rs.best_estimator_ : contains the information of the best model

rs.best_estimator_ -> in

RandomForestRegressor(bootstrap=True, criterion= 'mse', max_depth=8, max_features=8, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=12, min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1, oob_score=False, random_state=1111, verbose=0, warm_start=False) -> out

Comparing types of models
Random forest:
rfr.score(X_test, y_test) -> in

6.39 -> out

Gradient Boosting:

gb.score(X_test, y_test) -> in

6.23 -> out

Using .best_estimator_
Predict new data:
rs.best_estimator_.predict(<new_data>)

Check th parameters:
random_search.best_estimator_.get_params()

Save model for use later:
from sklearn.externals import joblib

joblib.dump(rfr, 'rfr best <date>.pkl')
'''


# Create a precision scorer
precision = make_scorer(precision_score)
# Finalize the random search
rs = RandomizedSearchCV(estimator=rfc, param_distributions=param_dist,
                        scoring=precision, cv=5, n_iter=10, random_state=1111)
rs.fit(X, y)

# print the mean test scores:
print('The accuracy for each run was: {}.'.format(
    rs.cv_results_['mean_test_score']))
# print the best model score:
print('The best accuracy for a single model was: {}'.format(rs.best_score_))
