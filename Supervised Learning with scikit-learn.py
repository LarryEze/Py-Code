''' Classification '''

'''
Machine learning with scikit-learn
What is machine learning?
- Machine learning is the process whereby:
* Computers are given the ability to learn to mae decisions from data
* without being explicitly programmed!

Examples of machine learning:
- learning to predict whether an email is spam or not spam given its content and sender.
- learning to cluster books into different categories based on the words they contain, then assigning any new book to one of the existing clusters

Unsupervised learning
- It is the process of uncovering hidden patterns and structures from unlabeled data
- Example:
* Grouping customers into distinct categories based on their purchasing behaviour without knowning in advance what these categories are (Clustering) 

Supervised learning
- It is a type of machine learning where the values to be predicted are already known
- Aim: The model is built to accurately predict the  target values of unseen data and it uses given features
- Examples:
* Predicting a basketball players position based on their points per game

Types of supervised learning
Classification: It is used to predict the label, or category, of an observation (i.e target variables consists of categories).
- Examples:
* Predicting whether a bank transaction is fraudulent or not (aka binary classification)

Regression: It is used to predict continuous values (i.e target variable is continuous)
- Examples:
* Predicting the price of a property from features such as no of berooms, and size of a property

Naming conventions
- Feature = predictor variable = independent variable
- Target variable = dependent variable = response variable

Before you use supervised learning
- Requirements:
* No missing values
* Data in numeric format
*  Data stored in pandas DataFrame or NumPy array

- Perform Exploratory Data Analysis (EDA) first

scikit-learn syntax
from sklearn.module import Model
model = Model()
model.fit(X, y)
predictions = model.predict(X_new)
print(predictions) -> in

array([0, 0, 0, 0, 1, 0]) -> out
'''


'''
The classification challenge
Classifying labels of unseen data
1. Build a model
2. Model learns from the labeled data we pass to it
3. Pass unlabeled data to the model as input
4. Model predicts the labels of the unseen data

* Labeled data = training data

K-Nearest Neighbors
- The idea is to predict the label of any data point by 
* looking at the k (closest) labeled data points
* Taking a majority vote 

Using scikit-learn to fit a classifier
from sklearn.neighbors import KNeighborsClassifier
X = churn_df[['total_day_charge', 'total_eve_charge']].values
y = churn_df['churn'].values
print(X.shape, y.shape) -> in

(3333, 2), (3333, ) -> out

knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(X, y)

Predicting on unlabeled data
X_new = np.array([[56.8, 17.5], [24.4, 24.1], [50.1, 10.9]])
print(X_new.shape) -> in

(3, 2) -> out

predictions = knn.predict(X_new)
print('Predictions: {}'.format(predictions)) -> in

Predictions: [1, 0, 0] -> out
'''

# Import KNeighborsClassifier

# Create arrays for the features and the target variable
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
y = churn_df["churn"].values
X = churn_df[["account_length", "customer_service_calls"]].values

# Create a KNN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)


X_new = np.array([[30.0, 17.5], [107.0, 24.1], [213.0, 10.9]])

# Predict the labels for the X_new
y_pred = knn.predict(X_new)

# Print the predictions for X_new
print("Predictions: {}".format(y_pred))


'''
Measuring model performance
- In classification, accuracy is a commonly used metric
- Accuracy: no of correct predictions / total no of observations

How do we measure accuracy?
- Could compute accuracy on the data used to fit the classifier
* The performance will not be indicative of how well it can generalize to unseen data

Computing accuracy
                                    Split data
                                /               \
                    Training set                 Test set
                        |                              \
Fit / train classifier on training set     ->          Calculate accuracy using test set

Train /test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

* It is best practice to ensure our split reflects the proportion of labels inout data. e.g if churn occurs in 10% of observations, we want 10% of labelss in out training and test sets to represent churn and it is achieved by setting stratify equal to y (stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test)) -> in

0.8800599700149925 -> out

Model complexity
- Decision boundaries are thresholds for determining what label a model assign to an observation
- incases where as k increases and the decision boundaries is less affected by individual observations, its said to be reflecting a simpler model
- Simpler models are less able to detect relationships in the dataset, which is known as underfitting
i.e 
* Larger k = less complex model = can cause underfitting
* Smaller k = more complex model = can lead to overfitting
- Complex models can be sensitive to noise in the training data, rather than reflecting general trends, and this is known as overfitting,

Model complexity and over / underfitting
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)
for neighbor in neighbors:
    knn = KNeighborsClassifier(n_neighbors = neighbor)
    knn.fit(X_train, y_train)
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(_test, y_test)

Plotting our results
plt.figure(figsize=(8, 6))
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.legend()
plt.xlable('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
'''

# Import the module

X = churn_df.drop("churn", axis=1).values
y = churn_df["churn"].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))


# Create neighbors
neighbors = np.arange(1, 13)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

    # Set up a KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=neighbor)

    #  Fit the model
    knn.fit(X_train, y_train)

    # Compute accuracy
    train_accuracies[neighbor] = knn.score(X_train, y_train)
    test_accuracies[neighbor] = knn.score(X_test, y_test)
print(neighbors, '\n', train_accuracies, '\n', test_accuracies)


# Add a title
plt.title("KNN: Varying Number of Neighbors")

#  Plot training accuracies
plt.plot(neighbors, train_accuracies.values(), label="Training Accuracy")

# Plot test accuracies
plt.plot(neighbors, test_accuracies.values(), label="Testing Accuracy")

plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

# Display the plot
plt.show()


''' Regression '''

'''
Introduction to regression
- In regression tasks, the target variable typically has continuous values, such as a country's GDP, or the price of a house.

Predicting blood glucose levels
import pandas as pd
diabetest_df = pd.read_csv('diabetes.csv')
print(diabetes_df.head()) -> in

    pregnancies glucose triceps insulin   bmi   age diabetes
0             6     148      35       0  33.6    50        1
1             1      85      29       0  26.6    31        0
2             8     183       0       0  23.3    32        1
3             1      89      23      94  28.1    21        0
4             0     137      35     168  43.1    33        1 -> out

Creating feature and trget arrays
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
print(type(X), type(y)) -> in

<class 'numpy.ndarray'> <class 'numpy.ndarray'> -> out

Making predictions from a single feature
X_bmi = X[:, 3]
print(y.shape, X_bmi.shape) -> in

(752, ) (752, ) -> out

X_bmi = X_bmi.reshape(-1, 1)
print(X_bmi.shape) -> in

(752, 1) -> out

Plotting glucose vs body mass index
import matplotlib.pyplot as plt
plt.scatter(X_bmi, y)
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('Body Mass Index')
plt.show()

Fitting a regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_bmi, y)
predictions = reg.predict(X_bmi)

plt.scatter(X_bmi, y)
plt.plot(X_bmi, predictions)
plt.ylabel('Blood Glucose (mg/dl)')
plt.xlabel('Body Mass Index')
plt.show()
'''


# Create X from the radio column's values
X = sales_df['radio'].values

# Create y from the sales column's values
y = sales_df['sales'].values

# Reshape X
X = X.reshape(-1, 1)

# Check the shape of the features and targets
print(X.shape, y.shape)


# Import LinearRegression

# Create the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

print(predictions[:5])


# Import matplotlib.pyplot

# Create scatter plot
plt.scatter(X, y, color="blue")

# Create line plot
plt.plot(X, predictions, color="red")
plt.xlabel("Radio Expenditure ($)")
plt.ylabel("Sales ($)")

# Display the plot
plt.show()


'''
The basics of linear regression
Regression mechanics
-                 y = ax  + b
* Simple linear regression uses one feature
y = target
x = single feature
a, b = parameters / coefficienst of the model - slope, intercept

- How do we choose a and b?
* Define an error function for any given line
* Choose the line that minimizes the error functions

- Error function = loss function = cost function

The loss function
- The line between each observation and the vertical distance (Residual) between it and the line (slope) 

Ordinary Least Squares
- Residual sum of squares (RSS) can be calculated by squaring the residuals  and adding all the squared residuals
- Ordinary Least Squares (OLS) are used to minimize the RSS

Linear rgression in higher dimensions
y = a1x1 + a2x2+ b

- To fit a linear regression model here:
* Need to specify 3 variables a1, a2, b
- In higher dimensions:
* Multiple regression is the process of adding more features
* When fitting, you must specify coefficients for each feature and the variable b
y = a1x1 + a2x2 + a3x3 + . . . + a(n)x(n) + b

- scikit-learn works exactly the same:
* Pass two arrays: feature and target

Linear regression using all features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
reg_all = LinearRegressio()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

NB* Linear regression in scikit-learn performs OLS under the hood

R-squared
- R^2: It is the default metric for linear regression and it quantifies the amount of variance in the target variable that is explained by the features
* Values range from 0 to 1

R-squared in scikit-learn
reg_all.score(X_test, y_test) -> in

0.356302876407827 -> out

Mean squared error and root mean squared error
- MSE It is gotten by taking the mean of the residual sum of squares
* MSE is measured in units of our target variable, squared

- RMSE measures in the same units at the target variable

RMSE in scikit-learn
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred, squared=False) -> in

24.028109426907236 -> out
'''

# Create X and y arrays
X = sales_df.drop("sales", axis=1).values
y = sales_df["sales"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))


# Import mean_squared_error

# Compute R-squared
r_squared = reg.score(X_test, y_test)

# Compute RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the metrics
print("R^2: {}".format(r_squared))
print("RMSE: {}".format(rmse))


'''
Cross-validation
Cross-validation motivation
- model performace is dependent on the way we split up the data
- It is not representative of the model's ability to generalize to unseen data
- Solution:  cross-validation!

Cross-validation basics
Split 1     Fold 1 (Test set)   Fold 2  Fold 3  Fold 4  Fold 5   Metric 1
Split 2     Fold 1  Fold 2 (Test set)   Fold 3  Fold 4  Fold 5   Metric 2
Split 3     Fold 1  Fold 2  Fold 3 (Test set)   Fold 4  Fold 5   Metric 3
Split 4     Fold 1  Fold 2  Fold 3  Fold 4 (Test set)   Fold 5   Metric 4
Split 5     Fold 1  Fold 2  Fold 3  Fold 4  Fold 5 (Test set)    Metric 5

Cross-validation and model performance
- 5 folds = 5-fold CV
- 10 folds = 10-fold CV
- k folds = k-fold CV
- More folds = More computationally expensive because we are fitting and predicting more times

Cross-validation in scikit-learn
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits=6, shuffle=True, random_state=42)

* n_splits argument has a default of 5
* shuffle = True, shuffles the dataset before splitting into folds

reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=kf)

* NB: The score reported is R squared (R^2), since its the default score for linear regression

Evaluating cross-validation performance
print(cv_results) -> in

[0.70262578, 0.7659624, 0.75188205, 0.76914482, 0.72551151, 0.73608277] -> out

print(np.mean(cv_results), np.std(cv_results)) -> in

0.7418682216666667 0.023330243960652888 -> out

print(np.quantile(cv_results, [0.025, 0.975])) -> in

array([0.7054865, 0.76874702]) -> out
'''

# Import the necessary modules

#  Create a KFold object
kf = KFold(n_splits=6, shuffle=True, random_state=5)

reg = LinearRegression()

# Compute 6-fold cross-validation scores
cv_scores = cross_val_score(reg, X, y, cv=kf)

# Print scores
print(cv_scores)


# Print the mean
print(np.mean(cv_results))

# Print the standard deviation
print(np.std(cv_results))

# Print the 95% confidence interval
print(np.quantile(cv_results, [0.025, 0.975]))


'''
Regularized regression
Why regularize?
- Recall: Linear regression minimizes a loss function 
- It chooses a coefficient, a (slope), for each feature variable, plus b (intercept)
- Large coefficients can lead to overfitting
- Regularization: Penalizes large coefficients

Ridge regression
- Loss function = OLS (Ordiary Least Squares) loss function + Squared value of each coefficient, multiplied by a constant, alpha 
- Ridge penalizes large positive or negative coefficients values
- alpha : Its the parameter we need to choose in order to fit and predict.
* Picking alpha for ridge is similar to picking k in KNN
* alpha in ridge is known as a Hyperparameter, which is a variable used for selecting a model's parameters
- alpha controls model complexity
* when alpha = zero (0) = OLS (Can lead to overfitting)
* Very high alpha: Can lead to underfitting

Ridge regression in scikit-learn
from sklearn.linear_model import Ridge
scores = []
for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    ridge = Ridge(alpha = alpha)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    scores.append(ridge.score(X_test, y_test))
print(scores) -> in

[0.2828466623222221, 0.28320633574804777, 0.2853000732200006, 0.26423984812668133, 0.19292424694100963] -> out

Lasso regression 
- Loss function = OLS (Ordiary Least Squares) loss function + absolute value of each coefficient, multiplied by a constant, alpha 

Lasso regression in scikit-learn
from sklearn.linear_model import Lasso
scores = []
for alpha in [0.01, 1.0, 10.0, 20.0, 50.0]:
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    scores.append(lasso.score(X_test, y_test))
print(scores) -> in

[0.99991649071123, 0.99961700284223, 0.93882227671069, 0.74855318676232, -0.05741034640016] -> out

Lasso regression for feature selection
- Lasso can be used to select important features of a dataset
* because it shrinks the coefficients of less important features to zero
* features whose coefficients are not shrun to zero are selected by lasso algorithm

Lasso for feature selection in scikit-learn
from sklearn.linear_model import Lasso
X = diabetes_df.drop('glucose', axis=1).values
y = diabetes_df['glucose'].values
names = diabetes_df.drop('glucose', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
plt.bar(names, lasso_coef)
plt.xticks(rotation=45)
plt.show()
'''

# Import Ridge
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
ridge_scores = []
for alpha in alphas:

    # Create a Ridge regression model
    ridge = Ridge(alpha=alpha)

    # Fit the data
    ridge.fit(X_train, y_train)

    # Obtain R-squared
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)
print(ridge_scores)


# Import Lasso

# Instantiate a lasso regression model
lasso = Lasso(alpha=0.3)

# Fit the model to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)
plt.bar(sales_columns, lasso_coef)
plt.xticks(rotation=45)
plt.show()


''' Fine-Tuning Your Model '''

'''
How good is your model?
Classification metrics
- Measuring model performance with accuracy:
* Fraction of correctly classified samples/samples
* Its not always a useful metric

Class imbalance
e.g Classification for predicting fraudulent bank transactions
* 99% of transactions are legitimate; 1% are fraudulent
- Could build a classifier that predicts NONE of the transactions are fraudulent
* 99%accurate!
* But terrible at actually predicting fraudulent transactions
* Fails at its original purpose
- Class imbalance: it has uneven frequency of classes
- It requires a different approach to assess ing the model's performance

Confusion matrix for assessing classification performance
- Confusion matrix

                    Predicted: Legitimate   Predicted: Fraudulent
Actual: Legitimate          True Negative          False Positive 
Actual: Fraudulent         False Negative           True Positive

True Positive - tp
True Negative - tn
False Positive - fp
False Negative - fn

- Confusion matrix can retrieve accuracy
* Accuracy: tp + tn / tp + tn + fp +fn

- Confusion matric can calculate precision (aka Positive predictive value)
* Precision: tp / tp + fp

* High precision = lower false positive rate
* High precision: i.e not many legitimate transactions are predicted to be fraudulent

- Confusion matric can calculate sensitivity (aka Recall)
* Recall: tp / tp + fn

* High recall = lower false negative rate
* High recall: i.e Predicted most fraudulent transactions correctly

- Confusion matric can calculate F1 score (Harmonic mean of precision and recall)
* F1 score: 2 x ( (precision x recall) / (precision + recall) )
* This metric gives equal weight to precision and recall, therefore it factors in both the number of errors made by the model and the type of errors
* F1 score favours models with similar precision and recall, and is a useful metric if we are seeking a model which perform reasonably well across both metrics

Confusion matrix in scikit-learn 
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors=7)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn.fit(X_train, y_train)

print(confusion_matrix(y_test, y_pred)) -> in

[   [1106 11]
    [ 183 34]   ] -> out

Classificaation report in scikit-learn
print(classification_report(y_test, y_pred)) -> in

                precision recall f1-score support
0                    0.86   0.99     0.92    1117
1                    0.76   0.16     0.26     217

accuracy                             0.85    1334
macro avg            0.81   0.57     0.59    1334
weighted avg         0.84   0.85     0.81    1334 -> out
'''

#  Import confusion matrix

knn = KNeighborsClassifier(n_neighbors=6)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


'''
Logistic regression and the ROC curve
Logistic regression for binary classification
- Logistic regression is used for classification problems
- Logistic regression calculates the probability, p, that an observation belongs to a binary class
- If the probability, p > 0.5:
* The data is labeled 1
- If the probability, p < 0.5:
* The data is labeled 0
- NB: logistic regression produces a linar decision boundary

Logistic regression in scikit-learn
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

Predicting probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]
print(y_pred_probs[0]) -> in

[0.08961376] -> out

Probability thresholds
- By default, logistic regression treshold = 0.5
- threshold is not specific to logistic regression
* KNN classifiers also have thresholds
- What happens if we ary the threshold?
* we can use a Receiver Operating Characteristic (ROC curve), to visualize how different thresholds affect true positive and false positive rates
- when the threshold = zero (0), the model predicts 1 for all observations(i.e it will correctly predict all positive values, and incorrectly predict all negative values) 
- when the threshold = one (1), the model predicts 0 for all data(i.e both true and positive rates are zero (0))

Plotting the ROC
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

fpr - false positive rate
tpr - true positive rate 

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()

ROC AUC (Area Under Curve) in scikit-learn
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, y_pred_probs)) -> in

0.6700964152663693 -> out
'''

#  Import LogisticRegression

# Instantiate the model
logreg = LogisticRegression()

# Fit the model
logreg.fit(X_train, y_train)

# Predict probabilities
y_pred_probs = logreg.predict_proba(X_test)[:, 1]

print(y_pred_probs[:10])


# Import roc_curve

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

plt.plot([0, 1], [0, 1], 'k--')

# Plot tpr against fpr
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Diabetes Prediction')
plt.show()


# Import roc_auc_score

# Calculate roc_auc_score
print(roc_auc_score(y_test, y_pred_probs))

# Calculate the confusion matrix
print(confusion_matrix(y_test, y_pred))

# Calculate the classification report
print(classification_report(y_test, y_pred))


'''
Hyperparameter tuning
- Ridge / Lasso regression: Choosing alpha
- KNN: Choosing n_neighbors
- Hyperparameters: Parameters we specify before fitting the model
* like alpha and n_neighbors

Choosing the correct hyperparameters
1. Try lots of different hyperparameter values
2. Fit all of them separately
3. See how well they perform
4. Choose the best performing values

- This is called hyperparameter tuning
- It is essential to use cross-validation to avoid overfitting them to the test set
- We can still split the data, but perform cross-validation on the training set
* we withhold the test set and use if for evaluating the tuned model (final evaluation)

Grid search cross-validation
e.g
                        metric
                euclidean manhattan
n_neighbors 2      0.8634    0.8646
            5      0.8748    0.8714
            8      0.8704    0.8688
            11     0.8716    0.8692

GridSearchCV in scikit-learn
from sklearn.model_selection import GridSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_) -> in

{'alpha': 0.0001, 'solver': 'sag'}
0.7529912278705785 -> out

Limitations and an alternative approach
- GridSearch no of fit is equal to the no of hyperparameters multiplied by the number of values multiplied by the number of folds, therefore, it doesnt scale well
- 3-fold cross-validation, 1 hyperparameter, 10 total values = 30 fits
- 10 fold cross-validation, 3 hyperparameter, 30 total values = 900 fits

RandomizedSearchCV
- this picks random hyperparameter values rather than exhaustively searching through all options.

from sklearn.model_selection import RandomizedSearchCV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {'alpha': np.arange(0.0001, 1, 10), 'solver': ['sag', 'lsqr']}
ridge = Ridge()
ridge_cv = RandomizedSearchCV(ridge, param_grid, cv=kf, n_iter=2)

- n_iter argument, determines the number of hyperparameter values tested

ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_) -> in

{'solver': 'sag', 'alpha': 0.0001}
0.7529912278705785 -> out

Evaluating on the test set
test_score = ridge_cv.score(X_test, y_test)
print(test_score) -> in

0.7564731534089224 -> out
'''

# Import GridSearchCV

#  Set up the parameter grid
param_grid = {"alpha": np.linspace(0.00001, 1, 20)}

# Instantiate lasso_cv
lasso_cv = GridSearchCV(lasso, param_grid, cv=kf)

# Fit to the training data
lasso_cv.fit(X_train, y_train)
print("Tuned lasso paramaters: {}".format(lasso_cv.best_params_))
print("Tuned lasso score: {}".format(lasso_cv.best_score_))


#  Create the parameter space
params = {"penalty": ["l1", "l2"], "tol": np.linspace(0.0001, 1.0, 50), "C": np.linspace(
    0.1, 1.0, 50), "class_weight": ["balanced", {0: 0.8, 1: 0.2}]}

# Instantiate the RandomizedSearchCV object
logreg_cv = RandomizedSearchCV(logreg, params, cv=kf)

# Fit the data to the model
logreg_cv.fit(X_train, y_train)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Best Accuracy Score: {}".format(
    logreg_cv.best_score_))


''' Preprocessing and Pipelines '''

'''
Preprocessing data
scikit-learn requirements
-Numeric data
- No missing values

- With real-world data:
* This is rarely the case
* we will often need to preprocess our data first

Dealing with categorical features
- scikit-learn will not accept categorical features by default
- need to convert categorical features into numeric values
- Conver to binary features called dummy variables
* 0: Observation was NOT that category
* 1: Observation was that category

Dummy variables
genre       Alternative Anime Blues Classical Country Electronic Hip-Hop Jazz Rap
Alternative           1     0     0         0       0          0       0    0   0
Anime                 0     1     0         0       0          0       0    0   0
Blues                 0     0     1         0       0          0       0    0   0
Classical             0     0     0         1       0          0       0    0   0
Country               0     0     0         0       1          0       0    0   0
Electronic            0     0     0         0       0          1       0    0   0
Hip-Hop               0     0     0         0       0          0       1    0   0
Jazz                  0     0     0         0       0          0       0    1   0
Rap                   0     0     0         0       0          0       0    0   1
Rock                  0     0     0         0       0          0       0    0   0

Dealing with categorical features in Python
- scikit-learn: OneHotEncoder()
- pandas: get_dummies()

Music dataset
- popularity: Target variable
- genre: Categorical feature

print(music.info()) -> in

    popularity acousticness danceability . . .      tempo valence   genre
0         41.0       0.6440        0.823 . . . 102.619000   0.649   Jazz
1         62.0       0.0855        0.686 . . . 173.915000   0.636   Rap
2         42.0       0.2390        0.669 . . . 145.061000   0.494   Electronic
3         64.0       0.0125        0.522 . . . 120.406497   0.595   Rock
4         60.0       0.1210        0.780 . . .  96.056000   0.312   Rap 

Encoding dummy variables
import pandas as pd
music_df = pd.read_csv('music.csv')
music_dummies = pd.get_dummies(music_df['genre'], drop_first=True)
print(music_dummies.head()) -> in

    Anime Blues Classical Country Electronic Hip-Hop Jazz Rap Rock
0       0     0         0       0          0       0    1   0    0
1       0     0         0       0          0       0    0   1    0
2       0     0         0       0          1       0    0   0    0
3       0     0         0       0          0       0    0   0    1
4       0     0         0       0          0       0    0   1    0 -> out

music_dummies = pd.concat([music_df, music_dummies], axis=1)
music_dummies = music_dummies.drop('genre', axis=1)

- If the DataFrame only has one categorical feature
music_dummies = pd.get_dummies(music_df, drop_first=True)
print(music_dummies.columns) -> in

Index(['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudeness', 'speechiness', 'tempo', 'valence', 'genre_Anime', 'genre_Blues', 'genre_Classical', 'genre_Country', 'genre_Electronic', 'genre_Hip-Hop', 'genre_Jazz', 'genre_Rap', 'genre_Rock'], dtype='object') -> out

Linear regression with dummy variables
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
X = music_dummies.drop('popularity', axis=1).values
y= music_dummies['popularity'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_szie=0.2, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')

- scoring='neg_mean_squared_error', and will return the negative MSE because scikit-learn CV metrics presume a higher score is better, so MSE is changed to negative to counteract this

print(np.sqrt(-linreg_cv)) -> in

[8.15792932, 8.63117538, 7.52275279, 8.6205778, 7.91329988] -> out
'''

# Create music_dummies
music_dummies = pd.get_dummies(music_df, drop_first=True)

# Print the new DataFrame's shape
print("Shape of music_dummies: {}".format(music_dummies.shape))


# Create X and y
X = music_dummies.drop('popularity', axis=1).values
y = music_dummies['popularity'].values

#  Instantiate a ridge model
ridge = Ridge(alpha=0.2)

#  Perform cross-validation
scores = cross_val_score(ridge, X, y, cv=kf, scoring="neg_mean_squared_error")

#  Calculate RMSE
rmse = np.sqrt(-scores)
print("Average RMSE: {}".format(np.mean(rmse)))
print("Standard Deviation of the target array: {}".format(np.std(y)))


'''
Handling missing data
Missing data
- This is when theres no value for a feature in a particular row
- This can occur because:
* There may have been no observation
* The data might be corrupt

Music dataset
print(music_df.isna().sum().sort_values()) -> in

genre               8
popularity         31
loudness           44
liveness           46
tempo              46
speechiness        59
duration_ms        91
instrumentalness   91
danceability      143
valence           143
acousticness      200
energy            200
dtype: int64 -> out

Dropping missing data
- Drop missing values in columns accounting for less than 5% of our data

music_df = music_df.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
print(music_df.isna().sum().sort_values()) -> in

popularity          0
liveness            0
loudness            0
tempo               0
genre               0
duration_ms        29
speechiness        29
instrumentalness   53
danceability      127
valence           127
acousticness      178
energy            178
dtype: int64 -> out

Inputing values
- Imputation - It is the use of subject-matter expertise to replace missing data with educated guesses
- Common to use the mean
- Can also use the median or another value
- For categorical values, we typically use the most frequent value - the mode
- NB*: we must split our data before imputing, to avoid leaking test set information to our model (data leakage)

Imputation with scikit-learn
from sklearn.impute import SimpleImputer
X_cat = music_df['genre'].values.reshape(-1, 1)
X_num = music_df.drop(['genre', 'popularity'], axis=1).values
y = music['popularity'].values
X_train_cat, X_test_cat, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=12)
X_train_num, X_test_num, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=12)
imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_cat)
X_test_cat = imp_cat.transform(X_test_cat)

- by default, SimpleImputer expects np.nan to represent missing values

imp_num = SimpleImputer()
X_train_num = imp_num.fit_transform(X_train_num)
X_test_num = imp_num.transform(X_test_num)

- by default, SimpleImputer fills nvalues with the mean

X_train = np.append(X_train_num, X_train_cat, axis=1) 
X_test = np.append(X_test_num, X_test_cat, axis=1) 

- Due to their ability to transform our data, imputers are knowns as transformers

Imputing within a pipeline
- A pipeline is an object used to run a series of transformations and build a model in a single workflow

from sklearn.pipeline import Pipeline
music_df = music_df.dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])
music_df['genre'] = np.where(music_df['genre'] == 'Rock', 1, 0)
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values

steps = [('imputation', SimpleImputer()), ('logistic_regression', LogisticRegression())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test) -> in

0.7593582887700535 -> out

- NB*: in a pipeline, each step but the last must be a transformer.
'''

# Print missing values for each column
print(music_df.isna().sum().sort_values())

# Remove values where less than 5% are missing
music_df = music_df.dropna(
    subset=["genre", "popularity", "loudness", "liveness", "tempo"])

# Convert genre to a binary feature
music_df["genre"] = np.where(music_df["genre"] == "Rock", 1, 0)

print(music_df.isna().sum().sort_values())
print("Shape of the `music_df`: {}".format(music_df.shape))


# Import modules

# Instantiate an imputer
imputer = SimpleImputer()

# Instantiate a knn model
knn = KNeighborsClassifier(n_neighbors=3)

# Build steps for the pipeline
steps = [("imputer", imputer), ("knn", knn)]


# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Print the confusion matrix
print(confusion_matrix(y_test, y_pred))


'''
Centering and scaling
print(music_df[['duration_ms', 'loudness', 'speechiness']].describe()) -> in

        duration_ms     loudness speechiness
count  1.000000e+03  1000.000000 1000.000000
mean   2.176493e+05    -8.284354    0.078642
std    1.137703e+05     5.065447    0.088291
min   -1.000000e+00   -38.718000    0.023400
25%    1.831070e+05    -9.658500    0.033700
50%    2.176493e+05    -7.033500    0.045000
75%    2.564468e+05    -5.034000    0.078642
max    1.617333e+06    -0.883000    0.710000 -> out

- duration_ms ranges from 0 - 1.617333e+06
- speechiness contains only decimal places
- loudness only has negative values

Why scale our data?
- Many models use some form of distance to inform them
- Features on larger scales can disproportionately influence the model
* Example: KNN uses distance explicitly when making predictions
- We want features to be on a similar scale
- It can be achieved by Normalizing or standardizing our data (scaling and centering)

How to scale our data
- Subtract the mean and divide by variance
* All features are centered around zero and have a variance of one
* This is called standardization
- Can also subtract the minimum and divide by the range
* Minimum zero and maximum one
- Can also normalize so the data ranges from -1 to +1
- See scikit-learn docs for further details

Scalining in scikit-learn
from sklearn.preprocessing import StandardScaler
X = music_df.drop('genre', axis=1).values
y = music_df['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(np.mean(X), np.std(X))
print(np.mean(X_train_scaled), np.std(X_train_scaled)) -> in

19801.42536120538, 71343.52910125865
2.260817795600319e-17 1.0 -> out

Scaling in a pipeline
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = knn_scaled.predict(X_test)
print(knn_scaled.score(X_test, y_test)) -> in

0.81 -> out

Comparing performance using unscaled data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_unscaled = KNeighborsClassified(n_neighbors=6).fit(X_train, y_train)
print(knn_unscaled.score(X_test, y_test)) -> in

0.53 -> out

CV and scaling in a pipeline
fromsklearn.model_selection import GridSearchCV
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
parameters = {'knn__n_neighbors': np.arange(1, 50)}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

Checking model parameters
print(cv.best_score_) -> in

0.8199999999999999 -> out

print(cv.best_params_) -> in

{'knn__n_neighbors': 12} -> out
'''

# Import StandardScaler

# Create pipeline steps
steps = [("scaler", StandardScaler()), ("lasso", Lasso(alpha=0.5))]

# Instantiate the pipeline
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

#  Calculate and print R-squared
print(pipeline.score(X_test, y_test))


# Build the steps
steps = [("scaler", StandardScaler()), ("logreg", LogisticRegression())]
pipeline = Pipeline(steps)

# Create the parameter space
parameters = {"logreg__C": np.linspace(0.001, 1.0, 20)}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=21)

# Instantiate the grid search object
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training data
cv.fit(X_train, y_train)
print(cv.best_score_, "\n", cv.best_params_)


'''
Evaluating multiple models
Different models for different problems
some guiding principles
- Size of the dataset
* Fewer features = simpler model, faster training time
* Some models such as Artificiaal Neural Networks, require a lot of data to perform well
- Interpretability
* Some models are easier to explain, which can be important for stakeholders
* Example: Linear regression has high interpretability, as we can calculate and interpret the model coefficients
- Flexibility
* They may improve accuracy, by making fewer assumptions about the data
* Example: KNN is a more flexible model, doesn't assume any linear relationships between the features and the target

It's all in the metrics
- Regression model performance:
* RMSE
* R-squared
- Classification model performance:
* Accuracy
* Confusion matrix
* Precision, recall, F1-score
* ROC AUC
- Train several models and evaluate performance out of the box

A note on scaling
- Models affected by scaling:
* KNN
* Linear Regression (plus Ridge, Lasso)
* Logistic Regression
* Artificial Neural Network
- Best to scale our data before evaluating models out of the box

Evaluating classification models
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X = music.drop('genre', axis=1).values
y = music['genre'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {'Logistic Regression': LogisticRegression(), 'KNN': KNeighborsClassifier(), 'Decision Tree': DecisionTreeClassifier()}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()

Test set performance
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print('{} Test Set Accuracy: {}'.format(name, test_score)) -> in

Logistic Regression Test Set Accuracy: 0.844
KNN Test Set Accuracy: 0.82
Decision Tree Test Set Accuracy: 0.832 -> out
'''

models = {"Linear Regression": LinearRegression(), "Ridge": Ridge(
    alpha=0.1), "Lasso": Lasso(alpha=0.1)}
results = []

# Loop through the models' values
for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)

    # Append the results
    results.append(cv_scores)

# Create a box plot of the results
plt.boxplot(results, labels=models.keys())
plt.show()


# Import mean_squared_error

for name, model in models.items():

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate the test_rmse
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("{} Test Set RMSE: {}".format(name, test_rmse))


#  Create models dictionary
models = {"Logistic Regression": LogisticRegression(), "KNN": KNeighborsClassifier(
), "Decision Tree Classifier": DecisionTreeClassifier()}
results = []

# Loop through the models' values
for model in models.values():

    #  Instantiate a KFold object
    kf = KFold(n_splits=6, random_state=12, shuffle=True)

    # Perform cross-validation
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)
plt.boxplot(results, labels=models.keys())
plt.show()


# Create steps
steps = [("imp_mean", SimpleImputer()), ("scaler", StandardScaler()),
         ("logreg", LogisticRegression())]

# Set up pipeline
pipeline = Pipeline(steps)
params = {"logreg__solver": ["newton-cg", "saga",
                             "lbfgs"], "logreg__C": np.linspace(0.001, 1.0, 10)}

# Create the GridSearchCV object
tuning = GridSearchCV(pipeline, param_grid=params)
tuning.fit(X_train, y_train)
y_pred = tuning.predict(X_test)

# Compute and print performance
print("Tuned Logistic Regression Parameters: {}, Accuracy: {}".format(
    tuning.best_params_, tuning.score(X_test, y_test)))
