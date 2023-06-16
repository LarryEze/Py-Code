# import necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_pandas import DataFrameMapper
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


''' Classification with XGBoost '''

'''
Welcome to the course!
Supervised learning
- Relies on labeled data
- Have some understanding of past behaviour

- e.g Does a specific image contain a person's face?
* Training data: vectors of pixel values
* Labels: 1 or 0

Classification problems
- Outcome can be binary or multi-class
- e.g Binary classification
* Will a person purchase the insurance package given some quote?
-The AUC (Area under the ROC curve) is the most versatile and common evaluation metric used to judge the quality of a binary classification model.
* Larger area under the ROC curve = better model

- e.g Multi-class classification
* Classifying the species of a given bird
- The Accuracy score and overall Confusion matrix are the common metric to evaluate the quality of a multi-class model.
* Confusion matrix

                    Predicted: Spam Email   Predicted Real Email
ACtual: Spam Email          True Positive         False Negative
Actual:Real Email          False Positive          True Negative

* Accuracy = ( tp + tn ) / ( tp + tn + fp + fn )

Other supervised learning considerations
- Features can be either numeric or categorical
- Numeric features should be scaled (Z-scored)
- Categorical features should be encoded (one-hot)

Other supervised learning problems include
- Ranking problems: It involves predicting an ordering on a set of choices
- e.g Google search suggestions

- Recommendation problems: It involves recommending an item or  set of items to a user based on his/her consumption history and profile
- e.g Netflix
'''


'''
Introducing XGBoost
What is XGBoost?
- It is an optimized gradient boosting machine learning library
- It was originally written in C++
- It has APIs in several languages:
* Python
* R
* Scala
* Julia
* Java

What makes XGBoost so popular?
- Speed and Performance
- The core XGBoost algorithm is parallelizable i.e it can harness all of the processing power of modern multi-core computers.
- It is also parallelizable onto GPU's and across networks of computers, making it feasible to train models on very large datasets on the order of hundreds of millions of training examples.
- It consistently outperforms almost all other single-algorithm methods in machine learning competitions
- It has been shown to achieve state-of-the-art performance in many machine learning tasks

Using XGBoost: a quick example
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
class_data = pd.read_csv('classification_data.csv')

X, y = class_data[:, :-1], class_data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)
accuracy = float(np.sum(pred==y_test)) / y_test.shape[0]

print('accuracy: %f' % (accuracy)) -> in

accuracy: 0.78333 -> out
'''

# Import xgboost

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic',
                          n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds == y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


'''
What is a decision tree?
Visualizing a decision tree
        Road tested?
    no /            \ yes
(DON'T BUY)          Mileage
                high /     \ low
                    Age    (BUY)
                old / \ recent
            DON'T BUY   BUY

Decision trees as base learners
- Base learner - Its is any individual learning algorithm in an ensemble algorithm
- It is composed of a series of binary decisions (yes / no OR true / false questions)
- Predictions happen at the 'leaves' of the tree

Decision trees and CART
- They are constructed iteratively (i.e one binary decision at a time) until some stopping criterion is met
- Individual decision trees in general are low-bias, high-variance learning models 
- They are very good at learning relationships within any data you train them on, but they tend to overfits the data you use to train them on and usually generalize to new data poorly.

CART: Classification and Regression Trees
- XGBoost uses CART as its decision tree
- CART tree contain a real-valued score in each leaf, regardless of whether they are used for classification or regression.
- The real-valued scores can be thresholded to convert into categories for classification problems if necessary.
'''

breast_cancer = datasets.load_breast_cancer()

X = breast_cancer.data

y = breast_cancer.target
# Import the necessary modules

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4 == y_test))/y_test.shape[0]
print("accuracy:", accuracy)


'''
What is Boosting?
Boosting overview
- It is not a specific machine learning algorithm, but a concept that can be applied to a set of machine learning models ('Meta-algorithm').
- It is an ensemble meta-algorithm, primarily used to reduce any given single learner's variance and to convert many weak learners into an arbitrarily strong learner.

Weak learners and strong learners
- Weak learner: It is any machine learning algorithm that is just slightly better than chance
* e.g A decision tree whose predictions are slightly better than 50%
- Boosting converts a collection of weak learners into a strong learner
- Strong learner: it is any algorithm that can be tuned to achieve arbitrarily good performance for some some supervised learning problem.

How boosting is accomplished
- By iteratively learning a set of weak models on subsets of the data
- Weighing each of their predictions according to each weak learner's performance.
- Combine all of the weak learner's predictions multiplied by their weights to obtain a  single final weighted prediction that is much better than any of the individual predictions themselves.

Model evaluation through cross-validation
- Cross-validation: It is a robust method for estimating the expected performance of a machine learning model on unseen data.
- By generating many non-overlapping train/test splits on the training data
- Reporting the average test set performance across all data splits.

Cross-validation in XGBoost example
import xgboost as xgb
import pandas as pd
churn_data = pd.read_csv('classification_data.csv')
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:, :-1], label=churn_data.month_5_still_here)
params = {'objective'='binary:logistic', 'max_depth':4}
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=4, num_boost_round=10, metrics='error', as_pandas=True)
print('Accuracy: %f' %((1-cv_results['test-error-mean']).iloc[-1])) -> in

Accuracy: 0.88315 -> out

- xgb.DMatrix converts the dataset into an optimized data structure for high performance and efficiency
- nfold = number of cross-validation folds
- num_boost_round = number of trees to build
- metrics = metric to compute
'''

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:, :-1], churn_data.iloc[:, -1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:logistic", "max_depth": 3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3,
                    num_boost_round=5, metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))


# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, nfold=3,
                    num_boost_round=5, metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])


'''
When should I use XGBoost?
When to use XGBoost
- Theres a large number of training samples i.e greater than 1000 training samples and less than 100 features (The number of features < number of training samples)
- When theres a mixture of categorical and numeric features or just numeric features

When to NOT use XGBoost
- Image recognition
- Computer vision
- Natural language processing and understanding problems
- When the number of training samples is significantly smaller than the number of features
'''


'''
Regression with XGBoost
Regression basics
- Regression problems involves predicting continuous, or real, values
- e.g Predicting the height in centimeters a given person will be at 30, given some of their physical attributes at birth.

Common regression metrics
- Root mean squared error (RMSE) 
- Mean absolute error (MAE)

Computing RMSE
- It is computed by: 
* taking the difference between the actual and the predicted values for what you are trying to predict
* squaring those differences
* computing their mean
* Taking the square root of the value
- NB*: It allows to treat negative and positive differences equally, but tends to punish larger difference between predicted and actual values much more than smaller ones.

Actual Predicted Error Squared Error
10     20        -10   100
3      8         -5    25
6      1         5     25
- Total Squared Error: 150
- Mean Squared Error: 50
- Root Mean Squared Error: 7.07

Computing MAE
- It simply sums the absolute differences between  predicted and actual values across all of the samples we build our model on.
- NB*: It isn't affected by large differences as much as RMSE, it lacks some nice mathematical properties that make it much less frequently used as an evaluation metric.

Actual Predicted Error
10     20        -10
3      8         -5
6      1         5
- Total Absolute Error: 20
- Mean Absolute Error: 6.67

Common regression algorithms
- Linear regression
- Decision trees (can be used for both regression and classification tasks)
'''


''' Regression with XGBoost '''

'''
Objective (loss) functions and base learners
Objective functions and why we use Them
- An objective or loss function quantifies how far off our prediction is from the actual result for a given data point.
- It measures the difference between estimated and true values for some collection of data
- The goal of any machine learning model is to find the model that yields the minimum value of the loss function across all of the data points we pass in.

Common loss functions and XGBoost
- Loss function names in xgboost:
* reg:linear - used for regression problems
* reg:logistic - used for binary classification problems when you want just decision (category of the target), not probability
* binary:logistic - used for binary classification problems when you want the actual predicted probability of the positive class rather than just decision

Base learners and why we need Them
- XGBoost involves creating a meta-model that is composed of many individual models that combine to give a final prediction
- individual models = base learners
- The goal of XGBoost is to have base learners that when combined creates final prediction that is non-linear i.e slightly better than random guessing on certain subsets of training examples, and uniformly bad at the remainder
- Each base learner should be good at distinguishing or predicting different parts of the dataset
- 2 kinds of base learners: Tree and Linear

Trees as base learners example: scikit-learn API
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
X, y = boston_data[:, :-1], boston_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print('RMSE: %f' &(rmse)) -> in

RMSE: 129043.2314 -> out

Linear base learners example: learning API only
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

boston_data = pd.read_csv('boston_housing.csv')
X, y = boston_data[:, :-1], boston_data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
DM_train = xgb.DMatrix(data= X_train, label=y_train)
DM_test = xgb.DMatrix(data= X_test, label=y_test)
params = {'booster':'gblinear', 'objective':'reg:linear'}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print('RMSE: %f' &(rmse)) -> in

RMSE: 124326.24465
'''

ames_housing = pd.read_csv(
    'Extreme Gradient Boosting with XGBoost/ames_housing_trimmed_processed.csv')

X = ames_housing.drop('SalePrice', axis=1)

y = ames_housing['SalePrice']

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective="reg:linear", n_estimators=10, seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)


# Create the parameter dictionary: params
params = {"booster": "gblinear", "objective": "reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4,
                    num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4,
                    num_boost_round=5, metrics='mae', as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))


'''
Regularization and base learners in XGBoost
Regularization in XGBoost
- Regularization is a control on model complexity
- Loss functions in XGBoost are used to find models that are both accurate and as simple as possible.
- Regularization parameters in XGBoost:
* gamma - it is a parameter for tree base learners that controls the minimum loss reduction allowed for a split to occur (higher values leads to fewer splits)
* Alpha - aka L1 regularization. It penalizes leaf weights rather than on feature weights, as is the case in linear or logistic regression (larger values mean more regularization)
* Lambda - aka L2 regularization, it is a much smoother penalty than L1 and causes leaf weights to smoothly decrease, instead of enforcing strong sparsity constraints on the leaf weights as in L1.

L1 regularization in XGBoost example
import xgboost as xgb
import pandas as pd
boston_data = pd.read_csv('boston_data.csv')
params = {'objective':'reg:linear', 'max_depth':4}
l1_params = [1, 10, 100]
rmses_l1 = []
fpr reg in l1_params:
    params['alpha'] = reg
    cv_results = xgb.cv(dtrain=boston_dmatrix, params=params, nfold=4, num_boost_round=10, metrics='rmse', as_pandas=True, seed=123)
    rmses_l1.append(cv_results['test-rmse-mean'].tail(1).values[0])
print('Best rmse as a function of l1:')
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=['l1', 'rmse'])) -> in

Best rmse as a function of l1:
    l1          rmse
0    1  69572.517742
1   10  73721.967141
2  100  82312.312413 -> out

Base learners in XGBoost
- Linear Base Learner: 
* It is a sum of linear terms
* The boosted model is weighted sum of linear models (thus is itself linear) 
* Rarely used, as you can get identical performance from a regularized linear model.
- Tree Base learner:
* It uses decision trees as base models
* The boosted model is weighted sum of decision trees (nonlinear)
* Almost exclusively used in XGBoost

Creating DataFrames from multiple equal-length lists
- pd.DataFrame(list(zip(list1, list2)), columns= ['list1', 'list2']))
- zip creates a generator of parallel values:
* zip([1, 2, 3], ['a', 'b', 'c']) = [1, 'a']. [2, 'b'], [3, 'c']
* generators need to be completely instantiated before they can be used in DataFrame objects
- List() instantiates the full generator and passing that into the DataFrame converts the whole expression
'''

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective": "reg:linear", "max_depth": 3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg

    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                             num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir='LR')
plt.show()


# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective": "reg:linear", "max_depth": 4}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()


''' Fine-tuning your XGBoost model '''

'''
Why tune your model?
Untuned model example
import xgboost as xgb
import pandas as pd
import numpy as np
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
untuned_params = {'objective':'reg:linear'}
untuned_cv_results_rmse = xgb.cv(dtrain=housing_matrix, params=untuned_params, nfold=4, metrics='rmse', as_pandas=True, seed=123)
print('Untuned rmse: %f' %((untuned_cv_results_rmse['test-rmse-mean']).tail(1))) -> in

Untuned rmse: 34624.229980 -> out

Tuned model example
import xgboost as xgb
import pandas as pd
import numpy as np
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
tuned_params = {'objective':'reg:linear', 'colsample_bytree': 0.3, 'learning_rate': 0.1, 'max_depth': 5}
tuned_cv_results_rmse = xgb.cv(dtrain=housing_matrix, params=tuned_params, nfold=4, num_boost_round=200, metrics='rmse', as_pandas=True, seed=123)
print('Tuned rmse: %f' %((tuned_cv_results_rmse['test-rmse-mean']).tail(1))) -> in

Tuned rmse: 29812.683594 -> out
'''

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3,
                        num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)

    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses, columns=["num_boosting_rounds", "rmse"]))


# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective": "reg:linear", "max_depth": 4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, metrics='rmse',
                    early_stopping_rounds=10, num_boost_round=50, seed=123, as_pandas=True)

# Print cv_results
print(cv_results)


'''
Overview of XGBoost's hyperparameters
Common tree tunable parameters
- learning rate: It affects how quickly the model fits residual errors using additional base learners  (learning rate / eta).
* A low learning rate will require more boosting rounds to achieve the same reduction in residual error as an XGBoost model with a high learning rate.
- gamma: minimum loss reduction to create new tree split
- Lambda: L2 regularization on leaf weights
- alpha: L1 regularization on leaf weights
- max_depth: maximum depth per tree
- subsample: percentage of samples (training set) that can be used per tree (value between 0 and 1)
- colsample_bytree: percentage of features that can be used per tree (value between 0 and 1)

Linear tunable parameters
- Lambda: L2 regularization on leaf weights
- alpha: L1 regularization on leaf weights
- Lambda_bias: L2 regularization term on bias
- You can also tune the number of estimators used for both base model types! 
'''

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta
for curr_val in eta_vals:

    params["eta"] = curr_val

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, early_stopping_rounds=5,
                        num_boost_round=10, metrics='rmse', as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)),
      columns=["eta", "best_rmse"]))


# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:linear"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, early_stopping_rounds=5,
                        num_boost_round=10, metrics='rmse', as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),
      columns=["max_depth", "best_rmse"]))


# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary
params = {"objective": "reg:linear", "max_depth": 3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value
for curr_val in colsample_bytree_vals:

    params["colsample_bytree"] = curr_val

    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=10,
                        early_stopping_rounds=5, metrics="rmse", as_pandas=True, seed=123)

    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)),
      columns=["colsample_bytree", "best_rmse"]))


'''
Review of grid search and random search
Grid search: review
- It is a method of exhaustively searching through a collection of possible parameter values.
- Number of models = number of distinct values per hyperparameter multiplied across each hyperparameter
- Pick final model hyperparameter values that gives best cross-validated evaluation metric value

Grid search: example
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': [0.01, -.1, -.5, 0.9], 'n_estimators':[200], 'subsample': [0.3, 0.5, 0.9]}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print('Best parameters found: ', grid_mse.best_params_)
print('Lowest RMSE found: ', np.sqrt(np.abs(grid_mse.best_score_))) -> in

Best parameters found: {'learning_rate':0.1, 'n_estimators': 200, 'subsample': 0.5}
Lowest RMSE found: 28530.1829341 -> out

Random search: review
- Its different from GridSearch since it creates a (possibly infinite) range of hyperparameter values per hyperparameter that you would like to search over
- Set the number of iterations you would like for the random search to continue
- During each iteration, randomly draw a value in the range of specified values for each hyperparameter searched over and train / evaluate a model with those hyperparameters
- After you've reached the maximum number of iterations, select the hyperparameter configuration with the best evaluated score.

Random search: example
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
housing_data = pd.read_csv('ames_housing_trimmed_processed.csv')
X, y = housing_data[housing_data.columns.tolist()[:-1]], housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': np.arange(0.05, 1.05, 0.5), 'n_estimators':[200], 'subsample': np.arange(0.05, 1.05, 0.5)}
gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=25, scoring='neg_mean_squared_error', cv=4, verbose=1)
randomized_mse.fit(X, y)
print('Best parameters found: ', randomized_mse.best_params_)
print('Lowest RMSE found: ', np.sqrt(np.abs(randomized_mse.best_score_))) -> in

Best parameters found: {'learning_rate':0.20000000000000000001, 'n_estimators': 200, 'subsample': 0.600000000000000000009}
Lowest RMSE found: 28300.2374291 -> out
'''

# Create the parameter grid: gbm_param_grid
gbm_param_grid = {'colsample_bytree': [
    0.3, 0.7], 'n_estimators': [50], 'max_depth': [2, 5]}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid,
                        scoring='neg_mean_squared_error', cv=4, verbose=1)

# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


# Create the parameter grid: gbm_param_grid
gbm_param_grid = {'n_estimators': [25], 'max_depth': range(2, 12)}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
                                    n_iter=5, scoring='neg_mean_squared_error', cv=4, verbose=1)

# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))


'''
Limits of grid search and random search
Grid search and random search limitations
- Grid Search
* As the number of hyperparameters grows, the time it takes to complete a full grid search increases exponentially.
* Number of models you must  build with every additional new parameter grows very quickly.

- Random Search
* As the number of hyperparameters grows, the size of the hyperparameter space explodes as it did in the grid search case.
* Randomly jumping throughout the space looking for a 'best' result becomes a waiting game

- NB*: The search space size can be massive for Grid Search in certain cases, whereas for Random Search the number of hyperparameters has a significant effect on how long it takes to run.
'''


''' Using XGBoost in pipelines '''

'''
Review of pipelines using sklearn
- Pipelines in sklearn are objects that take a list of named 2-tuples (name, pipeline_step) as input.
- Tuples can contain any arbitrary scikit-learn compatible estimator or transformer object
- Pipeline implements fit / predict methods
- Pipelines can be used as input estimator objects into other grid /  randomized search and cross_val_score methods.

Scikit-learn pipeline example
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']

data = pd.read_csv('boston_housing.csv', names=names)

X, y = data.iloc[:, :-1], data.iloc[:, -1]
rf_pipeline = Pipeline[('st_scaler', StandardScaler()), ('rf_model', RandomForestRegressor())]

scores =  cross_val_score(rf_pipeline, X, y, scoring='neg_mean_squared_error', cv=10) 

- 'neg_mean_squared_error' is scikit-learn way of calculating the mean squared error in an API-compatible way.

final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print('Final RMSE:', final_avg_rmse) -> in

Final RMSE: 4.545306866529 -> out

Preprocessing I: LabelEncoder and OneHotEncoder
- LabelEncoder: It converts a categorical column of strings into integers that maps onto those strings.
- OneHotEncoder: It takes the column of integers that are treated as categorical values, and encodes them as dummy variables,
- Cannot be done within a pipeline

Preprocessing II: DictVectorizer
- It is traditionally used in text processing pipelines
- It converts lists of feature mappings into vectors.
- need to convert DataFrame into a list of dictionary entries
'''

df = pd.read_csv(
    'Extreme Gradient Boosting with XGBoost/ames_unprocessed_data.csv')
# Import LabelEncoder

# Fill missing values with 0
df.LotFrontage = df['LotFrontage'].fillna(0)

# Create a boolean mask for categorical columns
categorical_mask = (df.dtypes == object)

# Get list of categorical column names
categorical_columns = df.columns[categorical_mask].tolist()

# Print the head of the categorical columns
print(df[categorical_columns].head())

# Create LabelEncoder object: le
le = LabelEncoder()

# Apply LabelEncoder to categorical columns
df[categorical_columns] = df[categorical_columns].apply(
    lambda x: le.fit_transform(x))

# Print the head of the LabelEncoded categorical columns
print(df[categorical_columns].head())


# Import OneHotEncoder

# Create OneHotEncoder: ohe
ohe = OneHotEncoder(sparse=False)

# Apply OneHotEncoder to categorical columns - output is no longer a dataframe: df_encoded
df_encoded = ohe.fit_transform(df)

# Print first 5 rows of the resulting dataset - again, this will no longer be a pandas dataframe
print(df_encoded[:5, :])

# Print the shape of the original DataFrame
print(df.shape)

# Print the shape of the transformed array
print(df_encoded.shape)


# Import DictVectorizer

# Convert df into a dictionary: df_dict
df_dict = df.to_dict('records')

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df: df_encoded
df_encoded = dv.fit_transform(df_dict)

# Print the resulting first five rows
print(df_encoded[:5, :])

# Print the vocabulary
print(dv.vocabulary_)


# Import necessary modules

# Fill LotFrontage missing values with 0
X.LotFrontage = X['LotFrontage'].fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor())]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Fit the pipeline
xgb_pipeline.fit(X.to_dict("records"), y)


'''
Incorporating XGBoost into pipelines
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']

data = pd.read_csv('boston_housing.csv', names=names)
X, y = data.iloc[:, :-1], data.iloc[:, -1]
xgb_pipeline = Pipeline[('st_scaler', StandardScaler()), ('xgb_model', XGBRegressor())]
scores =  cross_val_score(xgb_pipeline, X, y, scoring='neg_mean_squared_error', cv=10) 
final_avg_rmse = np.mean(np.sqrt(np.abs(scores)))
print('Final XGB RMSE:', final_avg_rmse) -> in

Final XGB RMSE: 4.02719593323 -> out

Additional components introduced for pipelines
- sklearn_pandas: Its a separate library that attempts to bridge the gap between working with pandas and working with scikit-learn, as they don't always work seamlessly together.
* DataFrameMapper- It is a generic class that allows for easy conversion between scikit-learn aware objects, or pure numpy arrays, and the DataFrames.
* CategoricalImputer - It allows for imputation of categorical variables before conversion to integers.
- sklearn.preprocessing:
* Imputer - It allows us to fill in missing numerical values in scikit-learn
- sklearn.pipeline:
* FeatureUnion - It allows us to combine multiple pipelines of features into a single pipeline of features.
'''

# Import necessary modules

# Fill LotFrontage missing values with 0
X.LotFrontage = X['LotFrontage'].fillna(0)

# Setup the pipeline steps: steps
steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

# Create the pipeline: xgb_pipeline
xgb_pipeline = Pipeline(steps)

# Cross-validate the model
cross_val_scores = cross_val_score(xgb_pipeline, X.to_dict(
    'records'), y, cv=10, scoring='neg_mean_squared_error')

# Print the 10-fold RMSE
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))


kidney_data = pd.read_csv(
    'Extreme Gradient Boosting with XGBoost\kidney_disease.csv')

X = kidney_data.drop(['id', 'classification'], axis=1)

y = kidney_data['classification'].map({'ckd': 0, 'notckd': 1})
# Import necessary modules

# Check number of nulls in each feature column
nulls_per_column = X.isnull().sum()
print(nulls_per_column)

# Create a boolean mask for categorical columns
categorical_feature_mask = X.dtypes == object

# Get list of categorical column names
categorical_columns = X.columns[categorical_feature_mask].tolist()

# Get list of non-categorical column names
non_categorical_columns = X.columns[~categorical_feature_mask].tolist()

# Apply numeric imputer
numeric_imputation_mapper = DataFrameMapper(
    [([numeric_feature], SimpleImputer(strategy="median"))
     for numeric_feature in non_categorical_columns],
    input_df=True,
    df_out=True
)

# Apply categorical imputer
categorical_imputation_mapper = DataFrameMapper(
    [([category_feature], SimpleImputer(strategy="most_frequent"))
     for category_feature in categorical_columns],
    input_df=True,
    df_out=True
)


# Import FeatureUnion

# Combine the numeric and categorical transformations
numeric_categorical_union = FeatureUnion(
    [("num_mapper", numeric_imputation_mapper), ("cat_mapper", categorical_imputation_mapper)])

# Apply the transformations
transformed_data = numeric_categorical_union.fit_transform(X)

# Define the Dictifier transformer


class Dictifier(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        return df.to_dict(orient="records")


# Create full pipeline
pipeline = Pipeline([("featureunion", numeric_categorical_union), ("dictifier", Dictifier(
)), ("vectorizer",  DictVectorizer(sort=False)), ("clf", xgb.XGBClassifier(max_depth=3))])

# Perform cross-validation
cross_val_scores = cross_val_score(
    pipeline, X, y, scoring="roc_auc", cv=3)

# Print avg. AUC
print("3-fold AUC: ", np.mean(cross_val_scores))


'''
Tuning XGBoost hyperparameters
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
names = ['crime', 'zone', 'industry', 'charles', 'no', 'rooms', 'age', 'distance', 'radial', 'tax', 'pupil', 'aam', 'lower', 'med_price']

data = pd.read_csv('boston_housing.csv', names=names)
X, y = data.iloc[:, :-1], data.iloc[:, -1]
xgb_pipeline = Pipeline[('st_scaler', StandardScaler()), ('xgb_model', XGBRegressor())]
gbm_param_grid = { 'xgb_model__subsample': np.arange(.05, 1, .05), 'xgb_model__max_depth': np.arange(3, 20, 1), 'xgb_model__colsample_bytree': np.arange(.1, 1.05, .05) }
randomized_neg_mse = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=gbm_param_grid, n_iter=10,  scoring='neg_mean_squared_error', cv=4)
randomized_neg_mse.fit(X, y)
print('Best rmse: ', np.sqrt(np.abs(randomized_neg_mse.best_score_))) -> in

Best rmse: 3.9966784203040677 -> out

print('Best model: ', randomized_neg_mse.best_estimator_) -> in

Best model: Pipeline(steps=[('st_scaler', StandardScaler(copy=True, withe_mean=True, with_std=True)),
('xgb_model', XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.9500000000000000000029, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=8, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=0.90000000000000000013))]) -> out
'''

# Create the parameter grid
gbm_param_grid = {'clf__learning_rate': np.arange(0.05, 1, 0.05), 'clf__max_depth': np.arange(
    3, 10, 1), 'clf__n_estimators': np.arange(50, 200, 50)}

# Perform RandomizedSearchCV
randomized_roc_auc = RandomizedSearchCV(
    estimator=pipeline, param_distributions=gbm_param_grid, n_iter=2, scoring='roc_auc', cv=2, verbose=1)

# Fit the estimator
randomized_roc_auc.fit(X, y)

# Compute metrics
print(randomized_roc_auc.best_score_)
print(randomized_roc_auc.best_estimator_)
