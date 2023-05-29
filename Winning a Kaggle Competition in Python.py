''' Kaggle competitions process '''

''' 
Competitions overview
- Kaggle is a web platform for Data Science and Machine Learning competitions.
* It allows us to solve Data Science challenges and compete with other participants in building the best predictive models.

Kaggle benefits
- Get practical experience on the real-world data
- Develop portfolio projects
- Meet a great Data Science community
- Try new domain or model type
- Keep up-to-date with te best performing methods

Competition process
Problem -> Data -> Model -> Submission -> Leaderboard

How to participate
- Go to http://kaggle.com website and select the competition
- Download the data
- Start building the models!

Train and Test data
import pandas as pd

# Read train data
taxi_train = pd.read_csv('taxi_train.csv')
taxi_train.columns.to_list() -> in

['Key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'] -> out

* The first column is an ID variable called 'key'
* The 'fare_amount' is a target variable we'd like to predict 
* The rest of the columns are features we could use to build the model.

# Read test data
taxi_test = pd.read_csv(taxi_test.csv')
taxi_test.columns.to_list() -> in

['Key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'] -> out

* It has the same list of columns except for the 'fare_amount', as this is the column we should predict.

Sample submission
# Read sample submission
taxi_sample_sub = pd.read_csv('taxi_sample_submission.csv')
taxi_sample_sub.head() -> in

                            key     fare_amount
0   2015-01-27 13:08:24.0000002     11.35
1   2015-01-27 13:08:24.0000003     11.35
2   2011-11-08 11:53:44.0000002     11.35
3   2012-12-01 21:12:12.0000002     11.35
4   2012-12-01 21:12:12.0000003     11.35 -> out
'''

# Import pandas

# Read train data
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import itertools
from math import sqrt
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')

# Look at the shape of the data
print('Train shape:', train.shape)

# Look at the head() of the data
print(train.head())


# Read the test data
test = pd.read_csv('test.csv')
# Print train and test columns
print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())

# Read the sample submission file
sample_submission = pd.read_csv('sample_submission.csv')

# Look at the head() of the sample submission
print(sample_submission.head())


'''
Prepare your first submission
What is submission
- It is usually a csv file that contains our test predictions and is submitted to the kaggle platform.
* Kaggle internally measures the quality of the predictions and shows the results on the Leaderboard.

New York city taxi fare prediction
# Read train data
taxi_train = pd.read_csv('taxi_train.csv')
taxi_train.columns.to_list() -> in

['Key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'] -> out

Problem type
- Before creating any Machine Learning model, we should determine the problem type we're addressing:
* Whether it's classification, regression or some other problem.

import matplotlib.pyplot as plt

# Plot a histogram
taxi_train.fare_amount.hist(bins=30, alpha=0.5)
plt.show()

Build a model
from sklearn.linear_model import LinearRegression

# Create a LinearRegression object
lr = LinearRegression()

# Fit the model on the train data
lr.fit(X = taxi_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']], y = taxi_train['fare_amount'])

Predict on test set
# Select features
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

# Make predictions on the test data
taxi_test['fare_amount'] = lr.predict(taxt_test[features])

Prepare submission
# Read a sample submission file
taxi_sample_sub = pd.read_csv('taxi_sample_submission.csv')
taxi_sample_sub.head(1) -> in

                            key     fare_amount
0   2015-01-27 13:08:24.0000002     11.35 -> out

# Prepare a submission file
taxi_submission = taxi_test[['key', 'fare_amount']]

# Save the submission file as .csv
taxi_submission.to_csv('first_sub.csv', index=False)
'''


# Plot a histogram
train.sales.hist(bins=30, alpha=0.5)
plt.show()


# Read the train data
train = pd.read_csv('train.csv')

# Create a Random Forest object
rf = RandomForestRegressor()

# Train a model
rf.fit(X=train[['store', 'item']], y=train['sales'])


# Read test and sample submission data
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Show the head() of the sample_submission
print(sample_submission.head())

# Get predictions for the test set
test['sales'] = rf.predict(test[['store', 'item']])

# Write test predictions using the sample_submission format
test[['id', 'sales']].to_csv('kaggle_submission.csv', index=False)


'''
Public vs Private leaderboard
Competition metric
- Each competition specifies a singe metric that is used to rank the participants.

Competition metric
Evaluation metric                               Type of problem
Area Under the ROC (AUC)                        Classification
F1 Score (F1)                                   Classification
Mean Log Loss (LogLoss)                         Classification
Mean Absolute Error (MAE)                       Regression
Mean Squared Error (MSE)                        Regression
Mean Average Precision at K (MAPK, MAP@K)       Ranking

Leaderboards
# Write a submission file to the disk
submission[['id', 'target']].to_csv('submission_1.csv', index=False)
'''


# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']], label=train['sales'])

# Define xgboost parameters
params = {'objective': 'reg:linear', 'max_depth': 2, 'verbosity': 0}

# Train xgboost model
xg_depth_2 = xgb.train(params=params, dtrain=dtrain)


# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']], label=train['sales'])

# Define xgboost parameters
params = {'objective': 'reg:linear', 'max_depth': 8, 'verbosity': 0}

# Train xgboost model
xg_depth_8 = xgb.train(params=params, dtrain=dtrain)


# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']], label=train['sales'])

# Define xgboost parameters
params = {'objective': 'reg:linear', 'max_depth': 15, 'verbosity': 0}

# Train xgboost model
xg_depth_15 = xgb.train(params=params, dtrain=dtrain)


dtrain = xgb.DMatrix(data=train[['store', 'item']])
dtest = xgb.DMatrix(data=test[['store', 'item']])

# For each of 3 trained models
for model in [xg_depth_2, xg_depth_8, xg_depth_15]:
    # Make predictions
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)

    # Calculate metrics
    mse_train = mean_squared_error(train['sales'], train_pred)
    mse_test = mean_squared_error(test['sales'], test_pred)
    print('MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))


''' Dive into the Competition '''

'''
Understand the problem
Solution Workflow
Understand the Problem & Competition metric -> Exploratory data analysis (EDA) -> Local Validation to prevent Overfitting -> Modeling

Understand the Problem
- Data type: tabular data, time series, images, text, etc.
- Problem type: classification, regression, ranking, etc.
- Evaluation metric: ROC AUC, F1-Score, MAE, MSE, etc.

Metric definition
# Some classification and regression metrics
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error

- Root Mean Squared Logarithmic Error (RMSLE) as an evaluation metric.
* It is not implemented in scikit-learn.

import numpy as np

def rmsle(y_true, y_pred):
    diffs = np.log(y_true + 1) - np.log(y_pred + 1)
    squares = np.power(diffs, 2)
    err = np.sqrt(np.mean(squares))
    return err
'''


# Import MSE from sklearn

# Define your own MSE function

def own_mse(y_true, y_pred):
    # Raise differences to the power of 2
    squares = np.power(y_true - y_pred, 2)
    # Find mean over all observations
    err = np.mean(squares)
    return err


print('Sklearn MSE: {:.5f}. '.format(
    mean_squared_error(y_regression_true, y_regression_pred)))
print('Your MSE: {:.5f}. '.format(
    own_mse(y_regression_true, y_regression_pred)))


# Import log_loss from sklearn

# Define your own LogLoss function

def own_logloss(y_true, prob_pred):
    # Find loss for each observation
    terms = y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
    # Find mean over all observations
    err = np.mean(terms)
    return -err


print('Sklearn LogLoss: {:.5f}'.format(
    log_loss(y_classification_true, y_classification_pred)))
print('Your LogLoss: {:.5f}'.format(own_logloss(
    y_classification_true, y_classification_pred)))


'''
Initial EDA
Goals of EDA
- Size of the data
- Properties of the target variable
- Properties of the features
- Generate ideas for feature engineering

Two sigma connect: rental listing inquiries
- Problem statement
* Predict the popularity of an apartment rental listing

- Target variable
* interest_level

- Problem type
* Classification with 3 classes: 'high', 'medium' and 'low'

- Metric
* Multi-class logarithmic loss

EDA. Part 1
# Size of the data
twosigma_train = pd.read_csv('twosigma_train.csv')
print('Train shape:', twosigma_train.shape)

twosigma_test = pd.read_csv('twosigma_test.csv')
print('Train shape:', twosigma_test.shape) -> in

Train shape: (49352, 11)
Test shape: (74659, 10) -> out

print(twosigma_train.columns.tolist()) -> in

['id', 'bathrooms', 'bedrooms', 'building_id', 'latitude', 'longitude', 'manager_id', 'price', 'interest_level'] -> out

twosigma_train.interest_level.value_counts() -> in

low         34284
medium      11229
high        3839 -> out

# Describe the train data
twosigma_train.describe() -> in

        bathrooms       bedrooms        latitude        longitude       price
count   49352.00000     49352.00000     49352.00000     49352.00000     4.935200e+04
mean    1.21218         1.541640        40.741545       -73.955716      3.830174e+03
std     0.50142         1.115018        0.638535        1.177912        2.206687e+04
min     0.00000         0.000000        0.00000         -118.271000     4.300000e+01
25%     1.00000         1.000000        40.728300       -73.991700      2.500000e+03
50%     1.00000         1.000000        40.751800       -73.977900      3.150000e+03
75%     1.00000         2.000000        40.774300       -73.954800      4.100000e+03
max     10.00000        8.000000        44.883500       0.000000        4.490000e+06 -> out

EDA. Part 2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Find the median price by the interest level
prices = twosigma_train.groupby('interest_level', as_index=False)['price'].median()

# Draw a bar plot
fig = plt.figure(figsize=(7, 5))
plt.bar(prices.interest_level, prices.price, width=0.5, alpha=0.8)

# Set titles
plt.xlabel('Interest level')
plt.ylabel('Median price')
plt.title('Median listing price across interest level')

# Show the plot
plt.show()

- To get the distance in kilometers between two geo-coordinates, you will use Haversine distance. Its calculation is available with the haversine_distance() function defined for you. The function expects train DataFrame as input.
'''

# Shapes of train and test data
print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Train head()
print(train.head())

# Describe the target variable
print(train.fare_amount.describe())

# Train distribution of passengers within rides
print(train.passenger_count.value_counts())


# Calculate the ride distance
train['distance_km'] = haversine_distance(train)

# Draw a scatterplot
plt.scatter(x=train['fare_amount'], y=train['distance_km'], alpha=0.5)
plt.xlabel('Fare amount')
plt.ylabel('Distance, km')
plt.title('Fare amount based on the distance')

# Limit on the distance
plt.ylim(0, 50)
plt.show()


# Create hour feature
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['hour'] = train.pickup_datetime.dt.hour

# Find median fare_amount for each hour
hour_price = train.groupby('hour', as_index=False)['fare_amount'].median()

# Plot the line plot
plt.plot(hour_price['hour'], hour_price['fare_amount'], marker='o')
plt.xlabel('Hour of the day')
plt.ylabel('Median fare amount')
plt.title('Fare amount based on day time')
plt.xticks(range(24))
plt.show()


'''
Local validation
- Using hold-out sets
* It is similar to the usual test data, but the target variable is known.
* It allows to compare predictions with the actual values and gives a fair estimate of the model's performance.

- Using K-fold cross-validation
* It gives our model the opportunity to train on multiple train-test splits instead of using a single holdout set.
* It gives a better indication of how well our model will perform on unseen data.

K-fold cross-validation
# Import KFold
from sklearn.model_selection import KFold

# Create a KFold object
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# Loop through each cross-validation split
for train_index, test_index in kf.split(train):
    # Get training and testing data for the corresponding split
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

- Stratified K-fold
* These folds are made by preserving the percentage of samples for each class of this variable.
* It is useful when we have a classification problem with high class imbalance in the target variable or our data size is very small.
* The only difference between it and the K-fold is that on top of the train data, we should also pass the target variable into the split() call in order to make a stratification.

Stratified K-fold
# Import StratifiedKFold
from sklearn.model_selection import StratifiedKFold

# Create a StratifiedKFold object
str_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Loop through each cross-validation split
for train_index, test_index in str_kf.split(train, train['target']):
    # Get training and testing data for the corresponding split
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

- The general rule is to prefer Stratified K-Fold over usual K-Fold in any classification problem.
'''

# Import KFold

# Create a KFold object
kf = KFold(n_splits=3, shuffle=True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in kf.split(train):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(
        sum(cv_train.interest_level == 'medium')))
    fold += 1


# Import StratifiedKFold

# Create a StratifiedKFold object
str_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in str_kf.split(train, train['interest_level']):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(
        sum(cv_train.interest_level == 'medium')))
    fold += 1


'''
Validation usage
Data leakage
- Leakage causes a model to seem accurate until we start making predictions in a real-world environment.
* We then reslize that the model is of low quality and becomes absolutely useless.

Types of Leakages
- Leak in features
* Using data that will not be available in the real setting
- Lak in validation strategy
* Validation strategy differs from the real-world situation.

Time data
Time K-fold cross-validation
# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit

# Create a TimeSeriesSplit object
time_kfold = TimeSeriesSplit(n_splits=5)

# Sort train by date
train = train.sort_values('date')

# Loop through each cross-validation split
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

Validation pipeline
# List for the results
fold_metrics = []

for train_index, test_index in CV_STRATEGY.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

    # Train a model
    model.fit(cv_train)

    # Make predictions
    predictions = model.predict(cv_test)

    # Calculate the metric
    metric = evaluate(cv_test, predictions)
    fold_metrics.append(metric)

* CV_STRATEGY : cross-validation strategy being used

Model comparison
Fold number     Model A MSE     Model B MSE
Fold 1          2.95            2.97
Fold 2          2.84            2.45
Fold 3          2.62            2.73
Fold 4          2.79            2.83 

Overall validation score
import numpy as np

# Simple mean over the folds
mean_score = np.mean(fold_metrics)

# Overall validation score
overall_score_minimizing = np.mean(fold_metrics) + np.std(fold_metrics)

# Or
overall_score_maximizing = np.mean(fold_metrics) - np.std(fold_metrics)

Model comparison
Fold number     Model A MSE     Model B MSE
Fold 1          2.95            2.97
Fold 2          2.84            2.45
Fold 3          2.62            2.73
Fold 4          2.79            2.83 
Mean            2.80            2.75
Overall         2.919           2.935
'''

# Create TimeSeriesSplit object
time_kfold = TimeSeriesSplit(n_splits=3)

# Sort train data by date
train = train.sort_values('date')

# Iterate through each split
fold = 0
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold :', fold)
    print('Train date range: from {} to {}'.format(
        cv_train.date.min(), cv_train.date.max()))
    print('Test date range: from {} to {}\n'.format(
        cv_test.date.min(), cv_test.date.max()))
    fold += 1


# Sort train data by date
train = train.sort_values('date')

# Initialize 3-fold time cross-validation
kf = TimeSeriesSplit(n_splits=3)

# Get MSE scores for each cross-validation split
mse_scores = get_fold_mse(train, kf)

print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
print('MSE by fold: {}'.format(mse_scores))
print('Overall validation MSE: {:.5f}'.format(
    np.mean(mse_scores) + np.std(mse_scores)))


''' Feature Engineering '''

'''
Feature engineering
Modeling stage
Modeling = Preprocess Data -> Create New Features -> Improve Models -> Apply Tricks

* Important rule is to twek only a single thing at a time, because changing multiple things does not allow us to detect what actually works and what doesn't.

Feature engineering
- It is the process of creating new features.
* It helps our ML models to get the additional information and consequently to better predict the target variable.

- Some sources of ideas for feature engineering include
* Prior experience
* EDA
* Domain knowledge

Feature type
- Numerical
- Categorical
- Datetime
- Coordinates
- Text
- Images

Creating features
# Concatenate the train and test data
data = pd.concat([train, test])

# Create new features for the data DataFrame . . .

# Get the train and test back
train = data[data.id.isin(train.id)]
test = data[data.id.isin(test.id)]

Arithmetical features
# Two sigma connect competition
two_sigma.head(1) -> in

    id  bathrooms   bedrooms    price   interest_level
0   10  1.5         3           3000    medium          -> out

# Arithmetical features
two_sigma['price_per_bedroom'] = two_sigma.price / two_sigma.bedrooms
two_sigma['rooms_number'] = two_sigma.bedrooms + two_sigma.bathrooms

Datetime features
# Demand forecasting challenge
dem.head(1) -> in

    id      date        store   item    sales
0   100000  2017-12-01  1       1       19      -> out

# Convert date to the datetime object
dem['date'] = pd.to_datetime(dem['date'])

# Year features
dem['year'] = dem['date'].dt.year

# Month features
dem['month'] = dem['date'].dt.month

# Week features
dem['week'] = dem['date'].dt.week -> in

date        year    month   week
2017-12-01  2017    12      48
2017-12-02  2017    12      48
2017-12-03  2017    12      48
2017-12-04  2017    12      48 -> out

# Day features
dem['dayofyear'] = dem['date'].dt.dayofyear
dem['dayofmonth'] = dem['date'].dt.day
dem['dayofweek'] = dem['date'].dt.dayofweek -> in

date        dayofyear   dayofmonth  dayofweek
2017-12-01  335         1           4
2017-12-02  336         2           5
2017-12-03  337         3           6
2017-12-04  338         4           0 -> out

- NB*: The day of the week encodes Monday as 0, Tuesday as 1 proceeding to Sunday as 6.
'''

# Look at the initial RMSE
print('RMSE before feature engineering:', get_kfold_rmse(train))

# Find the total area of the house
train['TotalArea'] = train['TotalBsmtSF'] + \
    train['FirstFlrSF'] + train['SecondFlrSF']
print('RMSE with total area:', get_kfold_rmse(train))

# Find the area of the garden
train['GardenArea'] = train['LotArea'] - train['FirstFlrSF']
print('RMSE with garden area:', get_kfold_rmse(train))

# Find total number of bathrooms
train['TotalBath'] = train['FullBath'] + train['HalfBath']
print('RMSE with number of bathrooms:', get_kfold_rmse(train))


# Concatenate train and test together
taxi = pd.concat([train, test])

# Convert pickup date to datetime object
taxi['pickup_datetime'] = pd.to_datetime(taxi['pickup_datetime'])

# Create a day of week feature
taxi['dayofweek'] = taxi['pickup_datetime'].dt.dayofweek

# Create an hour feature
taxi['hour'] = taxi['pickup_datetime'].dt.hour

# Split back into train and test
new_train = taxi[taxi['id'].isin(train['id'])]
new_test = taxi[taxi['id'].isin(test['id'])]


'''
Categorical features
Label encoding
ID  Categorical feature
1   A
2   B
3   C
4   A
5   D
6   A

ID  Label-encoded
1   0
2   1
3   2
4   0
5   3
6   0

# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Encode a categorical feature
df['cat_encoded'] = le.fit_transform(df['cat'])

    ID  cat     cat_encoded
0   1   A       0
1   2   B       1
2   3   C       2
3   4   A       0

* The problem with Label encoding is that we implicitly assume that there is a ranking dependency between the categories.
* Such an approach is hamful to linear models, although it still works for tree-based models.

One-Hotl encoding
ID  Categorical feature
1   A
2   B
3   C
4   A
5   D
6   A

ID  Cat == A    Cat == B    Cat == C    Cat == D
1   1           0           0           0
2   0           1           0           0
3   0           0           1           0
4   1           0           0           0 
5   0           0           0           1
6   1           0           0           0 

# Create One-Hot encoded features
ohe = pd.get_dummies(df['cat'], prefix='ohe_cat')

# Drop the initial feature
df.drop('cat', axis=1, inplace=True)

# Concatenate OHE features to the dataframe

    ID  ohe_cat_A   ohe_cat_B   ohe_cat_C   ohe_cat_D
0   1   1           0           0           0
1   2   0           1           0           0
2   3   0           0           1           0
3   4   1           0           0           0

* The drawback of such approach arises if the feature has a lot of different categories.

Binary Features
- It is a special case of categorical features
- It relates to categorical variables that have only two possible values.
- For such features, we always apply label encoding.

# DataFrame with a binary feature
binary_feature -> in

    binary_feat
0   Yes
1   No          -> out

le = LabelEncoder()
binary_feature['binary_encoded'] = le.fit_transform(binary_feature['binary_feat']) -> in

    binary_feat     binary_encoded
0   Yes             1
1   No              0               -> out

Other encoding approaches
- Backward Difference Coding
- BaseN
- Binary
- CatBoost Encoder
- Hashing
- Helmert Coding
- James-Stein Encoder
- Leave One Out
- M-estimate
- One Hot
- Ordinal
- Polynomial Coding
- Sum Coding
- Target Encoder
- Weight of Evidence
'''

# Concatenate train and test together
houses = pd.concat([train, test])

# Label encoder
le = LabelEncoder()

# Create new features
houses['RoofStyle_enc'] = le.fit_transform(houses['RoofStyle'])
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Look at new features
print(houses[['RoofStyle', 'RoofStyle_enc',
      'CentralAir', 'CentralAir_enc']].head())


# Concatenate train and test together
houses = pd.concat([train, test])

# Look at feature distributions
print(houses['RoofStyle'].value_counts(), '\n')
print(houses['CentralAir'].value_counts())


# Concatenate train and test together
houses = pd.concat([train, test])

# Label encode binary 'CentralAir' feature
le = LabelEncoder()
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Create One-Hot encoded features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')

# Concatenate OHE features to houses
houses = pd.concat([houses, ohe], axis=1)

# Look at OHE features
print(houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))


'''
Target encoding
High cardinality categorical features
- These are categorical features that have a large number of category values (at least over 10 different category values).
* Label encoder provides distinct number for each category
* One-hot encoder creates new feature for each category value
* Target encoder creates only a single column, but it also introduces the correlation between the categories and the target variable.

Mean target encoding
Train ID    Categorical     Target
1           A               1
2           B               0
3           B               0
4           A               1
5           B               0
6           A               0
7           B               1

Test ID     Categorical     Target
10          A               ?
11          A               ?
12          B               ?
13          A               ?

- Steps to apply target encoding include:
* Calculate mean on the train, apply to the test
* Split train into K folds. Calculate mean on (K-1) folds, apply to the K-th fold
* Add mean target encoded feature to the model

Calculate mean on the train
Train ID    Categorical     Target
1           A               1
2           B               0
3           B               0
4           A               1
5           B               0
6           A               0
7           B               1

- For category A = ( 1 + 1 + 0 ) / 3 = 0.66
- For category B = ( 0 + 0 + 0 + 1 ) / 4 = 0.25

Test encoding
Test ID     Categorical     Target  Mean encoded
10          A               ?       0.66
11          A               ?       0.66
12          B               ?       0.25
13          A               ?       0.66

Train encoding using out-of-fold
Train ID    Categorical     Target  Fold    Mean encoded
1           A               1       1       0
2           B               0       1       0.5
3           B               0       1       0.5
4           A               1       1       0
5           B               0       2       0
6           A               0       2       1
7           B               1       2       0

- Mean fold 1 A = 0 (i.e from out-of- fold obsevation fold 2)
- Mean fold 1 B = ( 0 + 1 ) / 2 = 0.5 (i.e from out-of- fold obsevation fold 2)
- Mean fold 2 A = ( 1 + 1 ) / 2 = 1 (i.e from out-of- fold obsevation fold 1)
- Mean fold 2 B = ( 0 + 0 ) / 2 = 0 (i.e from out-of- fold obsevation fold 1)

Practical guides
- Smoothing
mean_enc(i) = target_sum(i) / n(i)

smoothed_mean_enc(i) = ( target_sum(i) + alpha * global_mean ) / ( n(i) + alpha )

* alpha : usually, values from 5 to 10 work pretty well by default.

- New categories
* Fill new categories in the test data with a global_mean

Train ID    Categorical     Target
1           A               1
2           B               0
3           B               0
4           A               0
5           B               1

Test ID     Categorical     Target  Mean encoded
10          A               ?       0.43
11          B               ?       0.38
12          C               ?       0.40

* For binary classification usually mean target encoding is used
* For regression mean could be changed to median, quartiles, etc.
* For multi-class classification with N classes we create N features with target mean for each category in one vs. all fashion
'''


def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean *
                        alpha) / (category_size + alpha)

    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values


def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)

    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(
            cv_train, cv_test, target, categorical, alpha)

        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values


def mean_target_encoding(train, test, target, categorical, alpha=5):

    # Get the train feature
    train_feature = train_mean_target_encoding(
        train, target, categorical, alpha)

    # Get the test feature
    test_feature = test_mean_target_encoding(
        train, test, target, categorical, alpha)

    # Return new features to add to the model
    return train_feature, test_feature


# Create 5-fold cross-validation
kf = KFold(n_splits=5, random_state=123, shuffle=True)

# For each folds split
for train_index, test_index in kf.split(bryant_shots):
    cv_train, cv_test = bryant_shots.iloc[train_index], bryant_shots.iloc[test_index]

    # Create mean target encoded feature
    cv_train['game_id_enc'], cv_test['game_id_enc'] = mean_target_encoding(
        train=cv_train, test=cv_test, target='shot_made_flag', categorical='game_id', alpha=5)

    # Look at the encoding
    print(cv_train[['game_id', 'shot_made_flag', 'game_id_enc']].sample(n=1))


# Create mean target encoded feature
train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(
    train=train, test=test, target='SalePrice', categorical='RoofStyle', alpha=10)

# Look at the encoding
print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())


'''
Missing data
- Some machine learning algorithms like XGBoost or LightGBM can treat missing data without any preprocessing.

ID  Categorical feature     Numerical feature   Binary target
1   A                       5.1                 1
2   B                       7.2                 0
3   C                       3.4                 0
4   A                       NaN                 1
5   NaN                     2.6                 0
6   A                       5.3                 0

- Numerical data
* Mean / median imputation
* Constant value imputation (e.g -999) : It's not a good choice for linear models but works perfectly for tree-based models.

Categorical data
* Most frequent category imputation
* New category imputation (e.g 'MISS') : It allows the model to get information that this observation had missing value

Find missing data
df.isnull().head(1) -> in

    ID      cat     num     target
0   False   False   False   False -> out

df.isnull().sum() -> in

ID      0
cat     1
num     1
target  0 -> out

Numerical missing data
# Import SimpleImputer
from sklearn.impute import SimpleImputer

# Different types of imputers
mean_imputer = SimpleImputer(strategy = 'mean')
constant_imputer = SimpleImputer(strategy = 'constant', fill_value = -999)

# Imputation
df[['num']] = mean_imputer.fit_transform(df[['num']])

Categorical missing data
# Import SimpleImputer
from sklearn.impute import SimpleImputer

# Different types of imputers
frequent_imputer = SimpleImputer(strategy = 'most_frequent')
constant_imputer = SimpleImputer(strategy = 'constant', fill_value = 'MISS')

# Imputation
df[['cat']] = constant_imputer.fit_transform(df[['cat']])
'''

# Read DataFrame
twosigma = pd.read_csv('twosigma_train.csv')

# Find the number of missing values in each column
print(twosigma.isnull().sum())

# Look at the columns with the missing values
print(twosigma[['building_id', 'price']].head())


# Import SimpleImputer

# Create mean imputer
mean_imputer = SimpleImputer(strategy='mean')

# Price imputation
rental_listings[['price']] = mean_imputer.fit_transform(
    rental_listings[['price']])


# Import SimpleImputer

# Create constant imputer
constant_imputer = SimpleImputer(strategy='constant', fill_value='MISSING')

# building_id imputation
rental_listings[['building_id']] = constant_imputer.fit_transform(
    rental_listings[['building_id']])


''' Modeling '''

'''
Baseline model
New York city taxi validation
# Read data
taxi_train = pd.read_csv('taxi_train.csv')
taxi_test = pd.read_csv('taxi_test.csv')

from sklearn.model_selection import train_test_split

# Create local validation
validation_train, validation_test = train_test_split(taxi_train, test_size=0.3, random_state=123)

Baseline model 1
import numpy as np

# Assign the mean fare amount to all the test observations
taxi_test['fare_amount'] = np.mean(taxi_train.fare_amount)

# Write predictions to the file
taxi_test[['id', 'fare_amount']].to_csv('mean_sub.csv', index=False)

Validation RMSE     Public LB RMSE  Public LB Position
9.986               9.409           1449 / 1500

Baseline model 2
# Calculate the mean fare amount by group
naive_prediction_groups = taxi_train.groupby('passenger_count').fare_amount.mean()

# Make predictions on the test set
taxi_test['fare_amount'] = taxi_test.passenger_count.map(naive_prediction_groups)

# Write predictions to the file
taxi_test[['id', 'fare_amount']].to_csv('mean_group_sub.csv', index=False)

Validation RMSE     Public LB RMSE  Public LB Position
9.978               9.407           1411 / 1500

Baseline model 3
# Select only numeric features
features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']

from sklearn.ensemble import GradientBoostingRegressor

# Train a Gradient Boosting model
gb = GradientBoostingRegressor()
gb.fit(taxi_train[features], taxi_train.fare_amount)

# Make predictions on the test data
taxi_test['fare_amount'] = gb.predict(taxi_test[features])

# Write predictions to the file
taxi_test[['id', 'fare_amount']].to_csv('gb_sub.csv', index=False)

Validation RMSE     Public LB RMSE  Public LB Position
5.996               4.595           1109 / 1500

Intermediate results
Model               Validation RMSE     Public LB RMSE
Simple Mean         9.986               9.409
Group Mean          9.978               9.407
Gradient Boosting   5.996               4.595

Correlation with Public Leaderboard
Model       Validation RMSE     Public LB RMSE
Model A     3.500               3.800
Model B     3.300               4.100
Model C     3.200               3.900

* It is a sign that something could be wrong with our models or validation scheme.

Model       Validation RMSE     Public LB RMSE
Model A     3.400               3.900
Model B     3.100               3.400
Model C     2.900               3.300
'''


# Calculate the mean fare_amount on the validation_train data
naive_prediction = np.mean(validation_train['fare_amount'])

# Assign naive prediction to all the holdout observations
validation_test['pred'] = naive_prediction

# Measure the local RMSE
rmse = sqrt(mean_squared_error(
    validation_test['fare_amount'], validation_test['pred']))
print('Validation RMSE for Baseline I model: {:.3f}'.format(rmse))


# Get pickup hour from the pickup_datetime column
train['hour'] = train['pickup_datetime'].dt.hour
test['hour'] = test['pickup_datetime'].dt.hour

# Calculate average fare_amount grouped by pickup hour
hour_groups = train.groupby('hour')['fare_amount'].mean()

# Make predictions on the test set
test['fare_amount'] = test.hour.map(hour_groups)

# Write predictions
test[['id', 'fare_amount']].to_csv('hour_mean_sub.csv', index=False)


# Select only numeric features
features = ['pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'hour']

# Train a Random Forest model
rf = RandomForestRegressor()
rf.fit(train[features], train.fare_amount)

# Make predictions on the test data
test['fare_amount'] = rf.predict(test[features])

# Write predictions
test[['id', 'fare_amount']].to_csv('rf_sub.csv', index=False)


'''
Hyperparameter tuning
Iterations
Intermediate results
Model                   Validation RMSE     Public LB RMSE      Public LB Position
Simple Mean             9.986               9.409               1449 / 1500
Group Mean              9.978               9.407               1411 / 1500
Gradient Boosting       5.996               4.595               1109 / 1500
Add hour feature        5.553               4.352               1068 / 1500
Add distance feature    5.268               4.103               1006 / 1500
. . .                   . . .               . . .               . . . 

Hyperparameter optimization
Competition type            Feature engineering     Hyperparameter optimization
Classic Machine Learning    +++                     +
Deep Learning               -                       +++

* Classic Machine Learning ( with tabular or time series data)
* Deep Learning ( with text or images data), there is no need for feature engineering.
** Neural nets are generating features on their own, while we need to specify the architecture and a list of hyperparameters.

Hyperparameter optimization strategies
- Grid search : Choose the predefined grid of hyperparameter values
- Random search : Choose the search space of hyperparameter values
- Bayesian optimization : Choose the search space of hyperparameter values.
* It's difference with random or grid search is that it uses past evaluation results to choose the next hyperparameter values to evaluate.

Grid search
# Possible alpha values
alpha_grid = [0.01, 0.1, 1, 10]

from sklearn.linear_model import Ridge

result = {}

# For each value in the grid
for candidate_alpha in alpha_grid:
    # Create a model with a specific alpha value
    ridge_regression = Ridge(alpha = candidate_alpha)

    # Find the validation score for this model

    # Save the results for each alpha value
    results[candidate_alpha] = validation_score
'''

# Possible max depth values
max_depth_grid = [3, 6, 9, 12, 15]
results = {}

# For each value in the grid
for max_depth_candidate in max_depth_grid:
    # Specify parameters for the model
    params = {'max_depth': max_depth_candidate}

    # Calculate validation score for a particular hyperparameter
    validation_score = get_cv_score(train, params)

    # Save the results for each max depth value
    results[max_depth_candidate] = validation_score
print(results)


# Hyperparameter grids
max_depth_grid = [3, 5, 7]
subsample_grid = [0.8, 0.9, 1.0]
results = {}

# For each couple in the grid
for max_depth_candidate, subsample_candidate in itertools.product(max_depth_grid, subsample_grid):
    params = {'max_depth': max_depth_candidate,
              'subsample': subsample_candidate}
    validation_score = get_cv_score(train, params)

    # Save the results for each couple
    results[(max_depth_candidate, subsample_candidate)] = validation_score
print(results)


'''
Model ensembling
- It's the different ways to combine models.

Model blending
- This approach is to just find an average of our multiple models predictions.
- Regression problem
* Train two different models: A and B
* Make predictions on the test data

Test ID     Model A prediction  Model B prediction  Arithmetic mean
1           1.2                 1.5                 1.35
2           0.1                 0.4                 0.25
3           5.4                 7.2                 6.30

- Arithmetic mean : It works for both regression and classification problems.
- Geometric mean : Class probabilities predicted is better for classification

Model stacking
- Its idea is to train multiple single models, take their predictions and use these predictions as features in the 2nd level model.
- Steps involved in model stacking include:
* Split train data into two parts
* Train multiple models on Part 1
* Make predictions on Part 2
* Make predictions on the test data
* Train a new model on Part 2 using predictions as features ( a.k.a 2nd level model or meta-model )
* Make predictions on the test data using the 2nd level model

Stacking example
Train ID    feature_1   . . .   feature_N   Target
1           0.55        . . .   1.37        1
2           0.12        . . .   -2.50       0
3           0.65        . . .   3.14        0
4           0.10        . . .   2.87        1
5           0.54        . . .   -0.10       0

Test ID     feature_1   . . .   feature_N   Target
11          0.49        . . .   -2.32       ?
12          0.32        . . .   1.15        ?
13          0.91        . . .   0.81        ?

Split Train data in two parts
- Part 1
Train ID    feature_1   . . .   feature_N   Target
1           0.55        . . .   1.37        1
2           0.12        . . .   -2.50       0
3           0.65        . . .   3.14        0

* Train models A, B, C on Part 1

- Part 2
Train ID    feature_1   . . .   feature_N   Target  A_pred  B_pred  C_pred
4           0.10        . . .   2.87        1       0.71    0.52    0.98
5           0.54        . . .   -0.10       0       0.45    0.32    0.24

* Make predictions on Part 2 of the train data
* The columns with the predictions are denoted as A_pred, B_pred and C_pred

Make predictions on the test data
Test ID     feature_1   . . .   feature_N   Target  A_pred  B_pred  C_pred
11          0.49        . . .   -2.32       ?       0.62    0.45    0.81
12          0.32        . . .   1.15        ?       0.31    0.52    0.41
13          0.91        . . .   0.81        ?       0.74    0.55    0.92

Train a new model on Part 2 using predictions as features
Train ID    Target  A_pred  B_pred  C_pred
4           1       0.71    0.52    0.98
5           0       0.45    0.32    0.24

* Train 2nd level model on Part 2

Make predictions on the test data using the 2nd level model
Test ID     Target  A_pred  B_pred  C_pred  Stacking prediction
11          ?       0.62    0.45    0.81    0.73
12          ?       0.31    0.52    0.41    0.35
13          ?       0.74    0.55    0.92    0.88

* Usually, the 2nd level model is some simple model like Linear or Logistic Regressions. Also, note that you were not using intercept in the Linear Regression just to combine pure model predictions.
'''


# Train a Gradient Boosting model
gb = GradientBoostingRegressor().fit(train[features], train.fare_amount)

# Train a Random Forest model
rf = RandomForestRegressor().fit(train[features], train.fare_amount)

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])

# Find mean of model predictions
test['blend'] = (test['gb_pred'] + test['rf_pred']) / 2
print(test[['gb_pred', 'rf_pred', 'blend']].head(3))


# Split train data into two parts
part_1, part_2 = train_test_split(train, test_size=0.5, random_state=123)

# Train a Gradient Boosting model on Part 1
gb = GradientBoostingRegressor().fit(part_1[features], part_1.fare_amount)

# Train a Random Forest model on Part 1
rf = RandomForestRegressor().fit(part_1[features], part_1.fare_amount)

# Make predictions on the Part 2 data
part_2['gb_pred'] = gb.predict(part_2[features])
part_2['rf_pred'] = rf.predict(part_2[features])

# Make predictions on the test data
test['gb_pred'] = gb.predict(test[features])
test['rf_pred'] = rf.predict(test[features])


# Create linear regression model without the intercept
lr = LinearRegression(fit_intercept=False)

# Train 2nd level model on the Part 2 data
lr.fit(part_2[['gb_pred', 'rf_pred']], part_2.fare_amount)

# Make stacking predictions on the test data
test['stacking'] = lr.predict(test[['gb_pred', 'rf_pred']])

# Look at the model coefficients
print(lr.coef_)


'''
Final tips
Save information
- Save folds distribution to files
* Goal is to track the validation score during the competition.
* The validation score should always be calculated on the same folds.

- Save model runs
* It allow us to reproduce our experiments or go back if needed.
* One of the possibilities could be to create a separate git commit for each model run or submission

- Save model predictions to the disk
* It will allow us to simply build model ensembles near the end.

- Save performance results
* It could be done as comments to the git commits or as notes in a separate document

Kaggle forum and kernels
- Kaggle forum
* Competition discussion by the participants

- Kaggle kernels
* Scripts and notebooks shared by the participants
* Cloud computational environment

Forum and kernels usage
- Before the competition
Forum : Read winners' solutions from the past similar competitions 
Kernels : Go through baseline approaches from the past similar competitions

- During the competition
Forum : Follow the discussion to find the ideas and approaches for the problem
Kernels : Look at EDA, baseline models and validation strategies used by others

- After the competition
Forum : Read winners' solutions
Kernels : Look at the final solutions code sharing

Select final submissions
- Best submission on the local validation
- Best submission on the Public Leaderboard
'''

# Drop passenger_count column
new_train_1 = train.drop('passenger_count', axis=1)

# Compare validation scores
initial_score = get_cv_score(train)
new_score = get_cv_score(new_train_1)

print('Initial score is {} and the new score is {}'.format(initial_score, new_score))

# Create copy of the initial train DataFrame
new_train_2 = train.copy()

# Find sum of pickup latitude and ride distance
new_train_2['weird_feature'] = new_train_2['pickup_latitude'] + \
    new_train_2['distance_km']

# Compare validation scores
initial_score = get_cv_score(train)
new_score = get_cv_score(new_train_2)

print('Initial score is {} and the new score is {}'.format(initial_score, new_score))


'''
Kaggle vs Data Science
- Data analytics
* Kaggle does not help here

- Machine learning models
* Talk to Business. Define the problem
* Collect the data
* Select the metric
* Make train and test split
* Create the model
* Move model to the production
'''
