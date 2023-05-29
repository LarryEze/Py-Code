''' Exploring High Dimensional Data '''

'''
Introduction
- Dimensionality is the number of columns of data which is basically the attributes of data like name, age, sex and so on
- Data quality meets six dimensions: accuracy, completeness, consistency, timeliness, validity, and uniqueness.
- Dataset with many columns (e.g > 10) are considered high dimensional

Tidy data
pokemon_df -> in

    Name     Type   HP  Attack  Defense Speed   Generation
Bulbasaur   Grass   45  49      49      45      1
Ivysaur     Grass   60  62      63      60      1
Venusaur    Grass   80  82      83      80      1
Charmander   Fire   39  52      43      65      1
Charmeleon   Fire   58  64      58      80      1 -> out

The shape attribute
pokemon_df.shape -> in

(5, 7) -> out

The describe method
pokemon_df.describe() -> in

        HP   Attack Defense Speed Generation
count   5.0     5.0     5.0   5.0        5.0
mean   56.4    61.8    59.2  66.0        1.0
std    15.9    13.0    15.4  14.7        0.0
min    39.0    49.0    43.0  45.0        1.0
25%    45.0    52.0    49.0  60.0        1.0
50%    58.0    62.0    58.0  65.0        1.0
75%    60.0    64.0    63.0  80.0        1.0
max    80.0    82.0    83.0  80.0        1.0 -> out

pokemon_df.describe(exclude='number') -> in

                Name    Type 
count     5             5
unique    5             2
top       Charmander    Grass
freq      1             3-> out
'''

# Leave this list as is
number_cols = ['HP', 'Attack', 'Defense']

# Remove the feature without variance from this list
non_number_cols = ['Name', 'Type']

# Create a new DataFrame by subselecting the chosen features
df_selected = pokemon_df[number_cols + non_number_cols]

# Prints the first 5 lines of the new DataFrame
print(df_selected.head())


'''
Feature selection vs. feature extraction
Why reduce dimensionality?
Your dataset will:
- be less complex and simpler / easier to work with
- require less disk space to store the data
- require less computation time and run faster
- have lower chance of model overfitting

Feature selection
insurance_df -> in

income age favorite color
10000   18          Black
50000   47           Blue
20000   40           Blue
30000   29          Green
20000   22         Purple -> out

insurance_df.drop('favorite color', axis=1) -> in

income age 
10000   18 
50000   47 
20000   40 
30000   29 
20000   22  -> out

* axis=1 = column
* axis=0 = row

Building a pairplot on ANSUR data
sns.pairplot(ansur_df, hue='gender', diag_kind='hist')

- when we apply feature selection, we completely remove a feature and the information it holds from the datset.
- Feature selection is the process by which a subset of relevant features, or variables, are selected from a larger data set for constructing models.
- Feature extraction is a process of dimensionality reduction by which an initial set of raw data is reduced to more manageable groups for processing.
- PCA or Principal Component Analysis helps reduce dimensionality while keeping the variance of the data
'''

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_1, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Remove one of the redundant features
reduced_df = ansur_df_1.drop('stature_m', axis=1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue='Gender')

# Show the plot
plt.show()


# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(ansur_df_2, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()

# Remove the redundant feature
reduced_df = ansur_df_2.drop('n_legs', axis=1)

# Create a pairplot and color the points using the 'Gender' feature
sns.pairplot(reduced_df, hue='Gender', diag_kind='hist')

# Show the plot
plt.show()


'''
t-SNE visualization of high-dimensional data
- t-SNE or t-Distributed Stochastic Neighbor Embedding is a powerful technique to visualize high dimensional data using feature extraction.

t-SNE on female ANSUR dataset
df.shape -> in

(1986, 99) -> out

non-numeric = ['BMI_class', 'Height_class', 'Gender', 'Component', 'Branch']
df_numeric = df.drop(non_numeric, axis=1)
df_numeric.shape -> in

(1986, 94) -> out

- t-SNE does not work with non-numeric data

Fitting t-SNE
from sklearn.manifold import TSNE

m = TSNE(learning_rate = 50)
tsne_features = m.fit_transform(df_numeric)
tsne_features[1:4, :] -> in

array([[-37.962185, 15.066088], [-21.873512, 26.334448], [13.97476, 22.590828]], dtype=float32) -> out

df['x'] = tsne_features[:, 0]
df['y'] = tsne_features[:, 1]

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='x', y='y', hue='BMI_class', data=df)
sns.scatterplot(x='x', y='y', hue='Height_class', data=df)
plt.show()
'''

# Non-numerical columns in the dataset
non_numeric = ['Branch', 'Gender', 'Component']

# Drop the non-numerical columns from df
df_numeric = df.drop(non_numeric, axis=1)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

# Fit and transform the t-SNE model on the numeric dataset
tsne_features = m.fit_transform(df_numeric)
print(tsne_features.shape)


# Color the points according to Army Component
sns.scatterplot(x="x", y="y", hue='Component', data=df)

# Color the points by Army Branch
sns.scatterplot(x="x", y="y", hue='Branch', data=df)

# Color the points by Gender
sns.scatterplot(x="x", y="y", hue='Gender', data=df)

# Show the plot
plt.show()


''' Feature Selection I - Selecting for Feature Information '''

'''
The curse of dimensionality
house_df -> in

City    Price
Berlin  2.0
Berlin  3.1
Berlin  4.3
Paris   3.0
Paris   5.2
...     ... -> out

Building a city classifier - data split
Separate the feature we want to predict from the ones to train the model on.

y = house_df['City']
x = house_df.drop('City', axis=1)

Perform a 70% train and 30% test data split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3)

Building a city classifier - model fit
Create a Support Vector Machine Classifier and fit to training data

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

Building a city classifier - predict
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, svc.predict(X_test))) -> in

0.826 -> out

print(accuracy_score(y_train, svc.predict(X_train))) -> in

0.832 -> out

Adding features
City    Price
Berlin  2.0
Berlin  3.1
Berlin  4.3
Paris   3.0
Paris   5.2
...     ... 

City    Price   n_floors  n_bathroom    surface_m2
Berlin  2.0     1         1             190
Berlin  3.1     2         1             187
Berlin  4.3     2         2             240
Paris   3.0     2         1             170
Paris   5.2     2         2             290
...     ...     ...       ...           ...

-NB*: to avoid overfitting, the number of observations should increase exponentially with the number of features.
'''

# Import train_test_split()

# Select the Gender column as the feature to be predicted (y)
y = ansur_df['Gender']

# Remove the Gender column to create the training data
X = ansur_df.drop('Gender', axis=1)

# Perform a 70% train and 30% test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(
    f"{X_test.shape[0]} rows in test set vs. {X_train.shape[0]} in training set, {X_test.shape[1]} Features.")


# Import SVC from sklearn.svm and accuracy_score from sklearn.metrics

# Create an instance of the Support Vector Classification class
svc = SVC()

# Fit the model to the training data
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print(f"{accuracy_test:.1%} accuracy on test set vs. {accuracy_train:.1%} on training set")


# Assign just the 'neckcircumferencebase' column from ansur_df to X
X = ansur_df[['neckcircumferencebase']]

# Split the data, instantiate a classifier and fit the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC()
svc.fit(X_train, y_train)

# Calculate accuracy scores on both train and test data
accuracy_train = accuracy_score(y_train, svc.predict(X_train))
accuracy_test = accuracy_score(y_test, svc.predict(X_test))

print(f"{accuracy_test:.1%} accuracy on test set vs. {accuracy_train:.1%} on training set")


'''
Features with missing values or little variance
Creating a feature selector
print(ansur_df.shape) -> in

(6068, 94) -> in

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold = 1)
sel.fit(ansur_df)

mask = sel.get_support()
print(mask) -> in

array([ True, True, ..., False, True]) -> out

Applying a feature selector
print(ansur_df.shape) -> in

(6068, 94) -> in

reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape) -> in

(6068, 93) -> out

Normalizing the variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold = 0.005)
sel.fit(ansur_df / ansur_df.mean())
mask = sel.get_support()
reduced_df = ansur_df.loc[:, mask]
print(reduced_df.shape) -> in

(6068, 45) -> out

Missing value selector
Name        Type1   Type2 Total  HP Attack Defense
Bulbasaur   Grass  Poison   318  45     49      49
Ivysaur     Grass  Poison   405  60     62      63
Venusaur    Grass  Poison   525  80     82      83
Charmander   Fire     NaN   309  39     52      43
Charmeleon   Fire     NaN   405  58     64      58

Identifying missing values
pokemon_df.isna() -> in 

Name    Type1   Type2   Total      HP Attack Defense
False   False   False   False   False  False   False
False   False   False   False   False  False   False
False   False   False   False   False  False   False
False   False    True   False   False  False   False
False   False    True   False   False  False   False -> out

Counting missing values
pokemon_df.isna().sum() -> in 

Name     0
Type 1   0
Type 2 386
Total    0
HP       0
Attack   0
Defense  0
dtype: int64 -> out

pokemon_df.isna().sum() / len(pokemon_df) -> in 

Name    0.00
Type 1  0.00
Type 2  0.48
Total   0.00
HP      0.00
Attack  0.00
Defense 0.00
dtype: float64 -> out

Applying a missing value threshold
# Fewer than 30% missing values = True value
mask = pokemon_df.isna().sum() / len(pokemon_df) < 0.3
print(mask) -> in 

Name    True
Type 1  True
Type 2 False
Total   True
HP      True
Attack  True
Defense True
dtype: bool -> out

reduced_df = pokemon_df.loc[:, mask]
reduced_df.head() -> in

Name        Type1 Total  HP Attack Defense
Bulbasaur   Grass   318  45     49      49
Ivysaur     Grass   405  60     62      63
Venusaur    Grass   525  80     82      83
Charmander   Fire   309  39     52      43
Charmeleon   Fire   405  58     64      58 -> out
'''

# Create the boxplot
head_df.boxplot()

# Normalize the data
normalized_df = head_df / head_df.mean()

# Print the variances of the normalized data
print(normalized_df.var())

plt.show()


# Create a VarianceThreshold feature selector
sel = VarianceThreshold(threshold=0.001)

# Fit the selector to normalized head_df
sel.fit(head_df / head_df.mean())

# Create a boolean mask
mask = sel.get_support()

# Apply the mask to create a reduced DataFrame
reduced_df = head_df.loc[:, mask]

print(
    f"Dimensionality reduced from {head_df.shape[1]} to {reduced_df.shape[1]}.")


# Create a boolean mask on whether each feature less than 50% missing values.
mask = school_df.isna().sum() / len(school_df) < 0.5

# Create a reduced dataset by applying the mask
reduced_df = school_df.loc[:, mask]

print(school_df.shape)
print(reduced_df.shape)


'''
Pairwise correlation
sns.pairplot(ansur, hue='gender')

Correlation coefficient
r = -1 (Perfect negative correlation)
r = 0 (No correlation)
r = 1 (Perfect positive correlation)

Correlation matrix
weights_df.corr() -> in

            weight_lbs weight_kg heigh_in
weight_lbs        1.00      1.00     0.47
weight_kg         1.00      1.00     0.47
heigh_in          0.47      0.47     1.00 -> out

Visualizing the correkation matrix
cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
sns.heatmap(weights_df.corr(), center=0, cmap=cmap, linewidths=1, annot=True, fmt='.2f')

corr = weights_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool)) -> in

array([[True, True, True], [False, True, True], [False, False, True]]) -> out

sns.heatmap(weights_df.corr(), mask=mask, center=0, cmap=cmap, linewidths=1, annot=True, fmt='.2f')

- np.ones_like() creates a matrix filled with True values with the same dimensions as the correlation matrix
- np.triu() for triangle upper, function to set all non-upper triangle values to False
'''

# Create the correlation matrix
corr = ansur_df.corr()

# Draw a heatmap of the correlation matrix
sns.heatmap(corr,  cmap=cmap, center=0, linewidths=1, annot=True, fmt=".2f")
plt.show()


# Create the correlation matrix
corr = ansur_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Add the mask to the heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            linewidths=1, annot=True, fmt=".2f")
plt.show()


'''
Removing highly correlated features
# Create positive correlation matrix
corr_df = chest_df.corr().abs()

# Create and apply mask
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)
tri_df -> in

                        Suprasternale height Cervicale height Chest height
Suprasternale height                     NaN              NaN          NaN 
Cervicale height                    0.983033              NaN          NaN
Chest height                        0.956111         0.951101          NaN -> out

# Find columns that meet threshold
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]
print(to_drop) -> in

['Suprasternale height', 'Cervicale height'] -> out

# Drop those columns
reduced_df = chest_df.drop(to_drop, axis=1)

Correlation caveats - Anscombe's quartet (i.e the same correlation coefficients for linear and non-linear relationship which may also include outliers)

- NB*: Strong correlations do not imply causation.
'''

# Calculate the correlation matrix and take the absolute value
corr_df = ansur_df.corr().abs()

# Create a True/False mask and apply it
mask = np.triu(np.ones_like(corr_df, dtype=bool))
tri_df = corr_df.mask(mask)

# List column names of highly correlated features (r > 0.95)
to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.95)]

# Drop the features in the to_drop list
reduced_df = ansur_df.drop(to_drop, axis=1)

print(f"The reduced_df DataFrame has {reduced_df.shape[1]} columns.")


# Print the first five lines of weird_df
print(weird_df.head())

# Put nuclear energy production on the x-axis and the number of pool drownings on the y-axis
sns.scatterplot(x='nuclear_energy', y='pool_drownings', data=weird_df)
plt.show()

# Print out the correlation matrix of weird_df
print(weird_df.corr())


''' Feature Selection II - Selecting for Model Accuracy '''

'''
Selecting features for model performance
Ansur dataset sample
Gender chestdepth handlength neckcircumference shoulderlength earlength
Female        243        176               326            136        62
Female        219        177               325            135        58
Male          259        193               400            145        71
Male          253        195               380            141        62

Pre-processing the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

Creating a logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(X_train_std, y_train)

X_test_std = scaler.transform(X_test)

y_pred = lr.predict(X_test_std)
print(accuracy_score(y_test, y_pred)) -> in

0.99 -> out

Inspecting the feature coefficients
print(lr.coef_) -> in

array([[-3, 0.14, 7.46, 1.22, 0.87]]) -> out

print(dict(zip(X.columns, abs(lr.coef_[0])))) -> in

{'chestdepth': 3.0, 'handlength': 0.14, 'neckcircumference': 7.46, 'shoulderlength': 1.22, 'earlength': 0.87} -> out

Features that contribute little to a model
X.drop('handlength', axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lr.fit(scaler.fit_transform(X_train), y_train)
print(accuracy_score(y_test, lr.predict(scaler.transform(X_test)))) -> in

0.99 -> out

Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2, verbose=1)
rfe.fit(X_train_std, y_train) -> in

Fitting estimator with 5 features
Fitting estimator with 4 features
Fitting estimator with 3 features -> out

- Recursive method is safer, since dropping one feature will cause the other coefficients to change

Inspecting the RFE results
X.columns[rfe.support_] -> in

Index(['chestdepth', 'neckcircumference'], dtype='object') -> out

print(dict(zip(X.columns, rfe.ranking_))) -> in

{'chestdepth': 1, 'handlength': 4, 'neckcircumference': 1, 'shoulderlength': 2, 'earlength': 3} -> out

print(accuracy_score(y_test, rfe.predict(X_test_std))) -> in

0.99 -> out
'''

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train, y_train)

# Fit the logistic regression model on the scaled training data
lr.fit(X_train_std, y_train)

# Scale the test features
X_test_std = scaler.transform(X_test)

# Predict diabetes presence on the scaled test set
y_pred = lr.predict(X_test_std)

# Prints accuracy metrics and feature coefficients
print(f"{accuracy_score(y_test, y_pred):.1%} accuracy on test set.")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# Remove the feature with the lowest model coefficient
X = diabetes_df[['pregnant', 'glucose',
                 'triceps', 'insulin', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set.")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Remove the 2 features with the lowest model coefficients
X = diabetes_df[['glucose', 'triceps', 'bmi', 'family', 'age']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set.")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))

# Only keep the feature with the highest coefficient
X = diabetes_df[['glucose']]

# Performs a 25-75% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Scales features and fits the logistic regression model to the data
lr.fit(scaler.fit_transform(X_train), y_train)

# Calculates the accuracy on the test set and prints coefficients
acc = accuracy_score(y_test, lr.predict(scaler.transform(X_test)))
print(f"{acc:.1%} accuracy on test set.")
print(dict(zip(X.columns, abs(lr.coef_[0]).round(2))))


# Create the RFE with a LogisticRegression estimator and 3 features to select
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=3, verbose=1)

# Fits the eliminator to the data
rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)
print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated
print(X.columns[rfe.support_])

# Calculates the test set accuracy
acc = accuracy_score(y_test, rfe.predict(X_test))
print(f"{acc:.1%} accuracy on test set.")


'''
Tree-based feature selection
Random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(y_test, rf.predict(X_test))) -> in

0.99 -> out

Feature importance values
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(rf.feature_importances_) -> in

array([ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.04 , 0. , 0.01 , 0.01,
        0. , 0. , 0. , 0. , 0.01 , 0.01 , 0. , 0. , 0. , 0. , 0.05,
        ...
        0. , 0.14 , 0. , 0. , 0. , 0.06 , 0. , 0. , 0. , 0. , 0. ,
        0. , 0.07, 0. , 0. , 0.01, 0. ]) -> out

print(sum(rf.feature_importances_)) -> in

1.0 -> out

Feature importace as a feature selector
mask = rf.feature_importances_ > 0.1
print(mask) -> in

array([False, False, ..., True, False]) -> in

X_reduced = X.loc[:, mask]
print(X_reduced.columns) -> in

Index(['chestheight', 'neckcircumference', 'neckcircumferencebase', 'shouldercircumference'], dtype='object') -> out

RFE with random forests
from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=6, verbose=1)
rfe.fit(X_train, y_train)

Fitting estimator with 94 features
Fitting estimator with 93 features
...
Fitting estimator with 8 features
Fitting estimator with 7 features -> out

print(accuracy_score(y_test, rfe.predict(X_test)) -> in

0.99 -> out

from sklearn.feature_selection import RFE
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=6, step=10, verbose=1)
rfe.fit(X_train, y_train)

Fitting estimator with 94 features
Fitting estimator with 84 features
...
Fitting estimator with 24 features
Fitting estimator with 14 features -> out

print(X.columns[rfe.support_]) -> in

Index(['biacromiabreadth', 'handbreadth', 'handcircumference', 'neckcircumference', 'neckcircumferencebase', 'shouldercircumference'], dtype='object') -> out
'''

# Perform a 75% training and 25% test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Fit the random forest model to the training data
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Calculate the accuracy
acc = accuracy_score(y_test, rf.predict(X_test))

# Print the importances per feature
print(dict(zip(X.columns, rf.feature_importances_.round(2))))

# Print accuracy
print(f"{acc:.1%} accuracy on test set.")


# Create a mask for features importances above the threshold
mask = rf.feature_importances_ > 0.15

# Apply the mask to the feature dataset X
reduced_X = X.loc[:, mask]

# prints out the selected column names
print(reduced_X.columns)


# Set the feature eliminator to remove 2 features on each step
rfe = RFE(estimator=RandomForestClassifier(),
          n_features_to_select=2, step=2, verbose=1)

# Fit the model to the training data
rfe.fit(X_train, y_train)

# Create a mask
mask = rfe.support_

# Apply the mask to the feature dataset X and print the result
reduced_X = X.loc[:, mask]
print(reduced_X.columns)


'''
Regularized linear regression
Creating our own dataset
x1        x2      x3
1.76   -0.37   -0.60
0.40   -0.24   -1.12
0.98    1.10    0.77
...     ...     ...

Creating our own target feature:
y = 20 + 5x1 + 2x2 + 0x3 + error

Linear regression in Python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Actual coefficients = [5 2 0]
print(lr.coef_) -> in

[ 4.95 1.83 -0.05] -> out

# Actual intercept = 20
print(lr.intercept_) -> in

19.8 -> out

# Calculates R-squared
print(lr.score(X_test, y_test)) -> in

0.976 -> out

- Loss function: Mean Squared Error tries to make the model accurate
- Regularization tries to mae the model simple
- Lasso or Least Absolute Shrinkage and Selection Operator is a linear model with regularization

Lasso regressor
from sklearn.linear_model import Lasso
la = Lasso()
la.fit(X_train, y_train)

# Actual coefficients = [5 2 0]
print(la.coef_) -> in

[ 4.07 0.59 0. ] -> out

print(la.score(X_test, y_test)) -> in

0.861 -> out

la = Lasso(alpha = 0.05)
la.fit(X_train, y_train)

# Actual coefficients = [5 2 0]
print(la.coef_) -> in

[ 4.91 1.76 0. ] -> out

print(la.score(X_test, y_test)) -> in

0.974 -> out
'''

# Set the test size to 30% to get a 70-30% train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Fit the scaler on the training features and transform these in one go
X_train_std = scaler.fit_transform(X_train, y_train)

# Create the Lasso model
la = Lasso()

# Fit it to the standardized training data
la.fit(X_train_std, y_train)


# Transform the test set with the pre-fitted scaler
X_test_std = scaler.transform(X_test)

# Calculate the coefficient of determination (R squared) on X_test_std
r_squared = la.score(X_test_std, y_test)
print(
    f"The model can predict {r_squared:.1%} of the variance in the test set.")

# Create a list that has True values when coefficients equal 0
zero_coef = la.coef_ == 0

# Calculate how many features have a zero coefficient
n_ignored = sum(zero_coef)
print(f"The model has ignored {n_ignored} out of {len(la.coef_)} features.")


# Find the highest alpha value with R-squared above 98%
la = Lasso(alpha=0.1, random_state=0)

# Fits the model and calculates performance stats
la.fit(X_train_std, y_train)
r_squared = la.score(X_test_std, y_test)
n_ignored_features = sum(la.coef_ == 0)

# Print peformance stats
print(
    f"The model can predict {r_squared:.1%} of the variance in the test set.")
print(f"{n_ignored_features} out of {len(la.coef_)} features were ignored.")


'''
Combining feature selectors
Lasso regressor
from sklearn.linear_model import Lasso
la = Lasso(alpha = 0.05)
la.fit(X_train, y_train)

# Actual coefficients = [5 2 0]
print(la.coef_) -> in

[ 4.91 1.76 0. ] -> out

print(la.score(X_test, y_test)) -> in

0.974 -> out

LassoCV regressor
from sklearn.linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train, y_train)
print(lcv.alpha_) -> in

0.09 -> out

mask = lcv.coef_ != 0
print(mask) -> in

[ True True False ] -> out

reduced_X = X.loc[:, mask]

Taking a step back
- Random forest is a combination of decision trees
- We can use combination of models for feature selction too

Feature selection with LassoCV
from sklearn.linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train, y_train)
lcv.score(X_test, y_test) -> in

0.99 -> out

lcv_mask = lcv.coef_ != 0
sum(lcv_mask) -> in

66 -> out

Feature selection with random forest
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
rfe_rf = RFE(estimator=RandomForestRegressor(), n_features_to_select=66, step=5, verbose=1)
rfe_rf.fit(X_train, y_train)
rf_mask = rfe_rf.support_

Feature selection with gradient boosting
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
rfe_gb = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=66, step=5, verbose=1)
rfe_gb.fit(X_train, y_train)
gb_mask = rfe_gb.support_

Combining the feature selectors
import numpy as np
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)
print(votes) -> in

array([3, 2, 2, ..., 3, 0, 1]) -> out

mask = votes >= 2
reduced_X = X.loc[:, mask]
'''


# Create and fit the LassoCV model on the training set
lcv = LassoCV()
lcv.fit(X_train, y_train)
print(f'Optimal alpha = {lcv.alpha_:.3f}')

# Calculate R squared on the test set
r_squared = lcv.score(X_test, y_test)
print(f'The model explains {r_squared:.1%} of the test set variance')

# Create a mask for coefficients not equal to zero
lcv_mask = lcv.coef_ != 0
print(f'{sum(lcv_mask)} features out of {len(lcv_mask)} selected')


# Select 10 features with RFE on a GradientBoostingRegressor, drop 3 features on each step
rfe_gb = RFE(estimator=GradientBoostingRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_gb.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_gb.score(X_test, y_test)
print(f'The model can explain {r_squared:.1%} of the variance in the test set')

# Assign the support array to gb_mask
gb_mask = rfe_gb.support_


# Select 10 features with RFE on a RandomForestRegressor, drop 3 features on each step
rfe_rf = RFE(estimator=RandomForestRegressor(),
             n_features_to_select=10, step=3, verbose=1)
rfe_rf.fit(X_train, y_train)

# Calculate the R squared on the test set
r_squared = rfe_rf.score(X_test, y_test)
print(f'The model can explain {r_squared:.1%} of the variance in the test set')

# Assign the support array to rf_mask
rf_mask = rfe_rf.support_


# Sum the votes of the three models
votes = np.sum([lcv_mask, rf_mask, gb_mask], axis=0)

# Create a mask for features selected by all 3 models
meta_mask = votes == 3

# Apply the dimensionality reduction on X
X_reduced = X.loc[:, meta_mask]

# Plug the reduced dataset into a linear regression pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.3, random_state=0)
lm.fit(scaler.fit_transform(X_train), y_train)
r_squared = lm.score(scaler.transform(X_test), y_test)
print(
    f'The model can explain {r_squared:.1%} of the variance in the test set using {len(lm.coef_)} features.')


''' Feature Extraction '''

'''
Feature extraction
Feature generation - BMI
df_body['BMI'] = df_body['Weight kg'] / df_body['Height m'] ** 2 -> in

Weight kg   Height m    BMI
81.5           1.776  25.84
72.6           1.702  25.06
92.9           1.735  30.86 -> out

df_body.drop(['Weight kg', 'Height m'], axis=1) -> in

BMI
25.84
25.06
30.86 -> out

Feature generation - averages
left leg mm     right leg mm
882             885
870             869
901             900

leg_df['leg mm'] = leg_df[['left leg mm', 'right leg mm']].mean(axis = 1)

leg_df.drop(['left leg mm', 'right leg mm'], axis=1)

leg mm
883.5
869.5
900.5

Intro to PCA
sns.scatterplot(data=df, x='handlength', y='footlength')

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
'''

# Calculate the price from the quantity sold and revenue
sales_df['price'] = sales_df['revenue'] / sales_df['quantity']

# Drop the quantity and revenue features
reduced_df = sales_df.drop(['quantity', 'revenue'], axis=1)

print(reduced_df.head())


# Calculate the mean height
height_df['height'] = height_df[[
    'height_1', 'height_2', 'height_3']].mean(axis=1)

# Drop the 3 original height features
reduced_df = height_df.drop(['height_1', 'height_2', 'height_3'], axis=1)

print(reduced_df.head())


'''
Principal component analysis
Calculating the principal components
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_df = scaler.fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA()
print(pca.fit_transform(std_df)) -> in

[   [-0.08320426 -0.12242952]
    [ 0.31478004   0.57048158]
    ...
    [-0.5609523     0.13713944]
    [-0.0448304    -0.37898246] ] -> out

Principal component explained variance ratio
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(std_df)) 
print(pca.explained_variance_ratio_) -> in

array([0.90, 0.10]) -> out

PCA for dimensionality reduction
pca = PCA()
pca.fit(ansur_std_df)
print(pca.explained_variance_ratio.cumsum()) -> in

array([ 0.44, 0.62, 0.66, 0.69, 0.72, 0.74, 0.76, 0.77, 0.79, 0.8 , 0.81,
        0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.89, 0.89, 0.9 ,
        0.9 , 0.91, 0.92, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.94, 0,95,
        ...
        0.99, 0.99, 0.99, 0.99, 0.99, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
        1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
        1.  , 1.  , 1.  , 1.  , 1.  , 1.  ]) -> out
'''

# Create a pairplot to inspect ansur_df
sns.pairplot(ansur_df)

plt.show()


# Create the scaler
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Create the PCA instance and fit and transform the data with pca
pca = PCA()
pc = pca.fit_transform(ansur_std)
pc_df = pd.DataFrame(pc, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4'])

# Create a pairplot of the principal component DataFrame
sns.pairplot(pc_df)
plt.show()


# Scale the data
scaler = StandardScaler()
ansur_std = scaler.fit_transform(ansur_df)

# Apply PCA
pca = PCA()
pca.fit(ansur_std)


# Inspect the explained variance ratio per component
print(pca.explained_variance_ratio_)

# Print the cumulative sum of the explained variance ratio
print(pca.explained_variance_ratio_.cumsum())


'''
PCA applications
Understanding the components
print(pca.components_) -> in

array([ [  0.71, 0.71],
        [ -0.71, 0.71]  ]) -> out

PC 1 = 0.71 x Hand length + 0.71 x Foot length
PC 2 = -0.71 x Hand length + 0.71 x Foot length

PCA in a pipeline
- Always scale the data before applying PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA())])
pc = pipe.fit_transform(ansur_df)
print(pc[:, :2]) -> in

array([ [-3.46114925 , 1.5785215 ],
        [ 0.908606615, 2.02379935],
        ...,
        [10.7569818 , -1.40222755],
        [ 7.64802025,  1.07406209]]) -> in

Checking the effect of categorical features
- PCA is not the preferred algorithm to reduce the dimensionality of categorical datasets.
* but we can check whether they align with the most important sources of variance in the data.

print(ansur_categories.head()) -> in

                    Branch     Component    Gender  BMI_class Height_class
0              Combat Arms  Regular Army      Male Overweight         Tall
1           Combat Support  Regular Army      Male Overweight       Normal
2           Combat Support  Regular Army      Male Overweight       Normal
3   Combat Service Support  Regular Army      Male Overweight       Normal
4   Combat Service Support  Regular Army      Male Overweight         Tall -> out

ansur_categories['PC 1'] = pc[:, 0]
ansur_categories['PC 2'] = pc[:, 1]
sns.scatterplot(data=ansur_categories, x='PC 1', y='PC 2', hue='Height_class', alpha=0.4)
sns.scatterplot(data=ansur_categories, x='PC 1', y='PC 2', hue='Gender', alpha=0.4)
sns.scatterplot(data=ansur_categories, x='PC 1', y='PC 2', hue='BMI_class', alpha=0.4)
plt.show()

PCA in a model pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=3)), ('classifier', RandomForestClassifier())])
print(pipe['reducer']) -> in

PCA(n_components=3) -> out

pipe.fit(X_train, y_train)
pipe['reducer'].explained_variance_ratio_ -> in

array([0.56, 0.13, 0.05]) -> out

pipe['reducer'].explained_variance_ratio_.sum() -> in

0.74 -> out

print(pipe.score(X_test, y_test)) -> in

0.986 -> out
'''

# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=2))])

# Fit it to the dataset and extract the component vectors
pipe.fit(poke_df)
vectors = pipe['reducer'].components_.round(2)

# Print feature effects
print('PC 1 effects = ' + str(dict(zip(poke_df.columns, vectors[0]))))
print('PC 2 effects = ' + str(dict(zip(poke_df.columns, vectors[1]))))


pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=2))])

# Fit the pipeline to poke_df and transform the data
pc = pipe.fit_transform(poke_df)

# Add the 2 components to poke_cat_df
poke_cat_df['PC 1'] = pc[:, 0]
poke_cat_df['PC 2'] = pc[:, 1]

# Use the Type feature to color the PC 1 vs. PC 2 scatterplot
sns.scatterplot(data=poke_cat_df, x='PC 1', y='PC 2', hue='Type')
plt.show()


pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=2))])

# Fit the pipeline to poke_df and transform the data
pc = pipe.fit_transform(poke_df)

# Add the 2 components to poke_cat_df
poke_cat_df['PC 1'] = pc[:, 0]
poke_cat_df['PC 2'] = pc[:, 1]

# Use the Legendary feature to color the PC 1 vs. PC 2 scatterplot
sns.scatterplot(data=poke_cat_df, x='PC 1', y='PC 2', hue='Legendary')
plt.show()


# Build the pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(
    n_components=3)), ('classifier', RandomForestClassifier(random_state=0))])

# Fit the pipeline to the training data
pipe.fit(X_train, y_train)

# Score the accuracy on the test set
accuracy = pipe.score(X_test, y_test)

# Prints the explained variance ratio and accuracy
print(pipe['reducer'].explained_variance_ratio_)
print(f'{accuracy:.1%} test set accuracy')


'''
Principal Component selection
pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=0.9))])
# Fit the pipe to the data
pipe.fit(poke_df)
print(len(pipe['reducer'].components_)) -> in

5 -> out

- n_components = 0.9 will make sure to select enough components to explain 90% of the variance.

An optimal number of components
pipe.fit(poke_df)
var = pipe['reducer'].explained_variance_ratio_
plt.plot(var)
plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()

- NB*: The x-axis shows the index of the components and not the total number

Compressing images
print(X_test.shape) -> in

(15, 2914) -> out

- 62 x 47 pixels = 2914 grayscale values

print(X_train.shape) -> in

(1333, 2914) -> out

pipe = Pipeline([('scaler', StandardScaler()), ('reducer', PCA(n_components=290))])
pipe.fit(X_train)
pc = pipe.fit_transform(X_test)
print(pc.shape) -> in

(15, 290) -> out

Rebuilding images
pc = pipe.transform(X_test)
print(pc.shape) -> in

(15, 290) -> out

X_rebuilt = pipe.inverse_transform(pc)
print(X_rebuilt.shape) -> in

(15, 2914) -> out

img_plotter(X_rebuilt) -> in 
'''

# Pipe a scaler to PCA selecting 80% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=0.8))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print(f'{len(pipe["reducer"].components_)} components selected')


# Let PCA select 90% of the variance
pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=0.9))])

# Fit the pipe to the data
pipe.fit(ansur_df)

print(f'{len(pipe["reducer"].components_)} components selected')


# Pipeline a scaler and pca selecting 10 components
pipe = Pipeline([('scaler', StandardScaler()),
                ('reducer', PCA(n_components=10))])

# Fit the pipe to the data
pipe.fit(ansur_df)

# Plot the explained variance ratio
plt.plot(pipe['reducer'].explained_variance_ratio_)

plt.xlabel('Principal component index')
plt.ylabel('Explained variance ratio')
plt.show()


# Plot the MNIST sample data
plot_digits(X_test)

# Transform the input data to principal components
pc = pipe.transform(X_test)

# Prints the number of features per dataset
print(f"X_test has {X_test.shape[1]} features")
print(f"pc has {pc.shape[1]} features")

# Inverse transform the components to original feature space
X_rebuilt = pipe.inverse_transform(pc)

# Prints the number of features
print(f"X_rebuilt has {X_rebuilt.shape[1]} features")

# Plot the reconstructed data
plot_digits(X_rebuilt)
