# import necessary packages
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


''' Introduction to Data Preprocessing '''

'''
Introduction to preprocessing
what is data preprocessing?
- It comes after exploratory data analysis and data cleaning
- Its used in preparing data for modeling
- Example: Transforming categorical features into numerical features (dummy variables)

Why preprocess?
- Transform dataset so it's suitable for modeling
- Improve model performance
- Generate more reliable results

Recap: exploring data with pandas
import pandas as pd
hiking = pd.read_json('hiking.json')
print(hiking.head()) -> in

    Prop_ID                     Name . . .  lat     lon
0   B057     Salt Marsh Nature Trail . . .  NaN     NaN
1   B073                   Lullwater . . .  NaN     NaN
2   B073                     Midwood . . .  NaN     NaN
3   B073                   Peninsula . . .  NaN     NaN
4   B073                   Waterfall . . .  NaN     NaN -> out

print(hiking.info()) -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 33 entries, 0 to 32
Data columns (total 11 columns):
#           Column  Non-Null Count  Dtype
--   -------------  --------------  ----- 
0          Prop_ID  33 non-null     object
1             Name  33 non-null     object
2         Location  33 non-null     object
3        Park_Name  33 non-null     object
4           Length  29 non-null     object
5       Difficulty  27 non-null     object
6    Other_Details  31 non-null     object
7       Accessible  33 non-null     object
8   Limited_Access  33 non-null     object
9              lat   0 non-null    float64
10             lon   0 non-null    float64
dtypes: float64(2), object(9)
memory usage: 3.0+ KB -> out

print(wine.describe()) -> in

            Type    Alcohol . . . Alcalinity of ash
count 178.000000 178.000000 . . .        178.000000
mean    1.938202  13.000618 . . .         19.494944
std     0.775035   0.811827 . . .          3.339564
min     1.000000  11.030000 . . .         10.600000
25%     1.000000  12.362500 . . .         17.200000
50%     2.000000  13.050000 . . .         19.500000
75%     3.000000  13.677500 . . .         21.500000
max     3.000000  14.830000 . . .         30.000000 -> out 

Removing missing data
print(df) -> in

    A   B   C
0 1.0 NaN 2.0
1 4.0 7.0 3.0
2 7.0 NaN NaN
3 NaN 7.0 NaN
4 5.0 9.0 7.0 -> out

print(df.dropna()) -> in

    A   B   C
1 4.0 7.0 3.0
4 5.0 9.0 7.0 -> out

print(df.drop([0, 2, 3])) -> in

    A   B   C
1 4.0 7.0 3.0
4 5.0 9.0 7.0 -> out

print(df.drop('A', axis=1)) -> in

    B   C
0 NaN 2.0
1 7.0 3.0
2 NaN NaN
3 7.0 NaN
4 9.0 7.0 -> out

print(df.isna().sum()) -> in

A 1
B 2
C 2
dtype: int64 -> out

print(df.dropna(subset=['B'])) -> in

    A   B   C
1 4.0 7.0 3.0
3 NaN 7.0 NaN
4 5.0 9.0 7.0 -> out

print(df.dropna(thresh=2)) -> in

    A   B   C
0 1.0 NaN 2.0
1 4.0 7.0 3.0
4 5.0 9.0 7.0 -> out
'''

volunteer = pd.read_csv(
    'Preprocessing for Machine Learning in Python/volunteer_opportunities.csv')

# Drop the Latitude and Longitude columns from volunteer
volunteer_cols = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Drop rows with missing category_desc values from volunteer_cols
volunteer_subset = volunteer_cols.dropna(subset=['category_desc'])

# Print out the shape of the subset
print(volunteer_subset.shape)


'''
Working with data types
Why are types important?
print(volunteer.info()) - > in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 665 entries, 0 to 664
Data columns (total 35 columns):
#           Column  Non-Null Count  Dtype
--  -------------- -------------    -----
0   opportunity_id  665 non-null    int64
1       content_id  665 non-null    int64
2     vol_requests  665 non-null    int64
3       event_time  665 non-null    int64
4            title  665 non-null   object
. . .        . . .         . . .    . . .
34             NTA    0 non-null  float64
dtypes: float64(13), int64(8), object(14)
memory usage: 182.0+ KB -> out

* object: string / mixed types
* int64: integer
* float64: float
* datetime64: dates and times

Converting column types
print(df) -> in

    A        B   C
0   1   string 1.0
1   2  string2 2.0
2   3  string3 3.0 -> out

print(df.info()) -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
#   Column Non-Null Count Dtype
--  ------ -------------- -----
0   A      3 non-null     int64
1   B      3 non-null    object
2   C      3 non-null     int64
dtypes: int64(1), object(2)
memory usage: 200.0+ bytes -> out

df['C'] = df['C'].astype('float')
print(df.dtypes) -> in

A     int64
B    object
C   float64
dtype: object -> out
'''

# Print the head of the hits column
print(volunteer["hits"].head())

# Convert the hits column to type int
volunteer["hits"] = volunteer["hits"].astype('int64')

# Look at the dtypes of the dataset
print(volunteer.dtypes)


'''
Training and test sets
why split the dataset?
- To reduce overfitting 
- Evaluate performance on a holdout set

Splitting up your dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

- Class imbalance is when the test and training sets are not representative samples of the dataset, which could bias the model being trained.

Stratified sampling
- It's a way of sampling that takes into account the distribution of classes in the dataset.
* Dataset of 100 samples: 80 class 1 and 20 class 2
* Training set of 75 samples: 60 class 1 and 15 class 2
* Test set of 25 samples: 20 class 1 and 5 class 2

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42) 
y['labels'].value_counts() -> in

class1 80
class2 20
Name: labels, dtype: int64 -> out

y_train['labels'].value_counts() -> in

class1 60
class2 15
Name: labels, dtype: int64 -> out

y_test['labels'].value_counts() -> in

class1 20
class2 5
Name: labels, dtype: int64 -> out
'''

volunteer = volunteer.dropna(subset=['category_id', 'category_desc'])

# Create a DataFrame with all columns except category_desc
X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the y dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

# Print the category_desc counts from y_train
print(y_train['category_desc'].value_counts())


''' Standardizing Data '''

'''
Standardization
- It is a preprocessing method used to transform continuous data to make it look normally distributed
- scikit-learn models assume that the training data is normally distributed
- Using non-normal training data can introduce bias to the model
- Data can be standardized by many ways e.g:
* Log normalization
* Feature scaling
- NB*: Standardization is a preprocessing method applied to continuous, numerical data.

When to standardize: linear distance
- Model in linear space
- Dataset features have a high variance, which is also related to distance metrics
- Examples include: 
* K-Nearest Neighbors (kNN)
* Linear regression
* K-Means Clustering

When to standardize: different scales
- When the features are on different scales
- Linearity assumptions
- Example: Predicting house prices using no. of bedrooms and last sale price
'''

wine = pd.read_csv(
    'Preprocessing for Machine Learning in Python\wine_types.csv')

X = wine[['Proline', 'Total phenols', 'Hue', 'Nonflavanoid phenols']]

y = wine['Type']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

knn = KNeighborsClassifier()

# Fit the knn model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


'''
Log normalization
what is log normalization?
- It is a method for standardizing data with features with high variance
- It applies a logarithm transformation to the values
- It captures the relative changes, the magnitude of the change in a linear model and also keeps everything in the positive space.
- It takes the natural log of each number using the constant e(approx 2.718)
- e.g e^3.4 = 30

Number  Log
30      3.4
300     5.7
3000    8

Log normalization in Python
print(df) -> in

    col1     col2
0   1.00      3.0
1   1.20     45.5
2   0.75     28.0
3   1.60    100.0 -> out

print(df.var()) -> in

col1       0.128958
col2    1691.729167
dtype: float64 -> out

import numpy as np
df['log_2'] = np.log(df['col2'])
print(df) -> in

    col1   col2       log_2
0   1.00    3.0     1.098612
1   1.20   45.5     3.817712
2   0.75   28.0     3.332205
3   1.60  100.0     4.605170 -> out

print(df[['col1', 'log_2']].var()) -> in

col1    0.128958
log_2   2.262886
dtype: float64 -> out
'''

# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())


'''
Scaling data for feature comparison
What is feature scaling?
- It is a method of standardization that's most useful when working with a dataset that contains continuous features that are on different scales
- Model with linear characteristics
- It transforms the features in the dataset so they have a mean of zero ( 0 ) and a variance of one ( 1 )
- Transforms to approximately normal distribution

How to scale data
print(df) -> in

    col1    col2    col3
0   1.00    48.0   100.0
1   1.20    45.5   101.3
2   0.75    46.2   103.5
3   1.60    50.0   104.0 -> out

print(df.var()) -> in

col1    0.128958
col2    4.055833
col3    3.526667
dtype: float64 -> out

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
print(df_scaled) -> in

        col1        col2        col3
0 -0.442127     0.329683   -1.352726
1  0.200967    -1.103723   -0.553388
2 -1.245995    -0.702369    0.799338
3  1.487156     1.476409    1.106776 -> out

print(df_scaled.var()) -> in

col1    1.333333
col2    1.333333
col3    1.333333
dtype: float64 -> out
'''

# Import StandardScaler

# Create the scaler
scaler = StandardScaler()

# Subset the DataFrame you want to scale
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to wine_subset
wine_subset_scaled = scaler.fit_transform(wine_subset)


'''
Standardized data and modeling
K-nearest neighbors
- It is a model that classifies data based on its distance to training set data
- i.e a new data point is assigned a label based on the class that the majority of surrounding data points belong to.
- Data Leakage : It is when non-training data is used to train the model

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
knn = KNeighborsClassifier()
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train) 
knn.score(X_test_scaled, y_test)

X - contains features
y - contains labels
'''

X = wine.drop('Type', axis=1)

y = wine['Type']

# Split the dataset and labels into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train, y_train)

# Score the model on the test data
print(knn.score(X_test, y_test))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

# Instantiate a StandardScaler
scaler = StandardScaler()

# Scale the training and test features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the k-nearest neighbors model to the training data
knn.fit(X_train_scaled, y_train)

# Score the model on the test data
print(knn.score(X_test_scaled, y_test))


''' Feature Engineering '''

'''
Feature Engineering
What is Feature Engineering
- It is the creation of new features based on existing features
- It adds information to the dataset that can improve prediction or clustering tasks, or adds insights into relationships between features
- It requires an in-depth knowledge of the dataset
- It is very dependent on the particular dataset being analyzed

Feature engineering scenarios
id                  Text
1             'Feature engineering is fun!'
2   'Feature engineering is a lot of work.'
3       'I don't mind feature engineering.'

user fav_color
1         blue
2        green
3       orange

id         Date
4       July 30 2011
5    January 29 1011
6   February 05 2011

user test1 test2 test3
1     90.5  89.6  91.4
2     65.5  70.6  67.3
3     78.1  80.7  81.8
'''


'''
Encoding categorical variables
print(users) -> in

    user subscribed     fav_color
0   1             y     blue
1   2             n     green
2   3             n     orange
3   4             y     green -> out

Encoding binary variables - pandas
print(users['subscribed']) -> in

0   y
1   n
2   n
3   y
Name: subscribed, dtype: object -> out

users['sub_enc'] = users['subscribed'].apply(lambda val: 1 if val == 'y' else 0)

print(users[['subscribed', 'sub_enc']]) -> in

    subscribed sub_enc
0            y       1
1            n       0
2            n       0
3            y       1 -> out

Encoding binary variables - scikit-learn
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
users['sub_enc_le'] = le.fit_transform(users['subscribed'])

print(users[['subscribed', 'sub_enc_le']]) -> in

    subscribed sub_enc
0            y       1
1            n       0
2            n       0
3            y       1 -> out

One-hot encoding
- One-hot encoding encodes categorical variables into 1s and 0s when there are more than two values to encode.

e.g 
print(users['fav_color']) -> in

0   blue
1   green
2   orange
3   green
Name: fav_color, dtype: object -> out

values: [blue, green, orange]
* blue: [1, 0, 0]
* green: [0, 1, 0]
* orange: [0, 0, 1]

print(pd.get_dummies(users['fav_color'])) -> in

    blue    green   orange
0   1       0       0
1   0       1       0
2   0       0       1
3   0       1       0 -> out
'''

hiking = pd.read_json(
    'Preprocessing for Machine Learning in Python\hiking.json')

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())


volunteer = pd.read_csv(
    'Preprocessing for Machine Learning in Python/volunteer_opportunities.csv')

# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])

# Take a look at the encoded columns
print(category_enc.head())


'''
Engineering numerical features
print(temps) -> in

        city    day1    day2    day3
0        NYC    68.3    67.9    67.8
1         SF    75.1    75.5    74.9
2         LA    80.3    84.0    81.3
3     Boston    63.0    61.0    61.2 -> out

temps['mean'] = temps.loc[ :, 'day1' : 'day3'].mean(axis=1)
print(temps) -> in

        city    day1    day2    day3    mean
0        NYC    68.3    67.9    67.8   68.00
1         SF    75.1    75.5    74.9   75.17
2         LA    80.3    84.0    81.3   81.87
3     Boston    63.0    61.0    61.2   61.73 -> out

Dates
print(purchases) -> in

                date    purchase
0       July 30 2011      $45.08
1   February 01 2011      $19.48
2    January 29 2011      $76.09
3      March 31 2012      $32.61
4   February 05 2011      $75.98 -> out

purchases['date_converted'] = pd.to_datetime(purchases['date'])
purchases['month'] = purchases['date_converted'].dt.month
print(purchases) -> in

            date    purchase    date_converted  month
0     July 30 2011    $45.08        2011-07-30      7
1 February 01 2011    $19.48        2011-02-01      2
2  January 29 2011    $76.09        2011-01-29      1
3    March 31 2012    $32.61        2012-03-31      3
4 February 05 2011    $75.98        2011-02-05      2 -> out
'''

name = ['Sue', 'Mark', 'Sean', 'Erin', 'Jenny', 'Russell']
run1 = [20.1, 16.5, 23.5, 21.7, 25.8, 30.9]
run2 = [18.5, 17.1, 25.1, 21.1, 27.1, 29.6]
run3 = [19.6, 16.9, 25.2, 20.9, 26.1, 31.4]
run4 = [20.3, 17.6, 24.6, 22.1, 26.7, 30.4]
run5 = [18.3, 17.3, 23.9, 22.2, 26.9, 29.9]

running_times_5k = pd.DataFrame(
    {'name': name, 'run1': run1, 'run2': run2, 'run3': run3, 'run4': run4, 'run5': run5})

# Use .loc to create a mean column
running_times_5k["mean"] = running_times_5k.loc[:, 'run1':'run5'].mean(axis=1)

# Take a look at the results
print(running_times_5k.head())


# First, convert string column to date column
volunteer["start_date_converted"] = pd.to_datetime(
    volunteer['start_date_date'])

# Extract just the month from the converted column
volunteer["start_date_month"] = volunteer['start_date_converted'].dt.month

# Take a look at the converted and new month columns
print(volunteer[['start_date_converted', 'start_date_month']].head())


'''
Engineering text features
Extraction
- Regular expressions: They are patterns that can be used to extract information from text data

import re
my_string = 'temperature: 75.6 F'
temp = re.search('\d+\.\d+', my_string)
print(float(temp.group(0))) -> in

75.6 -> out

Vectorizing text
TF/IDF: It is a way of vectorizing text that reflects how important a word is in a document beyond how frequently it occurs.
* TF = Term Frequency
* IDF = Inverse Document Frequency

from sklearn.feature_extraction.text import TfidfVectorizer

print(documents.head()) -> in

0   Building on successful events last summer and . . . 
1                Build a website for an Afghan business
2   Please join us and the students from Mott Hall. . .
3  The Oxfam Action Corps is a group of dedicated . . . 
4   stop 'N' swap reduces NYC's waste by finding n. . . -> out

tfidf_vec = TfidfVectorizer()
text_tfidf = tfidf_vec.fit_transform(documents)

Text classification
- Naive Bayes treats each feature as independent from the others.

P(A|B) = P(B|A)P(A) / P(B)
'''

hiking = pd.read_json(
    'Preprocessing for Machine Learning in Python\hiking.json')

# Write a pattern to extract numbers and decimals


def return_mileage(length):

    # Search the text for matches
    mile = re.search('\d+\.\d+', length)

    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))


hiking = hiking.dropna(subset=['Length'])

# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking['Length'].apply(return_mileage)
print(hiking[["Length", "Length_num"]].head())


# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)


volunteer = volunteer[['category_desc', 'title']]
volunteer = volunteer.dropna(subset=['category_desc'])

# Take the title text
title_text = volunteer["title"]

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]

X_train, X_test, y_train, y_test = train_test_split(
    text_tfidf.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb = GaussianNB()
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


''' Selecting Features for Modeling '''

'''
Feature selection
What is feature selection?
- It is a method of selecting features from the feature set to be used for modeling.
- It doesn't create new features
- It improves model's performance

When to select features
city         state        lat        long
hico            tx  31.982778  -98.033333
mackinaw city   mi  45.783889  -84.727778
winchester      ky  37.990000  -84.179722

- To reduce noise
- Features are strongly statistically correlated
- To reduce overall variance
'''


''' 
Removing redundant features
Redundant features
- Remove noisy features
- Remove correlated features
- Remove duplicated features

Correlated features
- Statistically correlated: i.e features move together directionally
- Linear models assume feature independence
- Pearson's correlation coefficient: It is a measure of its directionality
* score closer to 1 = strongly positively correlated
* score closer to 0 = not correlated
* score closer to -1 = strongly negatively correlated

print(df) -> in 

        A       B       C
0    3.06    3.92    1.04
1    2.76    3.40    1.05
2    3.24    3.17    1.03
. . . -> out

print(df.corr()) -> in

            A           B           C
A    1.000000    0.787194    0.543479
B    0.787194    1.000000    0.565468
C    0.543479    0.565468    1.000000 -> out
'''

columns = ['vol_requests', 'title', 'hits', 'category_desc', 'locality', 'region', 'postalcode', 'created_date', 'vol_requests_lognorm',
           'created_month', 'Education', 'Emergency Preparedness', 'Environment', 'Health', 'Helping Neighbors in Need', 'Strengthening Communities']

volunteer = pd.read_csv(
    'Preprocessing for Machine Learning in Python/volunteer_opportunities.csv')

volunteer = volunteer.dropna(subset=['category_desc'])

volunteer['vol_requests_lognorm'] = np.log(volunteer['vol_requests'])

volunteer['created_date'] = pd.to_datetime(volunteer['created_date'])
volunteer['created_month'] = volunteer['created_date'].dt.month

category_enc = pd.get_dummies(volunteer["category_desc"])

volunteer = pd.concat([volunteer, category_enc], axis=1)

volunteer = volunteer[columns]

# Create a list of redundant column names to drop
to_drop = ['locality', 'region', 'created_date',
           "vol_requests", 'category_desc']

# Drop those columns from the dataset
volunteer_subset = volunteer.drop(to_drop, axis=1)

# Print out the head of volunteer_subset
print(volunteer_subset.head())


columns = ['Flavanoids', 'Total phenols', 'Malic acid',
           'OD280/OD315 of diluted wines', 'Hue']

wine = wine[columns]

# Print out the column correlations of the wine dataset
print(wine.corr())

# Drop that column from the DataFrame
wine = wine.drop('Flavanoids', axis=1)

print(wine.head())


'''
Selecting features using text vectors
Looking at word weights
print(tfidf_vec.vocabulary_) -> in

{'200': 0, '204th': 1, '33rd': 2, 'ahead': 3, 'alley': 4, . . . -> out

print(text_tfidf[3].data) -> in

[0.19392702 0.20261085 . . .] -> out

print(text_tfidf[3].indices) -> in

[ 31 102 20 70 5 . . . ] -> out

vocab = {v:k for k, v in tfidf_vec.vocabulary_.items()} 
print(vocab) -> in

{0: '200', 1: '204th', 2: '33rd', 3: 'ahead', 4: 'alley', . . . -> out 

zipped_row = dict(zip(text_tfidf[3].indices, text_tfidf[3].data))
print(zipped_row) -> in

{5: 0.159788, 7: 0.265764, 8: 0.185999, 9: 0.265764, 10: 0.130773 ... -> out

def return_weights(vocab, vector, vector_index):

    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    return {vocab[i]:zipped[i] for i in vector[vector_index].indices}

print(return_weights(vocab, text_tfidf, 3)) -> in

{'and': 0.159788, 'are': 0.265764, 'at':0.185999, . . . -> out 
'''

# Add in the rest of the arguments


def return_weights(vocab, original_vocab, vector, vector_index, top_n):
    zipped = dict(zip(vector[vector_index].indices, vector[vector_index].data))

    # Transform that zipped dict into a series
    zipped_series = pd.Series({vocab[i]: zipped[i]
                              for i in vector[vector_index].indices})

    # Sort the series to pull out the top n weighted words
    zipped_index = zipped_series.sort_values(ascending=False)[:top_n].index
    return [original_vocab[i] for i in zipped_index]


vocab = {v: k for k, v in tfidf_vec.vocabulary_.items()}

# Print out the weighted words
print(return_weights(vocab, tfidf_vec.vocabulary_,
      text_tfidf, vector_index=8, top_n=3))


def words_to_filter(vocab, original_vocab, vector, top_n):
    filter_list = []
    for i in range(0, vector.shape[0]):

        # Call the return_weights function and extend filter_list
        filtered = return_weights(vocab, original_vocab, vector, i, top_n)
        filter_list.extend(filtered)

    # Return the list in a set, so we don't get duplicate word indices
    return set(filter_list)


# Call the function to get the list of word indices
filtered_words = words_to_filter(vocab, tfidf_vec.vocabulary_, text_tfidf, 3)

# Filter the columns in text_tfidf to only those in filtered_words
filtered_text = text_tfidf[:, list(filtered_words)]


# Split the dataset according to the class distribution of category_desc
X_train, X_test, y_train, y_test = train_test_split(
    filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))


'''
Dimensionality reduction
Dimensionality reduction and PCA
- Dimensionality reduction is a form of unsupervised learning that transforms our data in a way that shrinks the number of features in the feature space.
- It is a feature extraction method used to reduce the feature space

Principal Component Analysis (PCA)
- It uses a linear transformation to project features into a space where they are completely uncorrelated.
- it captures as much variance as possible by combining features into components

PCA in scikit-learn
from sklearn.decomposition import PCA
pca = PCA()
df_pca = pca.fit_transform(df)
print(df_pca) -> in

[88.4583, 18.7764, -2.2379, . . ., 0.0954, 0.0361, -0.0034],
[93.4564, 18.6709, -1.7887, . . ., -0.0509, 0.1331, 0.0119],
[-186.9433, -0.2133, -5.6307, . . ., 0.0332, 0.0271, 0.0055] -> out

print(pca.explained_variance_ratio_) -> in

[0.9981, 0.0017, 0.0001, . . .] -> out

PCA caveats
- It can be very difficult to interpret PCA components
- Its a good step at the end of the preprocessing journey
'''

wine = pd.read_csv(
    'Preprocessing for Machine Learning in Python\wine_types.csv')

# Instantiate a PCA object
pca = PCA()

# Define the features and labels from the wine dataset
X = wine.drop('Type', axis=1)
y = wine["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

# Apply PCA to the wine dataset X vector
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# Look at the percentage of variance explained by the different components
print(pca.explained_variance_ratio_)


# Fit knn to the training data
knn.fit(pca_X_train, y_train)

# Score knn on the test data and print it out
print(knn.score(pca_X_test, y_test))


''' Putting It All Together '''

''' UFOs and preprocessing '''

ufo = pd.read_csv(
    'Preprocessing for Machine Learning in Python/ufo_sightings_large.csv')

# Print the DataFrame info
print(ufo.info())

# Change the type of seconds to float
ufo["seconds"] = ufo["seconds"].astype('float')

# Change the date column to type datetime
ufo["date"] = pd.to_datetime(ufo["date"])

# Check the column types
print(ufo.info())


columns = ['date', 'state', 'type', 'length_of_time']
ufo = ufo[columns]

# Count the missing values in the length_of_time, state, and type columns, in that order
print(ufo[['length_of_time', 'state', 'type']].isna().sum())

# Drop rows where length_of_time, state, or type are missing
ufo_no_missing = ufo.dropna()

# Print out the shape of the new dataset
print(ufo_no_missing.shape)


''' Categorical variables and standardization '''

ufo = pd.read_csv(
    'Preprocessing for Machine Learning in Python/ufo_sightings_large.csv', parse_dates=['date'])

# Convert the column to numeric, setting non-convertible values to NaN
ufo['lat'] = pd.to_numeric(ufo['lat'], errors='coerce')

ufo = ufo[(ufo['seconds'] >= 60) &
          (ufo['seconds'] <= 6000) & (ufo['lat'] >= 0) & (ufo['lat'] <= 64.283) & (ufo['long'] <= 6.742)]

ufo = ufo.dropna()


def return_minutes(time_string):

    # Search for numbers in time_string
    num = re.search('\d+', time_string)
    if num is not None:
        return int(num.group(0))


# Apply the extraction to the length_of_time column
ufo["minutes"] = ufo["length_of_time"].apply(return_minutes)

# Take a look at the head of both of the columns
print(ufo[["length_of_time", "minutes"]].head())


# Check the variance of the seconds and minutes columns
print(ufo[['seconds', 'minutes']].var())

# Log normalize the seconds column
ufo["seconds_log"] = np.log(ufo['seconds'])

# Print out the variance of just the seconds_log column
print(ufo['seconds_log'].var())


''' Engineering new features '''

# Use pandas to encode us values as 1 and others as 0
ufo["country_enc"] = ufo["country"].apply(lambda val: 1 if val == 'us' else 0)

# Print the number of unique type values
print(len(ufo['type'].unique()))

# Create a one-hot encoded set of the type values
type_set = pd.get_dummies(ufo['type'])

# Concatenate this set back to the ufo DataFrame
ufo = pd.concat([ufo, type_set], axis=1)


# Look at the first 5 rows of the date column
print(ufo['date'].head())

# Extract the month from the date column
ufo["month"] = ufo["date"].dt.month

# Extract the year from the date column
ufo["year"] = ufo["date"].dt.year

# Take a look at the head of all three columns
print(ufo[['date', 'month', 'year']].head())


# Take a look at the head of the desc field
print(ufo['desc'].head())

# Instantiate the tfidf vectorizer object
vec = TfidfVectorizer()

# Fit and transform desc using vec
desc_tfidf = vec.fit_transform(ufo['desc'])

# Look at the number of columns and rows
print(desc_tfidf.shape)


''' Feature selection and modeling '''

# Make a list of features to drop
to_drop = ['date', 'city', 'state', 'country', 'length_of_time',
           'lat', 'long', 'recorded', 'seconds', 'minutes', 'desc']

# Drop those features
ufo_dropped = ufo.drop(to_drop, axis=1)

# Let's also filter some words out of the text vector we created
filtered_words = words_to_filter(vocab, vec.vocabulary_, desc_tfidf, 4)


# Take a look at the features in the X set of data
print(X.columns)

# Split the X and y sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)

# Fit knn to the training sets
knn.fit(X_train, y_train)

# Print the score of knn on the test sets
print(knn.score(X_test, y_test))


# Use the list of filtered words we created to filter the text vector
filtered_text = desc_tfidf[:, list(filtered_words)]

# Split the X and y sets using train_test_split, setting stratify=y
X_train, X_test, y_train, y_test = train_test_split(
    filtered_text.toarray(), y, stratify=y, random_state=42)

# Fit nb to the training sets
nb.fit(X_train, y_train)

# Print the score of nb on the test sets
print(nb.score(X_test, y_test))
