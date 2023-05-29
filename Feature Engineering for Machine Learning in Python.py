''' Creating Features '''

'''
Why generate features?
Feature Engineering
- This is the act of taking raw data and extracting features for ML from it that are suitable for the task
- Features refers to the the informations stored in the columns of the dataset

Different types of data
- Continuous: either integers (or whole numbers) or floats (decimals)
- Categorical: one of a limited set of values, e.g. gender, country of birth
- Ordinal: ranked values, often with no detail of dictance between them
- Boolean: True / False values
- Datetime: dates and times

Pandas
import pandas as pd
df = pd.read_csv(path_to_csv_file)
print(df.head()) -> in

Dataset
            SurveyDate                           FormalEducation
0  2018-02-28 20:20:00  Bachelor's degreen(BA. BS. B.Eng.. etc.)
1  2018-06-28 13:26:00  Bachelor's degreen(BA. BS. B.Eng.. etc.)
2  2018-06-06 03:37:00  Bachelor's degreen(BA. BS. B.Eng.. etc.)
3  2018-05-09 01:06:00  Some college / university study ...
4  2018-04-12 22:41:00  Bachelor's degreen(BA. BS. B.Eng.. etc.) -> out

Column names
print(df.columns) -> in

Index(['surveyDate', 'FormalEducation', 'ConvertedSalary', 'Hobby', 'Country', 'StackOverflowJobsRecommend', 'VersionControl', 'Age', 'Years Experience', 'Gender', 'RawSalary'], dtype='object') -> out

Column types
print(df.types) -> in

SurveyDate          object
FormalEducation     object
ConvertedSalary     float64
. . .
Years Experience    int64
Gender              object
RawSalary           object
dtype: object -> out

- In pandas, objects are columns that contain strings

Selectting specific data types
only_ints = df.select_dtypes(include=['int'])
print(only_ints.columns) -> in

Index(['Age', 'Years Experience'], dtype='object') -> out
'''

# Import pandas

# Import so_survey_csv into so_survey_df
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head())

# Print the data type of each column
print(so_survey_df.dtypes)


# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int', 'float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)


'''
Dealing with categorical features
- Categorical variables are used to represent groups that are qualitative in nature e.g colors, country of birth etc.

Encoding categorical features
Index   Country
1       'India'
2         'USA'
3          'UK'
4          'UK'
5      'France'
. . .     . . .


Index   C_India     C_USA   C_UK    C_France
1             1         0      0           0
2             0         1      0           0
3             0         0      1           0
4             0         0      1           0
5             0         0      0           1
. . .     . . .     . . .  . . .       . . . 

Encoding categorical features
- One-hot encoding
- Dummy encoding

- By default, pandas performs one-hot encoding when you use the get_dummies() function.

One-hot encoding
pd.get_dummies(df, columns=['Country'], prefix='C') -> in

    C_France    C_India     C_UK    C_USA
0          0          1        0        0
1          0          0        0        1
2          0          0        1        0
3          0          0        1        0
4          1          0        0        0 -> out

Dummy encoding
pd.get_dummies(df, columns=['Country'], drop_first=True, prefix='C') -> in

    C_India     C_UK    C_USA
0         1        0        0
1         0        0        1
2         0        1        0
3         0        1        0
4         0        0        0 -> out

- Dummy encoding creates n-1 features for n categories, omitting the first category.
- In dummy encoding, the base value, France in this case, is encoded by the absence of all other countries as you can see on the last row here and its value is represented by the intercept

One-hot vs. dummies
- One-hot encoding: Explainable features
- Dummy encoding: Necessary information without duplication

Dealing with categorical features
Index      Sex
0         Male
1       Female
2         Male

Index   Male    Female
0          1         0
1          0         1
2          1         0

Index   Male
0          1
1          0
2          1

Limiting your columns
counts = df['Country'].value_counts()
print(counts) -> in

'USA'       8
'UK'        6
'India'     2
'France'    1
Name: Country, dtype: object -> out

mask = df['Country'].isin(counts[counts < 5].index)
df['Country'][mask] = 'Other'
print(pd.value_counts(colors)) -> in

'USA'       8
'UK'        6
'Other'     3
Name: Country, dtype: object -> out
'''

# Convert the Country column to a one hot encoded Data Frame
one_hot_encoded = pd.get_dummies(
    so_survey_df, columns=['Country'], prefix='OH')

# Print the columns names
print(one_hot_encoded.columns)

# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=[
                       'Country'], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)


# Create a series out of the Country column
countries = so_survey_df['Country']

# Get the counts of each category
country_counts = countries.value_counts()

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())


'''
Numeric variables
Types of numeric features
- Age
- Price
- Counts
- Geospatial data

Does size matter?
    Resturant_ID    Number_of_Violations
0           RS_1                       0
1           RS_2                       0
2           RS_3                       2
3           RS_4                       1
4           RS_5                       0
5           RS_6                       0
6           RS_7                       4
7           RS_8                       4
8           RS_9                       1
9          RS_10                       0

Binarizing numeric variables
df['Binary_Violation'] = 0
df.loc[df['Number_of_Violations'] > 0, 'Binary_violation'] = 1

    Resturant_ID    Number_of_Violations    Binary_Violation
0           RS_1                       0                   0
1           RS_2                       0                   0
2           RS_3                       2                   1
3           RS_4                       1                   1
4           RS_5                       0                   0
5           RS_6                       0                   0
6           RS_7                       4                   1
7           RS_8                       4                   1
8           RS_9                       1                   1
9          RS_10                       0                   0

Binning numeric variables
import numpy as np
df['Binned_Group'] = pd.cut(df['Number_of_Violations'], bins=[-np.inf, 0, 2, np.inf], labels=[1, 2, 3])

    Resturant_ID    Number_of_Violations    Binned_Group
0           RS_1                       0               1 
1           RS_2                       0               1
2           RS_3                       2               2
3           RS_4                       1               2
4           RS_5                       0               1
5           RS_6                       0               1
6           RS_7                       4               3
7           RS_8                       4               3
8           RS_9                       1               2
9          RS_10                       0               1

- NB*: to include 0 in the first bin, we must set the leftmost edge to lower than that, so all values between negative infinity and 0 are labeled as 1.
* all values equal to q or 2 are labeled as 2, and values greater than 2 are labeled as 3.
'''

# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary'] > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())


# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())

# Import numpy

# Specify the boundaries of the bins
bins = [-np.inf, 10000, 50000, 100000, 150000, np.inf]

# Bin labels
labels = ['Very low', 'Low', 'Medium', 'High', 'Very high']

# Bin the continuous variable ConvertedSalary using these boundaries
so_survey_df['boundary_binned'] = pd.cut(
    so_survey_df['ConvertedSalary'], bins=bins, labels=labels)

# Print the first 5 rows of the boundary_binned column
print(so_survey_df[['boundary_binned', 'ConvertedSalary']].head())


''' Dealing with Messy Data '''

'''
Why do missing values exist?
How gaps in data  occur
- Data not being collected properly
- Collection and management errors
- Data intentionally being omitted
- Could be created due to transformations of the data

Why we care?
- Some models cannot work with missing data (Nulls /NaNs)
- Missing data may be a sign of a wider data issue
- Missing data can be a useful feature

Missing value discovery
print(df.info()) -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 999 entries, 0 to 998
Data columns (total 12 columns):
#                       Column     Non-Null Count   Dtype
-- --------------------------- ------------------  ------
0                  SurveryDate       999 non-null  object
. . .                    . . .              . . .   . . .
8   StackOverflowJobsRecommend       467 non-null float64
9               VersionControl       999 non-null  object
10                      Gender       693 non-null  object
11                   RawSalary       665 non-null  object
dtypes: float64(2), int64(2), object(8)
memory usage: 93.7+ KB -> out

Finding missing values
print(df.isnull()) -> in

    StackOverflowJobsRecommend  VersionControl . . . Gender     RawSalary
0                         True           False . . .  False          True
1                        False           False . . .  False          True
2                        False           False . . .   True          True
3                         True           False . . .  False         False
4                        False           False . . .  False         False -> out

print(df['StackOverflowJobsRecommend'].isnull().sum()) -> in

512 -> out

Finding non-missing values
print(df.notnull()) -> in

    StackOverflowJobsRecommend  VersionControl . . . Gender     RawSalary
0                        False            True . . .   True         False
1                         True            True . . .   True          True
2                         True            True . . .  False         False
3                        False            True . . .   True          True
4                         True            True . . .   True          True -> out
'''

# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.notnull().sum())


# Print the locations of the missing values
print(sub_df.head(10).isnull())

# Print the locations of the non-missing values
print(sub_df.head(10).notnull())


'''
Dealing with missing values (I)
- If confident that the missing values in the dataset are occuring at random (i.e not intentionally omitted), the most effective and statistically sound approach to dealing with them is called 'complete case analysis' or listwise deletion.

Listwise deletion
- In this method, a record is fully excluded from your model if any of its values are missing.

        SurveyDate  ConvertedSalary     Hobby . . . \
0    2/28/18 20:20              NaN       Yes . . .
1    6/28/18 13:26          70841.0       Yes . . .
2     6/6/18  3:37              NaN        No . . .
3     5/9/18  1:06          21426.0       Yes . . .
4    4/12/18 22:41          42671.0       Yes . . .

# Drop all rows with at least one missing values
df.dropna(how='any') 

# Drop rows with missing values in a specific column
df.dropna(subset=['VersionControl']) 

Issues with deletion
- It deletes valid data points
- It relies on randomness
- It reduces information

Replacing with strings
# Replace missing values in a specific column with a given string
df['VersionControl'].fillna(value='None Give', inplace=True)

Recording missing values
# Record where the values are not missing
df['SalaryGiven'] = df['ConvertedSalary'].notnull()

# Drop a specific column
df.drop(columns=['ConvertedSalary'])
'''

# Print the number of rows and columns
print(so_survey_df.shape)

# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna(how='any')

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)


# Replace missing values
so_survey_df['Gender'].fillna(value='Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())


'''
Dealing with missing values (II)
- One of the most common issues with removing all rows with missing values is if you were building a predictive model.
* Can't delete rows with missing values in the test set

What else can you do?
- Categorical columns: Replace missing values with the most common occuring value or with a string that flags missing values such as 'None'
- Numeric columns: Replace missing values with a suitable value

Measures of central tendency
- Mean
- Median

- NB: Using this method can lead to biased estimates of the variances and covariances of the features.
* Also, the standard error and test statistics can be incorctly estimated.

Calculating the measures of central tendency
print(df['ConvertedSalary'].mean())
print(df['ConvertedSalary'].median()) -> in

92565.16992481203
55562.0 -> out

- NB*: The missing values are excluded by default when calculating these statistics.

Fill the missing values
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(df['ConvertedSalary'].mean())
df['ConvertedSalary'] = df['ConvertedSalary'].astype('int64')

Rounding values
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(round(df['ConvertedSalary'].mean()))
'''

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(
    so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(
    so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())


'''
Dealing with other data issues
Bad characters
print(df['RawSalary'].dtype) -> in

dtype('O') -> out

print(df['RawSalary'].head()) -> in

0           NaN
1     70,841.00
2           NaN
3     21,426.00
4     41.671.00
Name: RawSalary, dtype: object -> out

Dealing with bad characters
df['RawSalary'] = df['RawSalary'].str.replace(',' , '')

df['RawSalary'] = df['RawSalary'].astype('float') -> in

Finding other stray characters
coerced_vals = df.to_numeric(errors = 'coerce')

print(df[coerced_vals.isna()].head()) -> in

0           NaN
2           NaN
4     $51408.00
Name: RawSalary, dtype: object -> out

Chaining methods
df['column_name'] = df['column_name'].method1()
df['column_name'] = df['column_name'].method2()
df['column_name'] = df['column_name'].method3()

Same as:
df['column_name'] = df['column_name'].method1().method2().method3()
'''

# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$', '')


# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])

# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£', '')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])


# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(
    ',', '').str.replace('$', '').str.replace('£', '').astype('float')

# Print the RawSalary column
print(so_survey_df['RawSalary'])


''' Conforming to Statistical Assumptions '''

'''
Data distributions
Distribution assumptions
- Almost every model besides tree based models assume that your data is normally distributed.
- Normal distributions follow a bell shaped 
- The main characteristics of a normal distribution is that 68% of the data lies within 1 Standard deviation of the mean
* 95% lies within 2 standard deviation from the mean
* 99.7% fall within 3 standard deviations from the mean

Observing your data
import matplotlib as plt

df.hist()
plt.show()

Delving deeper with box plots
- A box plot shows the distribution of the data by calculating where the middle 50% of the data sits, also known as the Inter quartile range or IQR.
* It sits between the 1st (25th Percentile) and 3rd (75th Percentile) and marking it with the box
- The wishkers extend to the minimum of 1.5 times the IQR from the edge of the box or the maximum range of the data.
- Any points outside this are marked as outliers

Box plots in pandas
df[['column_1']].boxplot()
plt.show()

Pairing distributions
import seaborn as sns
sns.pairplot(df)

- Pairplot are useful to see if multiple columns are correlated with each other or whether they have any association at all

Further details on your distributions
df.describe() -> in

            Col1        Col2        Col3        Col4
count 100.000000  100.000000  100.000000  100.000000
mean   -0.163779   -0.014801   -0.087965   -0.045790
std     1.046370    0.920881    0.936678    0.916474
min    -2.781872   -2.156124   -2.647595   -1.957858
25%    -0.849232   -0.655239   -0.602699   -0.736089
50%    -0.179495    0.032115   -0.051863    0.066803
75%     0.663515    0.615688    0.417917    0.689591
max     2.466219    2.353921    2.059511    1.838561 -> out
'''

# Create a histogram
so_numeric_df.hist()
plt.show()

# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()

# Create a boxplot of ConvertedSalary
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()


# Import packages

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()

# Print summary statistics
print(so_numeric_df.describe())


'''
Scaling and transformations
Scaling data
- Min-Max scaling (sometimes referred to as normalization)
- Standardization

Min-Max scaling
- This is when the data is scaled linearly between a minimum and maximum value often 0 and 1.

Min-Max scaling in Python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['normalized_age'] = scaler.transform(df[['Age']])

Standardization
- This finds the mean of the data and centers the distribution around it, calculating the number of standard deviations away from the mean each point is.

Standardization in Python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df[['Age']])
df['standardized_col'] = scaler.transform(df[['Age']])

Log Transformation
- It is used to make highly skewed distributions less skewed.

Log Transformation in Python
from sklearn.preprocessing import PowerTransformer

log = PowerTransformer()
log.fit(df[['ConvertedSalary']])
df['log_ConvertedSalary'] = log.transform(df[['ConvertedSalary']])
'''

# Import MinMaxScaler

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())


# Import StandardScaler

# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())


# Import PowerTransformer

# Instantiate PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(
    so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()


'''
Removing outliers
What are outliers?
- They are data points tat exist far away from the majority of the data.

Quantile based detection
- This approach is useful when trying to avoid the upper values of the data columns

Quantiles in Python
q_cutoff = df['col_name'].quantile(0.95)
mask = df['col_name'] < q_cutoff
trimmed_df = df[mask]

Standard deviation based detection
- This approach has the benefits of ony removing genuinely extreme values

Standard deviation detection in Python
mean = df['col_name'].mean()
std = df['col_name'].std()
cut_off = std * 3

lowr, upper = mean - cutoff, mean + cut_off
new_df = df[(df['col_name'] < upper) & df['col_name'] > lower)]
'''

# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()


# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) & (
    so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()


'''
Scaling and transforming new data
Reuse training scalers
scaler = StandardScaler()
scaler.fit(train[['col']])
train['scaled_col'] = scaler.transform(train[['col']])

# FIT SOME MODEL
test = pd.read_csv('test_csv')
test['scaled_col'] = scaler.transform(test[['col']])

- NB*: The scaler is fitted only on the training data. 
* i.e fit and transform the training data, but only transform the test data.

Training transformations for reuse
train_mean = train[['col']].mean()
train_std = train[['col']].std()

cut_off = train_std * 3
train_lower = train_mean - cut_off
train_upper = train_mean + cut_off

# Subset train data
test = pd.read_csv('test_csv')

# Subset test data
test = test[(test[['col']] < train_upper & test[['col']] > train_upper)]

Why only use training data ?
Data leakage: Using data that you won't have access to when assessing the performance of your model.
'''

# Import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())


train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) & (
    so_test_numeric['ConvertedSalary'] > train_lower)]


''' Dealing with Text Data '''

'''
Encoding text
Standarizing your text
Example of free text:
Fellow-Citizens of the Senate and of the House of Representatives: AMONG the vicissitudes incident to life no event could have filled me with greater anxieties than that of which the notification was transmitted by your order, and received on the th day of the present month.

- Data that is not in a predefined form is called unstructured data, and free text data is a good example of this.

Dataset
print(speech_df.head()) -> in

                Name          Inaugural Address                         Date                           text
0  George Washington    First Inaugural Address     Thursday, April 30, 1789    Fellow-Citizens of the Sena. . .
1  George Washington   Second Inaugural Address        Monday, March 4, 1793    Fellow Citizens: I AM again. . .
2         John Adams          Inaugural Address      Saturday, March 4, 1797    WHEN it was first perceived. . . 
3   Thomas Jefferson    First Inaugural Address     Wednesday, March 4, 1801    Friends and Feloow-Citizens. . .  
    Thomas Jefferson   Second Inaugural Address        Monday, March 4, 1805    PROCEEDING, fellow-citizens. . . -> out

Removing unwanted charcaters
- [a-zA-Z]: All letter characters
- [^a-zA-Z]: All non letter characters

speech_df['text'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

Before:
'Fellow-Citizens of the Senate and of the House of Representatives: AMONG the vicissitudes incident to life no event could have filled me with greater' . . .

After:
'Fellow Citizens of the Senate and of the House of Representatives  AMONG the vicissitudes incident to life no event could have filled me with greater' . . .

Standardize the case
speech_df['text'] = speech_df['text'].str.lower()
print(speech_df['text'][0]) -> in

'fellow citizens of the senate and of the house of representatives  among the vicissitudes incident to life no event could have filled me with greater' . . . -> out

Length of text
speech_df['char_cnt'] = speech_df['text'].str.len()
print(speech_df['char_cnt'].head()) -> in

0   1889
1    806
2   2408
3   1495
4   2465
Name: char_cnt, dtype: int64 -> out

Word counts
speech_df['word_cnt'] = speech_df['text'].str.split()
print(speech_df['word_cnt'].head(1)) -> in

['fellow', 'citizens', 'of', 'the', 'senate', 'and', . . .  -> out

speech_df['word_counts'] = speech_df['text'].str.split().str.len()
print(speech_df['word_splits'].head()) -> in

0   1432
1    135
2   2323
3   1736
4   2169
Name: word_cnt, dtype: int64 -> out

Average length of word
speech_df['avg_word_len'] = speech_df['char_cnt'] / speech_df['word_cnt']
'''

# Print the first 5 rows of the text column
print(speech_df['text'].head(5))

# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())


# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()

# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()

# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']

# Print the first 5 rows of these columns
print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])


'''
Word counts
Text to columns
'citizens of the senate and of the house of representatives'

Index   citizens    of  the     senate  and     house   representatives
1              1     3    2          1    1         1                 1

Initializing the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
print(cv) -> in

CountVectorizer(analyzer=u'word', binary=False, decode_error=u'strict', dtype=<type 'numpy.int64'>, encoding=u'utf-8', input=u'content', lowercase=True, max_df=1.0, max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None, stop_words=None, strip_accents=None, token_pattern=u'(?u)\\b\\w\\w+\b', tokenizer=None, vocabulary=None

Specifying the vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0.1, max_df=0.9)

min_df: minimum fraction of documents the word must occur in
max_df: maximum fraction of documents the word can occur in 

Fit the vectorizer
cv.fit(speech_df['text_clean'])

transforming your text
cv_transformed = cv.transform(speech_df['text_clean'])
print(cv_transformed) -> in

<58x8839 sparse matrix of type '<type 'numpy.int64'>' -> out

Transforming your text
cv_transformed.toarray()

Getting the features
feature_names = cv.get_feature_names()
print(feature_names) -> in

[u'abandon', u'abandoned', u'abandonment', u'abate', u'abdicated', u'abeyance' u'abhorring', u'abide', u'abiding', u'abilities', u'ability', u'abject'. . . -> out

Fitting and transforming
cv_transformed = cv.fit_transformed(speech_df['text_clean'])
print(cv_transformed) -> in

<58x8839 sparse matrix of type '<type 'numpy.int64'>' -> out

Putting it all together
cv_df = pd.DataFrame(cv_transformed.toarray(), columns = cv.get_feature_names()).add_prefix('Counts_')
print(cv_df.head()) -> in

    Counts_aback    Counts_abandoned    Counts_a. . .
0              1                   0            . . .
1              0                   0            . . .
2              0                   1            . . .
3              0                   1            . . .
4              0                   0            . . . -> out

Updating your DataFrame
speech_df = pd.concat([speech_df, cv_df], axis=1, sort=False)
print(speech_df.shape) -> in

(58, 8845) -> out
'''

# Import CountVectorizer

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit the vectorizer
cv.fit(speech_df['text_clean'])

# Print feature names
print(cv.get_feature_names())


# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()

# Print the shape of cv_array
print(cv_array.shape)


# Import CountVectorizer

# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=0.2, max_df=0.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()

# Print the array shape
print(cv_array.shape)


# Create a DataFrame with these features
cv_df = pd.DataFrame(
    cv_array, columns=cv.get_feature_names()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)
print(speech_df_new.head())


'''
Term frequency-inverse document frequency
Introducing TF-IDF
print(speech_df['Counts_the'].head()) -> in

0   21
1   13
2   29
3   22
4   20 -> out

TF-IDF = ( count of word occurences / Total words in document ) / log( Number of documents word is in / Total number of documents )

Importing the vectorizer
from sklearn.feature_extraction.text import TfidVectorizer
tv = TfidfVectorizer()
print(tv) -> in

TfidfVectorizer(analyzer=u'word', binary=False, decode_error=u'strict', dtype=<type 'numpy.float64'>, encoding=u'utf-8', input=u'content', lowercase=True, max_df=1.0, max_features=None, min_df=1, ngram_range=(1, 1), norm=u'l2', preprocessor=None, stop_words=None, strip_accents=None, sublinear_tf=False, token_pattern=u'(?u)\\b\\w\\w+\b', tokenizer=None, vocabulary=None)

Max features and stopwords
tv = TfidfVectorizer(max_features=100, stop_words='english')

- max_features: Maximum number of columns created from TF-IDF
- stop_words: List of common words to omit e.g. 'and', 'the' etc.

Fitting your text
tv.fit(train_speech_df['text'])
train_tv_transformed =  tv.transformed(train_speech_df['text'])

Putting it all together
train_tv_df = pd.DataFrame(train_tv_transformed.toarray(), columns=tv.get_feature_names()).add_prefix('TFIDF_')

train_speech_df = pd.concat([train_speech_df, train_tv_df], axis=1, sort=False)

Inspecting your transforms
examine_row = train_tv_df.iloc[0]
print(examine_row.sort_values(ascending=False))

TFIDF_government    0.367430
TFIDF_public        0.333237
TFIDF_present       0.315182
TFIDF_duty          0.238637
TFIDF_citizens      0.229644
Name: 0, dtype: float64 -> out

Applying the vectorizer to new data
test_tv_transformed = tv.transform(test_df['text_clean'])

test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), columns=tv.get_feature_names()).add_prefix('TFIDF_')

test_speech_df = pd.concat([test_speech_df, test_tv_df], axis=1, sort=False)
'''

# Import TfidfVectorizer

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(),
                     columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())


# Isolate the row to be examined
sample_row = tv_df.iloc[0]

# Print the top 5 words of the sorted output
print(sample_row.sort_values(ascending=False).head())


# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])

# Transform test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])

# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(
), columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(test_tv_df.head())


'''
N-grams
- When looking at individual words on their own without any context or word order, its called a Bag-of-Words model.
* As the words are treated as if they are being drawn from a bag with no concept of order or grammar.

Issues with bag of words
Positive meaning
Single word: happy

Negative meaning
Bi-gram: not happy

Positive meaning
Trigram: never not happy

Using N-grams
tv_bi_gram_vec = TfidfVectorizer(ngram_range = (2, 2))

# Fit and apply bigram vectorizer
tv_bi_gram = tv_bi_gram_vec.fit_transform(speech_df['text'])

# Print the bigram features
print(tv_bi_gram_vec.get_feature_names()) -> in

[u'american people', u'best ability', u'beloved country', u'best interests' . . . ] -> out

Finding common words
# Create a DataFrame with the Counts features
tv_df = pd.DataFrame(tv_bi_gram.toarray(), columns=tv_bi_gram_vec.get_feature_names()).add_prefix('Counts_')

tv_sums = tv_df.sum()
print(tv_sums.head()) -> in

Counts_administration government    12
Counts_almighty god                 15
Counts_american people              36
Counts_beloved country               8
Counts_best ability                  8
dtype: int64 -> out

Finding common  words
print(tv_sums.sort_values(ascending=False)).head() -> in

Counts_united states            152
Counts_fellow citizens           97
Counts_american people           36
Counts_federal government        35
Counts_self government           30
dtype: int64 -> out
'''

# Import CountVectorizer

# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(
    max_features=100, stop_words='english', ngram_range=(3, 3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print(cv_trigram_vec.get_feature_names())


# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(cv_trigram.toarray(
), columns=cv_trigram_vec.get_feature_names()).add_prefix('Counts_')

# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending=False).head())
