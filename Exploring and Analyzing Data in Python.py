# import necessary packages
from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from empiricaldist import Pmf, Cdf
from scipy.stats import norm, linregress
import statsmodels.formula.api as smf


''' Read, clean, and validate '''

'''
DataFrames and Series
Reading data
e.g
import pandas as pd
df = pd.read_hdf('data.hdf5', 'data')
type(df)

*   df - DataFrame name
*   int - Integer
*   Nan - Not a Number
*   ~   - It is called the Tilde Operator and it is used to get the inverse values
*   A variable that's computed from other variables is sometimes called a 'recode'

df.head() - shows first 5 rows of the DataFrame

Columns and rows
e.g
df.shape - gives the number of rows and columns in the dataset

df.columns - gives a list of variables names for the columns

Selecting a column
e.g
df1 = df['column_name']
'''

# Display the number of rows and columns
nsfg = pd.read_hdf('Exploring and Analyzing Data in Python/nsfg.hdf5', 'nsfg')
print(nsfg.shape)

# Display the names of the columns
print(nsfg.columns)

# Select column birthwgt_oz1: ounces
ounces = nsfg['birthwgt_oz1']

# Print the first 5 elements of ounces
print(ounces.head())


'''
Clean and Validate
Validating Data
Selecting a column
e.g
df1 = df['column_name']
df1.value_counts().sort_index()
OR
df1.describe()

Cleaning Date
e.g
df1 = df1.replace([value1, value2], np.nan)
OR
df1.replace([value1, value2], np.nan, inplace = True)

Arithmetic with Series
e.g
df2 = df1 / int
'''

# Replace the value 8 with NaN
nsfg['nbrnaliv'].replace(8, np.nan, inplace=True)

# Print the values and their frequencies
print(nsfg['nbrnaliv'].value_counts())


# Select the columns and divide by 100
agecon = nsfg['agecon'] / 100
agepreg = nsfg['agepreg'] / 100

# Compute the difference
preg_length = agepreg - agecon

# Compute summary statistics
print(preg_length.describe())


'''
Filter and visualize
Histogram
e.g
import matplotlib.pyplot as plt
plt.hist(df1.dropna(), bins = int)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

Boolean Series
e.g
df1 = df['column_name'] < int
df1.sum() - Count the no of True values in the series

Filtering
e.g
df2 = df [ df1 ] - Filter the values the are True
df2 = df [ ~df1 ] - Filter the values the are False

Other logical operators:
& for AND (both must be true)
| for OR (either or both can be true)
e.g
df [ df1 & df2 ] # Both True
df [ df1 | df2 ] # Either or both True

Resampling
It is used for correcting Oversampling  #Read_More
'''

# Plot the histogram
plt.hist(agecon.dropna(), bins=20)

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()


# Plot the histogram
plt.hist(agecon, bins=20, histtype='step')

# Label the axes
plt.xlabel('Age at conception')
plt.ylabel('Number of pregnancies')

# Show the figure
plt.show()


# Resample the data
def resample_rows_weighted(df, column='finalwgt', seed=None):
    np.random.seed(seed)
    weights = df[column]
    probabilities = weights / weights.sum()
    indices = np.random.choice(df.index, size=len(
        df), replace=True, p=probabilities)
    resampled_df = df.loc[indices].reset_index(drop=True)
    return resampled_df


nsfg = resample_rows_weighted(nsfg, 'wgt2013_2015')

# Clean the weight variables
pounds = nsfg['birthwgt_lb1'].replace([98, 99], np.nan)
ounces = nsfg['birthwgt_oz1'].replace([98, 99], np.nan)

# Compute total birth weight
birth_weight = pounds + ounces/16

# Create a Boolean Series for full-term babies
full_term = nsfg['prglngth'] >= 37

# Select the weights of full-term babies
full_term_weight = birth_weight[full_term]

# Compute the mean weight of full-term babies
print(full_term_weight.mean())


# Filter full-term babies
full_term = nsfg['prglngth'] >= 37

# Filter single births
single = nsfg['nbrnaliv'] == 1

# Compute birth weight for single full-term babies
single_full_term_weight = birth_weight[full_term & single]
print('Single full-term mean:', single_full_term_weight.mean())

# Compute birth weight for multiple full-term babies
mult_full_term_weight = birth_weight[full_term & ~single]
print('Multiple full-term mean:', mult_full_term_weight.mean())


''' Distributions '''

'''
Probability mass functions
#   PMF represents the possible values in a distribution and their probabilities

- Run on the Command Prompt
pip help
pip install empiricaldist

Read the data
e.g
df = pd.read_hdf('data.hdf5', 'data')
df.head()

PMF - Probability mass functions
Normalize - If set to True will give a fraction of the result value

e.g
from empiricaldist import Pmf
pmf_colname = Pmf(df['column_name'], normalize=False)
pmf_colname.bar(label = 'column label')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

Evaluating the PMF
e.g
q = int
p = pmf_colname(q)
print(p)
'''

gss = pd.read_hdf('Exploring and Analyzing Data in Python\gss.hdf5', 'gss')

# Compute the PMF for year
pmf_year = Pmf.from_seq(gss['year'], normalize=False)

# Print the result
print(pmf_year)


# Select the age column
age = gss['age']

# Make a PMF of age
pmf_age = Pmf.from_seq(age)

# Plot the PMF
pmf_age.bar()

# Label the axes
plt.xlabel('Age')
plt.ylabel('PMF')
plt.show()


'''
Cumulative distribution functions
CDF are useful for some computations; they are also a great way to visualize and compare distributions.

From PMF to CDF
If you draw a random element from a distribution:
- PMF (Probability Mass Function) is the probability that you get exactly x, for any given value of x.
- CDF (Cumulative Distribution Function) is the probability that you get a value <= x, for any given value of x.
e.g
PMF of {1, 2, 2, 3, 5}      CDF is the cumulative sum of the PMF
PMF(1) = 1/5                CDF(1) = 1/5
PMF(2) = 2/5                CDF(2) = 3/5
PMF(3) = 1/5                CDF(3) = 4/5
PMF(5) = 1/5                CDF(5) = 1

from empiricaldist import Cdf
cdf_colname = Cdf(df['column_name'])
cdf_colname.plot()
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

Evaluating the CDF
e.g
q = int
p = cdf_colname(q)
print(p)

Evaluating the inverse CDF
e.g
p = float
q = cdf_colname.inverse(p)
print(q)
'''

# Select the age column
age = gss['age']

# Compute the CDF of age
cdf_age = Cdf.from_seq(age)

# Calculate the CDF of 30
print(cdf_age[30])


# Calculate the 75th percentile
income = gss['realinc']
cdf_income = Cdf.from_seq(income)

percentile_75th = cdf_income.inverse(0.75)

# Calculate the 25th percentile
percentile_25th = cdf_income.inverse(0.25)

# Calculate the interquartile range
iqr = percentile_75th - percentile_25th

# Print the interquartile range
print(iqr)


# Select realinc
income = gss['realinc']

# Make the CDF
cdf_income = Cdf.from_seq(income)

# Plot it
cdf_income.plot()

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.show()


'''
Comparing distributions
df1 = df['column_name'] == int
df2 = df['column_name1']
df3 = df2[df1]
df4 = df2[~df1]

Multiple PMFs
e.g
Pmf(df3).plot(label='plot_label')
Pmf(df4).plot(label='plot_label1')

plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

Multiple CDFs
e.g
Cdf(df3).plot(label='plot_label')
Cdf(df4).plot(label='plot_label1')

plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

# In general, CDFs are smoother than PMFs. Because they smooth out randomness
'''

# Select educ
educ = gss['educ']

# Bachelor's degree
bach = (educ >= 16)

# Associate degree
assc = (educ >= 14) & (educ < 16)

# High school (12 or fewer years of education)
high = (educ <= 12)
print(high.mean())

income = gss['realinc']

# Plot the CDFs
Cdf.from_seq(income[high]).plot(label='High school')
Cdf.from_seq(income[assc]).plot(label='Associate')
Cdf.from_seq(income[bach]).plot(label='Bachelor')

# Label the axes
plt.xlabel('Income (1986 USD)')
plt.ylabel('CDF')
plt.legend()
plt.show()


'''
Modeling distributions
The normal distribution (Gaussian distribution)
e.g
sample = np.random.normal(size=1000)
Cdf(sample).plot()
plt.show()

The normal CDF
SciPy provides an object called norm that represents the normal distribution
e.g
from scipy.stats import norm

xs = np.linspace(-3, 3) - To create an array of equally-spaced points from -3 to 3
ys = norm(0, 1).cdf(xs) - Creates an object that represents a normal distribution with Mean 0 and Standard Deviation 1.
plt.plot(xs, ys, color='gray')
Cdf(sample).plot()

The bell curve
PDF - Probability Density Function
e.g
xs = np.linspace(-3, 3)
ys = norm(0, 1).pdf(xs)
plt.plot(xs, ys, color='gray')

# When the points in a sample are used to estimate the PDF of the distribution they came from, the process is called Kernel Density Estimation (KDE)
# KDE is a way of getting from a PMF, a Probability Mass Function, to a PDF, a Probability Density Function.

KDE Plot
e.g
import seaborn as sns
sns.kdeplot(sample)

KDE and PDF
e.g
xs = np.linspace(-3, 3)
ys = norm.pdf(xs)
plt.plot(xs, ys, color='gray')
sns.kdeplot(sample)


PMF, CDF, KDE
- Use CDFs for exploration.
- Use PMFs if there are a small number of unique values.
- Use KDE if there are a lot of values.
'''

# Extract realinc and compute its log
income = gss['realinc']
log_income = np.log10(income)

# Compute mean and standard deviation
mean = log_income.mean()
std = log_income.std()
print(mean, std)

# Make a norm object
dist = norm(mean, std)


# Evaluate the model CDF
xs = np.linspace(2, 5.5)
ys = dist.cdf(xs)

# Plot the model CDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Create and plot the Cdf of log_income
Cdf.from_seq(log_income).plot()

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('CDF')
plt.show()


# Evaluate the normal PDF
xs = np.linspace(2, 5.5)
ys = dist.pdf(xs)

# Plot the model PDF
plt.clf()
plt.plot(xs, ys, color='gray')

# Plot the data KDE
sns.kdeplot(log_income)

# Label the axes
plt.xlabel('log10 of realinc')
plt.ylabel('PDF')
plt.show()


''' Relationships '''

'''
Exploring relationships
Scatter plot
e.g
df = pd.read_hdf('brfss.hdf5', 'brfss')
df1 = df['column_name']
df2 = df['column_name1']

plt.plot(df1, df2, 'o')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.show()

# 'o' - This plots a circle for each data point

Transparency
e.g
plt.plot(df1, df2, 'o', alpha=0.02)
plt.show()

Marker size
e.g
plt.plot(df1, df2, 'o', markersize=1, alpha=0.02)
plt.show()

Jittering
e.g
df1_jitter = df1 + np.random.normal(0, 2, size=len(brfss))
df2_jitter = df2 + np.random.normal(0, 2, size=len(brfss))
plt.plot(df1_jitter, df2_jitter, 'o', markersize=1, alpha=0.02)
plt.show()

Zoom
e.g
plt.plot(df1_jitter, df2_jitter, 'o', markersize=1, alpha=0.02)
plt.axis([x_int, x_int, y_int, y_int])
plt.show()
'''

brfss = pd.read_hdf(
    'Exploring and Analyzing Data in Python/brfss.hdf5', 'brfss')

# Extract age
age = brfss['AGE']

# Plot the PMF
pmf_age = Pmf.from_seq(age)
pmf_age.bar()

# Label the axes
plt.xlabel('Age in years')
plt.ylabel('PMF')
plt.show()


# Select the first 1000 respondents
brfss = brfss[:1000]

# Extract age and weight
age = brfss['AGE']
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', alpha=0.1)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')

plt.show()


# Select the first 1000 respondents
brfss = brfss[:1000]

# Add jittering to age (mean = 0, std = 2.5)
age = brfss['AGE'] + np.random.normal(0, 2.5, size=len(brfss))
# Extract weight
weight = brfss['WTKG3']

# Make a scatter plot
plt.plot(age, weight, 'o', markersize=5, alpha=0.2)

plt.xlabel('Age in years')
plt.ylabel('Weight in kg')
plt.show()


'''
Visualizing relationships
The Violin plot
- This can be gotten when a KDE is used to estimate the density function in each column of a dataframe.
- Before plotting, first get rid of rows with missing data.
e.g
df = brfss.dropna(subset=['column_name', 'column_name1'])
sns.violinplot(x='column_name', y='column_name1', data=df, inner=None)
plt.show()

# inner = None is used to simplify the plot

Box plot
e.g
sns.boxplot(x='column_name', y='column_name1', data=df, whis=10)
plt.show()

# whis = 10 is used to turn off a feature we dont need

Log scale
This is useful for data that skews toward higher values
e.g
sns.boxplot(x='column_name', y='column_name1', data=df, whis=10)
plt.yscale('log')
plt.show()
'''

# Drop rows with missing data
data = brfss.dropna(subset=['_HTMG10', 'WTKG3'])

# Make a box plot
sns.boxplot(x='_HTMG10', y='WTKG3', data=data, whis=10)

# Plot the y-axis on a log scale
plt.yscale('log')

# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Height in cm')
plt.ylabel('Weight in kg')
plt.show()


# Extract income
income = brfss['INCOME2']

# Plot the PMF
Pmf.from_seq(income).bar()

# Label the axes
plt.xlabel('Income level')
plt.ylabel('PMF')
plt.show()


# Drop rows with missing data
data = brfss.dropna(subset=['INCOME2', 'HTM4'])

# Make a violin plot
sns.violinplot(x='INCOME2', y='HTM4', data=data, inner=None)

# Remove unneeded lines and label axes
sns.despine(left=True, bottom=True)
plt.xlabel('Income level')
plt.ylabel('Height in cm')
plt.show()


'''
Correlation (Pearson's correlation coefficient)
- Coefficient of correlation, quantifies the strength of the relationships between pairs of variables
- in statistics, Pearson's correlation coefficient is a number between -1 and 1 that quantifies the strength of a linear relationship between variables.
- Correlation only works with linear relationships
e.g
columns = ['column_name', 'column_name1', 'column_name2']
subset = df[columns]

subset.corr()
'''

# Select columns
columns = ['AGE', 'INCOME2', '_VEGESU1']
subset = brfss[columns]

# Compute the correlation matrix
print(subset.corr())


'''
Simple regression
linregress() from the SciPy stats module can be used to estimate the slope of a line.
The result of a linregress object contains 5 values: 
* Slope - This is the slope of the line of best fit for the data
* Intercept - This is the intercept
* rvalue - This is the correlation
* pvalue
* stderr

Regression lines
- linregress() can't handle Nans, so dropna()
e.g
df = brfss.dropna(subset=['column_name', 'column_name1'])
df1 = df['column_name']
df2 = df['column_name1']
res = linregress(df1, df2)

fx = np.array([df1.min(), df1.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-')

# Linear Regressions only measures the strength of a linear relationship 
'''

# Extract the variables
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']

# Compute the linear regression
res = linregress(xs, ys)
print(res)


# Plot the scatter plot
plt.clf()
x_jitter = xs + np.random.normal(0, 0.15, len(xs))
plt.plot(x_jitter, ys, 'o', alpha=0.2)

# Plot the line of best fit
fx = np.array([xs.min(), xs.max()])
fy = res.intercept + res.slope * fx
plt.plot(fx, fy, '-', alpha=0.7)

plt.xlabel('Income code')
plt.ylabel('Vegetable servings per day')
plt.ylim([0, 6])
plt.show()


''' Multivariate Thinking '''

'''
Limits of simple regression
- Regression is not symmetric
- Regression is not causation
SciPy doesn't do multiple regression
StatsModels library can do multiple regression

Multiple regression
e.g
import statsmodels.formula.api as smf

df = smf.ols('column_name ~ column_name1', data=data).fit()
df.params

# 'ols' stands for 'Ordinary Least Squares', another name for regression.
# '.fit()' - It is used to get the results
# 'params' - It contains the estimated slope and intercept
'''


# Run regression with linregress
subset = brfss.dropna(subset=['INCOME2', '_VEGESU1'])
xs = subset['INCOME2']
ys = subset['_VEGESU1']
res = linregress(xs, ys)
print(res)

# Run regression with StatsModels
results = smf.ols('_VEGESU1 ~ INCOME2', data=brfss).fit()
print(results.params)


'''
Multiple regression
2 variables
e.g
data = pd.read_hdf('data.hdf5', 'data')

df = smf.ols('column_name ~ column_name1', data=data).fit()
df.params

Adding a 3rd variable
e.g
df = smf.ols('column_name ~ column_name1 + column_name2', data=data).fit()
df.params

Using GroupBy
grouped = data.groupby('column_name')

df_mean = grouped['column_name1'].mean()
plt.plot(df_mean, 'o', alpha=0.5)
plt.xlabel('x axis label')
plt.ylabel('y axis label')

# Multiple regression can measure and describe non-Linear relationship

Adding a quadratic term
data['column_name3'] = data['column_name2']**2

model = smf.ols('column_name ~ column_name1 + column_name2 + column_name3', data=data)
df = model.fit()
df.params
'''

# Group by educ
grouped = gss.groupby('educ')

# Compute mean income in each group
mean_income_by_educ = grouped['realinc'].mean()

# Plot mean income as a scatter plot
plt.plot(mean_income_by_educ, 'o', alpha=0.5)

# Label the axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.show()


# Add a new column with educ squared
gss['educ2'] = gss['educ']**2

# Add a new column with age squared
gss['age2'] = gss['age']**2

# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()

# Print the estimated parameters
print(results.params)


'''
Visualizing regression results
e.g
data['column_name3'] = data['column_name1']**2
data['column_name2'] = data['column_name4']**2

model = smf.ols('column_name ~ column_name1 + column_name3 + column_name2 + column_name4', data = data)
df = model.fit()
df.params

Generating predictions
The regression results object provides a method called predict() that uses the model to generate predictions.
e.g
df1 = pd.DataFrame()
df1['column_name1'] = np.linspace(1, 10)
df1['column_name3'] = df1['column_name1']**2

df1['column_name2'] = int
df1['column_name4'] = df1['column_name2']**2

pred1 = df.predict(df1)

Plotting predictions
plt.plot(df1['column_name1'], pred1, label='plot label')
plt.plot(df_mean, 'o', alpha=0.5)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.legend()
'''

# Run a regression model with educ, educ2, age, and age2
results = smf.ols('realinc ~ educ + educ2 + age + age2', data=gss).fit()

# Make the DataFrame
df = pd.DataFrame()
df['educ'] = np.linspace(0, 20)
df['age'] = 30
df['educ2'] = df['educ']**2
df['age2'] = df['age']**2

# Generate and plot the predictions
pred = results.predict(df)
print(pred.head())


# Plot mean income in each age group
plt.clf()
grouped = gss.groupby('educ')
mean_income_by_educ = grouped['realinc'].mean()
plt.plot(mean_income_by_educ, 'o', alpha=0.5)

# Plot the predictions
pred = results.predict(df)
plt.plot(df['educ'], pred, label='Age 30')

# Label axes
plt.xlabel('Education (years)')
plt.ylabel('Income (1986 $)')
plt.legend()
plt.show()


'''
Logistic regression
Categorical variables
Numerical variables: income, age, years of education.
Categorical variables: sex, race.
e.g
formula = 'column_name ~ column_name1 + column_name3 + column_name2 + column_name4 + C(column_name5)'
df = smf.ols(formula, data=data).fit()
df.params

#   C(column_name5) is used to specify the categorical variable.

Boolean variable
data['column_name'].value_counts()
data['column_name'].replace([2], [0], inplace=True)
data['column_name'].value_counts()

Logistic regression
e.g
formula = 'column_name ~ column_name1 + column_name3 + column_name2 + column_name4 + C(column_name5)'
df = smf.logit(formula, data=data).fit()
df.params


Generating predictions
e.g
df1 = pd.DataFrame()
df1['column_name1'] = np.linspace(1, 10)
df1['column_name2'] = int

df1['column_name3'] = df1['column_name1']**2
df1['column_name4'] = df1['column_name2']**2

df1['column_name5'] = int
pred1 = df.predict(df1)

df1['column_name5'] = int
pred2 = df.predict(df1)

Visualizing results
e.g
grouped = data.groupby('column_name2')
df_mean = grouped['column_name6'].mean()
plt.plot(df_mean, 'o', alpha=0.5)

plt.plot(df1['column_name2'], pred1, label='plot label')

plt.plot(df1['column_name2'], pred2, label='plot label1')

plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.legend()
'''

# Recode grass
gss['grass'].replace(2, 0, inplace=True)

# Run logistic regression
results = smf.logit(
    'grass ~ age + age2 + educ + educ2 + C(sex)', data=gss).fit()
print(results.params)

# Make a DataFrame with a range of ages
df = pd.DataFrame()
df['age'] = np.linspace(18, 89)
df['age2'] = df['age']**2

# Set the education level to 12
df['educ'] = 12
df['educ2'] = df['educ']**2

# Generate predictions for men and women
df['sex'] = 1
pred1 = results.predict(df)

df['sex'] = 2
pred2 = results.predict(df)

plt.clf()
grouped = gss.groupby('age')
favor_by_age = grouped['grass'].mean()
plt.plot(favor_by_age, 'o', alpha=0.5)

plt.plot(df['age'], pred1, label='Male')
plt.plot(df['age'], pred2, label='Female')

plt.xlabel('Age')
plt.ylabel('Probability of favoring legalization')
plt.legend()
plt.show()
