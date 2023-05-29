''' Getting to Know a Dataset '''

'''
Initial exploration
Exploratory Data Analysis
The process of reviewing and cleaning data to ...
- derive insights such as
* descriptive statistics
* correlation
- generate hypotheses for experiments

A first look with .head()
books = pd.read_csv('books.csv')
books.head()    -> in

name                                              author  rating    year          genre
10-Day Green Smoothie Cleanse                   JJ Smith    4.73    2016    Non Fiction
11/22/63: A Novel                           Stephen King    4.62    2011        Fiction
12 Rules for Life                     Jordan B. Peterson    4.69    2028    Non Fiction
1984 (Signet Classics)                     George Orwell    4.73    2017        Fiction
5,000 Awesome Facts             National Geographic Kids    4.81    2019      Childrens     -> out

Gathering more .info()
books.info()    -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 350 entries, 0 to 349
Data columns (total 5 columns):
#   Column  Non-Null Count     Dtype
0     name    350 non-null    object
1   author    350 non-null    object
2   rating    350 non-null   float64
3     year    350 non-null     int64
4    genre    350 non-null    object
dtypes: float64(1), int64(1), object(3)
memory usage: 13.8+ KB  -> out

A closer look at categorical columns
books.value_counts('genre') -> in

genre
Non Fiction     179
Fiction         131
Childrens        40
dtype: int64    -> out

.describe() numerical columns
books.describe()    -> in

            rating         year
count   350.000000   350.000000
mean      4.608571  2013.508571
std       0.226941     3.284711
min       3.300000  2009.000000
25%       4.500000  2010.000000
50%       4.600000  2013.000000
75%       4.800000  2016.000000
max       4.900000  2019.000000     -> out

Visualizing numerical data
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data=books, x='rating', binwidth=0.1)
plt.show()
'''

# Print the first five rows of unemployment
print(unemployment.head())

# Print a summary of non-missing values and data types in the unemployment DataFrame
print(unemployment.info())

# Print summary statistics for numerical columns in unemployment
print(unemployment.describe())


# Count the values associated with each continent in unemployment
print(unemployment['continent'].value_counts())


# Import the required visualization libraries
# import seaborn as sns
# import matplotlib.pyplot as plt

# Create a histogram of 2021 unemployment; show a full percent in each bin
sns.histplot(data=unemployment, x='2021', binwidth=1)
plt.show()


'''
Data validation
Validating data types
books.info()    -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 350 entries, 0 to 349
Data columns (total 5 columns):
#   Column  Non-Null Count     Dtype
0     name    350 non-null    object
1   author    350 non-null    object
2   rating    350 non-null   float64
3     year    350 non-null     int64
4    genre    350 non-null    object
dtypes: float64(1), int64(1), object(3)
memory usage: 13.8+ KB  -> out

books.dtypes -> in

name     object
author   object
rating  float64
year    float64
genre    object
dtype:   object     -> out

Upating data types
books['year'] = books['year'].astype(int)
books.dtypes    -> in

name     object
author   object
rating  float64
year      int64
genre    object
dtype:   object     -> out

Updating data types
Type        Python Name
String              str
Integer             int
Float             float
Dictionary         dict
List               list
Boolean            bool

Validating categorical data
books['genre'].isin(['Fiction', 'Non Fiction'])     -> in

0       True
1       True
2       True
3       True
4      False
    ...
345     True
346     True
347     True
348     True
349    False
Name: genre, Length: 350, dtype: bool   -> out

Inverting the values
~books['genre'].isin(['Fiction', 'Non Fiction'])    -> in

0       False
1       False
2       False
3       False
4        True
    ...
345     False
346     False
347     False
348     False
349      True
Name: genre, Length: 350, dtype: bool   -> out

books[books['genre'].isin(['Fiction', 'Non Fiction'])].head()   -> in

                            name                     author  rating    year          genre
0  10-Day Green Smoothie Cleanse                   JJ Smith     4.7    2016    Non Fiction
1  11/22/63: A Novel                           Stephen King     4.6    2011        Fiction
2  12 Rules for Life                     Jordan B. Peterson     4.7    2028    Non Fiction
3  1984 (Signet Classics)                     George Orwell     4.7    2017        Fiction
4  A Dance with Dragons                 George R. R. Martin     4.4    2011        Fiction     -> out

Validating numerical data
books.select_dtypes('number').head()    -> in

    rating  year
0      4.7  2016
1      4.6  2011
2      4.7  2018
3      4.7  2017
4      4.8  2019    -> out

books['year'].min()     -> in

2009    -> out

books['year'].max()     -> in

2019    -> out

sns.boxplot(data=books, x='year')
plt.show()

# year grouped by genre
sns.boxplot(data=books, x='year', y='genre')
plt.show()
'''

# Update the data type of the 2019 column to a float
unemployment["2019"] = unemployment["2019"].astype(float)

# Print the dtypes to check your work
print(unemployment.dtypes)


# Define a Series describing whether each continent is outside of Oceania
not_oceania = ~unemployment["continent"].isin(["Oceania"])

# Print unemployment without records related to countries in Oceania
print(unemployment[~unemployment["continent"].isin(["Oceania"])])


# Print the minimum and maximum unemployment rates during 2021
print(unemployment['2021'].min(), unemployment['2021'].max())

# Create a boxplot of 2021 unemployment rates, broken down by continent
sns.boxplot(data=unemployment, x='2021', y='continent')
plt.show()


'''
Data summarization
Exploring groups of data
- .groupby() groups data by category
- Aggregating function indicates how to summarize grouped data

books.groupby('genre').mean()   -> in

genre             rating           year
Childrens       4.780000    2015.075000
Fiction         4.570229    2013.022901
Non Fiction     4.598324    2013.513966     -> out

Aggregating functions
- Sum:                  .sum()
- Count:              .count()
- Minimum:              .min()
- Maximum:              .max()
- Variance:             .var()
- Standard deviation:   .std()

Aggregating ungrouped data
- .agg() applies aggregating functions across a DataFrame

books.agg(['mean', 'std'])  -> in

        rating             year
mean    4.608571    2013.508571
std     0.226941       3.28471      -> out

Specifying aggregations for columns
books.agg({'rating': ['mean', 'std'], 'year': ['median']})  -> in

        rating        year
mean    4.608571       NaN
std     0.226941       NaN
median       NaN    2013.0  -> out

Named summary columns
books.groupby('genre').agg(mean_rating=('rating', 'mean'), std_rating=('rating', 'std'), median_year=('year', 'median'))    -> in

genre        mean_rating  std_rating   median_year
Childrens       4.780000    0.122370        2015.0
Fiction         4.570229    0.281123        2013.0
Non Fiction     4.598324    0.179411        2013.0  -> out

Visualizing categorical summaries
sns.barplot(data=books, x='genre', y='rating')
plt.show()
'''

# Print the mean and standard deviation of rates by year
print(unemployment.agg(['mean', 'std']))

# Print yearly mean and standard deviation grouped by continent
print(unemployment.groupby('continent').agg(['mean', 'std']))


continent_summary = unemployment.groupby("continent").agg(
    # Create the mean_rate_2021 column
    mean_rate_2021=('2021', 'mean'),
    # Create the std_rate_2021 column
    std_rate_2021=('2021', 'std'),
)
print(continent_summary)


# Create a bar plot of continents and their average unemployment
sns.barplot(data=unemployment, x='continent', y='2021')
plt.show()


''' Data Cleaning and Imputation '''

'''
Addressing missing data
Why is missing data a problem?
- Affects distibutions
* Missing heights of taller students
- Less representative of the population
* Certain groups disproportionately represented, e.g, lacking data on oldest students
- Can result in drawing incorrect conclusions

Data professional's job data
Working_Year            Year the data was obtained                      Float
Designation             Job title                                       String
Experience              Experience level e.g, 'Mid', 'Senior'           String
Employment_Status       Type of employment contract e.g, 'FT', 'PT'     String
Employee_Location       Country of employment                           String
Company_Size            Labels for company size e.g, 'S', 'M', 'L'      String
Remote_Working_Ratio    Percentage of time working remotely             Integer
Salary_USD              Salary in US dollars                            Float

Checking for missing values
print(salaries.isna().sum()) -> in

Working_Year            12
Designation             27
Experience              33
Employment_Status       31
Employee_Location       28
Company_Size            40
Remote_Working_Ratio    24
Salary_USD              60
dtype: int64 -> out

Strategies for addressing missing data
- Drop missing values
* 5% or less of total values
- Impute mean, median, mode
* Depends on distribution and context
- Impute by sub-group
* Different experience levels have different median salary

Dropping missing values
threshold = len(Salaries) * 0.05
print(threshold)    -> in

30  -> out

Dropping missing values
cols_to_drop = salaries.columns[salaries.isna().sum() <= threshold]
print(cols_to_drop)     -> in

Index(['Working_Year', 'Designation', 'Employee_Location', 'Remote_Working_Ratio'], dtype='object')     -> out

salaries.dropna(subset=cols_to_drop, inplace=True)

Imputing a summary statistic
cols_with_missing_values = salaries.columns[salaries.isna().sum() > 0]
print(cols_with_missing_values)     -> in

Index(['Experience', 'Employment_Status', 'Company_Size', 'Salary_USD'], dtype='object')    -> out

for col in cols_with_missing_values[:-1]:
    salaries[col].fillna(salaries[col].mode()[0])

Checking the remaining missing values
print(salaries.isna().sum())    -> in

Working_Year            0
Designation             0
Experience              0
Employment_Status       0
Employee_Location       0
Company_Size            0
Remote_Working_Ratio    0
Salary_USD             41
dtype: int64 -> out

Imputing by sub-group
salaries_dict = salaries.groupby('Experience')['Salary_USD'].median().to_dict()
print(salaries_dict)    -> in

{'Entry': 55380.0, 'Executive': 135439.0, 'Mid': 74173.5, 'Senior': 128903.0}   -> out

salaries['Salary_USD'] = salaries['Salary_USD'].fillna(salaries['Experience'].map(salaries_dict))

No more missing values!
print(salaries.isna().sum())    -> in

Working_Year            0
Designation             0
Experience              0
Employment_Status       0
Employee_Location       0
Company_Size            0
Remote_Working_Ratio    0
Salary_USD              0
dtype: int64 -> out
'''

# Count the number of missing values in each column
print(planes.isna().sum())

# Find the five percent threshold
threshold = len(planes) * 0.05

#  Create a filter
cols_to_drop = planes.columns[planes.isna().sum() <= threshold]

# Drop missing values for columns below the threshold
planes.dropna(subset=cols_to_drop, inplace=True)

print(planes.isna().sum())


# Check the values of the Additional_Info column
print(planes["Additional_Info"].value_counts())

# Create a box plot of Price by Airline
sns.boxplot(data=planes, x='Airline', y='Price')

plt.show()


# Calculate median plane ticket prices by Airline
airline_prices = planes.groupby("Airline")["Price"].median()

print(airline_prices)

# Convert to a dictionary
prices_dict = airline_prices.to_dict()

# Map the dictionary to missing values of Price by Airline
planes["Price"] = planes["Price"].fillna(planes["Airline"].map(prices_dict))

# Check for missing values
print(planes.isna().sum())


'''
Converting and analyzing categorical data
Previewing the data
print(salaries.select_dtypes('object').head()) -> in

                Designation     Experience  Employment_status   Employee_Location   Company_Size
0  Machine Learning Scientist   Senior      FT                  JP                  S
1  Big Data Engineer            Senior      FT                  GB                  M
2  Product Data Analyst         Mid         FT                  HM                  S
3  Machine Learning Engineer    Senior      FT                  US                  L
4  Data Analyst                 Entry       FT                  US                  L  -> out

Job titles
print(salaries['Designation'].value_counts()) -> in

Data Scientist             124
Data Engineer              114
Data Analyst                79
Machine Learning Engineer   34
Research Scientist          16
Data Architect              10
Data Science MAnager         8
Big Data Engineer            8
Data Science Consultant      7
...  -> out

print(salaries['Designation'].nunique()) -> in

50  -> out

Extracting value from categories
- Current format limits our ability to generate insights
- pandas.Series.str.contains()
* Seach a column for a specific string or multiple strings

salaries['Designation'].str.contains('Scientist') -> in

0        True
1       False
2       False
3       False
    ...
515     False
516     False
517      True
Name: Designation, Length: 518, dtype: bool  -> out

Finding multiple phrases in strings
- Words of interest: Machine Learning or AI

salaries['Designation'].str.contains('Machine Learning|AI') -> in

0        True
1       False
2       False
3        True
    ...
515     False
516     False
517     False
Name: Designation, Length: 518, dtype: bool  -> out

job_categories = ['Data Science', 'Data Analytics', 'Data Engineering', 'Machine Learning', 'Managerial', 'Consultant']

data_science = 'Data Scientist|NLP'
data_analyst = 'Analyst|Analytics'
data_engineer = 'Data Engineering|ETL|Architect|Infrastructure'
ml_engineer = 'Machine Learning|ML|Big Data|AI'
manager = 'Manager|Head|Director|Lead|Principal|Staff'
consultant = 'Consultant|Freelance'

conditions = [
    (salaries['Designation'].str.contains(data_science)), 
    (salaries['Designation'].str.contains(data_analyst)),
    (salaries['Designation'].str.contains(data_engineer)),
    (salaries['Designation'].str.contains(ml_engineer)),
    (salaries['Designation'].str.contains(manager)),
    (salaries['Designation'].str.contains(consultant))
    ]

Creating the categorical column
salaries['Job_Category'] = np.select(conditions, job_categories, default='Other')

Previewing job categories
print(salaries[['Designation', 'Job_Category']].head()) -> in

                Designation             Job_Category
0  Machine Learning Scientist       Machine Learning
1  Big Data Engineer                Data Engineering
2  Product Data Analyst               Data Analytics
3  Machine Learning Engineer        Machine Learning
4  Data Analyst                       Data Analytics      -> out

Visualizing job category frequency
sns.countplot(data=salaries, x='Job_Category')
plt.show()
'''

# Filter the DataFrame for object columns
non_numeric = planes.select_dtypes("object")

# Loop through columns
for col in non_numeric.columns:

    # Print the number of unique values
    print(
        f"Number of unique values in {col} column: ", non_numeric[col].nunique())


#  Create a list of categories
flight_categories = ["Short-haul", "Medium", "Long-haul"]

#  Create short-haul values
short_flights = "0h|1h|2h|3h|4h"

# Create medium-haul values
medium_flights = "5h|6h|7h|8h|9h"

# Create long-haul values
long_flights = "10h|11h|12h|13h|14h|15h|16h"


# Create conditions for values in flight_categories to be created
conditions = [
    (planes["Duration"].str.contains(short_flights)),
    (planes["Duration"].str.contains(medium_flights)),
    (planes["Duration"].str.contains(long_flights))
]

# Apply the conditions list to the flight_categories
planes["Duration_Category"] = np.select(
    conditions, flight_categories, default="Extreme duration")

#  Plot the counts of each category
sns.countplot(data=planes, x="Duration_Category")
plt.show()


'''
Working with numeric data
The original salaries dataset
print(salaries.info())  -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 594 entries, 0 to 593
Data columns (total 9 columns):
#  Column                   Non-Null Count      Dtype
0  Working_Year             594 non-null        int64
1  Designation              567 non-null       object
2  Experience               561 non-null       object
3  Employment_Status        563 non-null       object
4  Salary_In_Rupees         566 non-null       object
5  Employee_Location        554 non-null       object
6  Company_Location         570 non-null       object
7  Company_Size             535 non-null       object
8  Remote_Working_Ratio     571 non-null      float64
dtypes: float64(1), int64(1), object(7)
memory usage: 41.9+ KB  -> out

Salary in rupees
print(salaries['Salary_In_Rupees'].head())  -> in

0   20,688,070.00
1    8,674,985.00
2    1,591,390.00
3   11,935,425.00
4    5,729,004.00
Name: Salary_In_Rupees, dtype: object  -> out

Converting strings to numbers
- Remove comma values in 'Salary_In_Rupees'
- Convert the column to 'float' data type
- Create a new column by converting the currency

Converting strings to numbers
pd.Series.str.replace('characters to remove', 'characters to replace them with')

salaries['Salary_In_Rupees'] = salaries['Salary_In_Rupees'].str.replace(',', '')
print(salaries['Salary_In_Rupees'].head())  -> in

0   20688070.00
1    8674985.00
2    1591390.00
3   11935425.00
4    5729004.00
Name: Salary_In_Rupees, dtype: object  -> out

Converting strings to numbers
salaries['Salary_In_Rupees'] = salaries['Salary_In_Rupees'].astype(float)

* 1 Indian Rupee = 0.012 US Dollars

salaries['Salary_USD'] = salaries['Salary_In_Rupees'] * 0.012

Previewing the new column
print(salaries[['Salary_In_Rupees', 'Salary_USD']].head())  -> in

    Salary_In_Rupees    Salary_USD
0   20688070.0          248256.840
1    8674985.0          104099.820
2    1591390.0           19096.680
3   11935425.0          143225.100
4    5729004.0           68748.048  -> out

Adding summary statistics into a DataFrame
salaries.groupby('Company_Size')['Salary_USD'].mean()  -> in

Company_Size
L   111934.432174
M   110706.628527
S    69880.980179
Name: Salary_USD, dtype: float64  -> out

Groupby (Experience)    ->  Select(Salary_USD)  ->  Call transform()    ->  Apply lambda function

salaries['std_dev'] = salaries.groupby('Experience')['Salary_USD'].transform(lambda x: x.std())

print(salaries[['Experience', 'std_dev']].value_counts())   -> in

Experience       std_dev
SE          52995.385395    257
MI          63217.397353    197
EN          43367.256303     83
EX          86426.611619     24     -> out

salaries['median_by_comp_size'] = salaries.groupby('Company_Size')['Salary_USD'].transform(lambda x: x.median())

print(salaries[['Company_size', 'median_by_comp_size']].head())  -> in

    Company_size    median_by_comp_size
0   S                60833.424
1   M               105914.964
2   S                60833.424
3   L                95483.400
4   L                95483.400  -> out
'''

#  Preview the column
print(planes["Duration"].head())

#  Remove the string character
planes["Duration"] = planes["Duration"].str.replace("h", "")

# Convert to float data type
planes["Duration"] = planes["Duration"].astype(float)

#  Plot a histogram
sns.histplot(data=planes, x='Duration')
plt.show()


# Price standard deviation by Airline
planes["airline_price_st_dev"] = planes.groupby(
    "Airline")["Price"].transform(lambda x: x.std())

print(planes[["Airline", "airline_price_st_dev"]].value_counts())

# Median Duration by Airline
planes["airline_median_duration"] = planes.groupby(
    "Airline")["Duration"].transform(lambda x: x.median())

print(planes[["Airline", "airline_median_duration"]].value_counts())

# Mean Price by Destination
planes["price_destination_mean"] = planes.groupby(
    "Destination")["Price"].transform(lambda x: x.mean())

print(planes[["Destination", "price_destination_mean"]].value_counts())


'''
Handling outliers
What is an outlier?
- An observation far away from other data points
* Median house price: $400,000
* Outlier house price: $5,000,000
- Should consider why the value is different: 
* Location, number of bedrooms, overall size etc

Using descriptive statistics
print(salaries['Salary_USD'].describe()) -> in

count      518.000
mean    104905.826
std      62660.107
min       3819.000
25%      61191.000
50%      95483.000
75%     137496.000
max     429675.000
Name: Salary_USD, dtype: float64 -> out

Using the interquartile range
Interquartile range (IQR)
* IQR = 75th - 25th percentile
* Upper Outliers > 75th percentile + (1.5 * IQR)
* Lower Outliers < 25th percentile + (1.5 * IQR)

IQR in box plots
sns.boxplot(data=salaries, y='Salary_USD')
plt.show()

Identifying thresholds
# 75th percentile
seventy_fifth = salaries['Salary_USD'].quantile(0.75)

# 25th percentile
twenty_fifth = salaries['Salary_USD'].quantile(0.25)

# Interquartile range
salaries_iqr = seventy_fifth - twenty_fifth

print(salaries_iqr) -> in

76305.0 -> out

Identifying outliers
# Upper threshold
upper = seventy_fifth + (1.5 * salaries_iqr)

# lower threshold
lower = twenty_fifth - (1.5 * salaries_iqr)

print(upper, lower) -> in

251953.5  -53266.5 -> out

Subsetting our data
salaries[(salaries['Salary_USD'] < lower) | (salaries['Salary_USD'] > upper)][['Experience', 'Employee_Location', 'Salary_USD']] -> in

        Experience      Employee_Location   Salary_USD
29      Mid             US                  429675.0
67      Mid             US                  257805.0
80      Senior          US                  263534.0
83      Mid             US                  429675.0
133     Mid             US                  403895.0
410     Executive       US                  309366.0
441     Senior          US                  362837.0
445     Senior          US                  386708.0
454     Senior          US                  254368.0    -> out

Why look for outliers?
- Outliers are extreme values
* may not accurately represent our data
- Can change the mean and standard deviation
- Statistical tests and machine learning models need normally distributed data

What to do about outliers?
Questions to ask:
Why do these outliers exist?
- More senior roles / different countries pay more
* Consider leaving them in the dataset

Is the data accurate?
- Could there have been an error in data collection?
* If so, remove them

Dropping outliers
no_outliers = salaries[(salaries['Salary_USD'] > lower) & (salaries['Salary_USD'] < upper)]

print(no_outliers['Salary_USD'].describe()) -> in

count      509.000000
mean    100674.567780
std      53643.050057
min       3819.000000
25%      60928.000000
50%      95483.000000
75%     134059.000000
max     248257.000000
Name: Salary_USD, dtype: float64 -> out
'''

#  Plot a histogram of flight prices
sns.histplot(data=planes, x="Price")
plt.show()

# Display descriptive statistics for flight duration
print(planes['Duration'].describe())


# Find the 75th and 25th percentiles
price_seventy_fifth = planes["Price"].quantile(0.75)
price_twenty_fifth = planes["Price"].quantile(0.25)

# Calculate iqr
prices_iqr = price_seventy_fifth - price_twenty_fifth

# Calculate the thresholds
upper = price_seventy_fifth + (1.5 * prices_iqr)
lower = price_twenty_fifth - (1.5 * prices_iqr)

# Subset the data
planes = planes[(planes["Price"] > lower) & (planes["Price"] < upper)]

print(planes["Price"].describe())


''' Relationships in Data '''

'''
Patterns over time
divorce = pd.read_csv('divorce.csv')
divorce.head() -> in

    Marriage_date   marriage_duration
0   2000-06-26       5.0
1   2000-02-02       2.0
2   1991-10-09      10.0
3   1993-01-02      10.0
4   2998-12-11       7.0    -> out

Importing DateTime data
- DateTime data needs to be explicitly declared to Pandas

divorce.dtypes -> in

marriage_date            object
marriage_duration       float64
dtype: object  -> out

divorce = pd.read_csv('divorce.csv', parse_dates=['marriage_date'])
divorce.dtypes -> in

marriage_date           datetime64[ns]
marriage_duration              float64
dtype: object  -> out

Converting to DateTime data
- pd.to_datetime() converts arguments to DateTime data

divorce['marriage_date'] = pd.to_datetime(divorce['marriage_date'])
divorce.dtypes -> in

marriage_date           datetime64[ns]
marriage_duration              float64
dtype: object  -> out

Creating DateTime data
divorce.head(2) -> in

    month   day     year    marriage_duration
0   6       26      2000    5.0
1   2        2      2000    2.0     -> in

divorce['marriage_date'] = pd.to_datetime(divorce[['month', 'day', 'year']])
divorce.head(2) -> in

    month   day     year    marriage_duration   marriage_date
0   6       26      2000    5.0                 2000-06-26
1   2        2      2000    2.0                 2000-02-02      -> in

Creating DateTime data
- Extract parts of a full date using dt.month, dt.day, and dt.year attributes

divorce['marriage_month'] = divorce['marriage_date'].dt.month
divorce.head() -> in

    Marriage_date   marriage_duration   marriage_month
0   2000-06-26       5.0                 6
1   2000-02-02       2.0                 2
2   1991-10-09      10.0                10
3   1993-01-02      10.0                 1
4   2998-12-11       7.0                12      -> out

Visualizing patterns over time
sns.lineplot(data=divorce, x='marriage_month', y='marriage_duration')
plt.show()
'''

# Import divorce.csv, parsing the appropriate columns as dates in the import
divorce = pd.read_csv('divorce.csv', parse_dates=[
                      'divorce_date', 'dob_man', 'dob_woman', 'marriage_date'])
print(divorce.dtypes)


# Convert the marriage_date column to DateTime values
divorce["marriage_date"] = pd.to_datetime(divorce["marriage_date"])


# Define the marriage_year column
divorce["marriage_year"] = divorce["marriage_date"].dt.year

# Create a line plot showing the average number of kids by year
sns.lineplot(data=divorce, x='marriage_year', y='num_kids')
plt.show()


'''
Correlation
- Describes direction and strength of relationship between two variables
- Can help us use variables to predict future outcomes

divorce.corr() -> in

                    income_man      income_woman    marriage_duration   num_kids    marriage_year
income_man          1.000            0.318           0.085               0.041       0.019
income_woman        0.318            1.000           0.079              -0.018       0.026     
marriage_duration   0.085            0.079           1.000               0.447      -0.812
num_kids            0.041           -0.018           0.447               1.000      -0.461
marriage_year       0.019            0.026          -0.812              -0.461       1.000      -> out

- .corr() calculates Pearson correlation coefficient, measuring linear relationship.

Correlation heatmaps
sns.heatmap(divorce.corr(), annot=True)
plt.show()

Correlation in context
divorce['divorce_date'].min() -> in

Timestamp('2000-01-08  00:00:00')  -> out

divorce['divorce_date'].max() -> in

Timestamp('2015-11-03  00:00:00')  -> out

Scatter plots
sns.scatterplot(data=divorce, x='income_man', y='income_woman')
plt.show()

Pairplots
sns.pairplot(data=divorce)
plt.show()

sns.pairplot(data=divorce, vars=['income_man', 'income_woman', 'marriage_duration'])
plt.show()
'''

# Create the scatterplot
sns.scatterplot(data=divorce, x='marriage_duration', y='num_kids')
plt.show()


# Create a pairplot for income_woman and marriage_duration
sns.pairplot(data=divorce, vars=['income_woman', 'marriage_duration'])
plt.show()


'''
Factor relationships and distributions
Level of education: male partner
divorce['education_man'].value_counts() -> in

Professional    1313
Preparatory      501
Secondary        288
Primary          100
None               4
Other              3
Name: education_man, dtype: int64 -> out

Exploring categorical relationships
sns.histplot(data=divorce, x='marriage_duration', hue='education_man', binwidth=1)
plt.show()

Kernel Density Estimate (KDE) plots
sns.kdeplot(data=divorce, x='marriage_duration', hue='education_man', cut=0)
plt.show()

Cumulative KDE plots
sns.kdeplot(data=divorce, x='marriage_duration', hue='education_man', cut=0, cumulative=True)
plt.show()

Relationship between marriage age and education
- Is there a relationship between age at marriage and education level?

divorce['man_age_marriage'] = divorce['marriage_year'] - divorce['dob_man'].dt.year
divorce['woman_age_marriage'] = divorce['marriage_year'] - divorce['dob_woman'].dt.year

Scatter plot with categorical variables
sns.scatterplot(data=divorce, x='woman_age_marriage', y='man_age_marriage', hue='education_man')
plt.show() 
'''

# Create the scatter plot
sns.scatterplot(data=divorce, x='woman_age_marriage',
                y='income_woman', hue='education_woman')
plt.show()


# Update the KDE plot to show a cumulative distribution function
sns.kdeplot(data=divorce, x="marriage_duration",
            hue="num_kids", cut=0, cumulative=True)
plt.show()


''' Turning Exploratory Analysis into Action '''

'''
Considerations for categorical data
Why perform EDA?
- Detecting patterns and relationships
- Generating questions, or hypotheses
- Preparing data for machine learning

Representative data
- Sample represents the population
For example:
- Eduaction versus income in USA
* Can't use data from France

Categorical classes
- Classes = Labels
- Survey people's attitudes towards marriage
- Marital status
* Single
* Married
* Divorced

Class imbalance
It is where one class occurs more frquently than others.

Class frequency
print(planes['Destination'].value_counts()) -> in

Cochin          4391
Banglore        2773
Delhi           1219
New Delhi        888
Hyderabad        673
Kolkata          369
Name: Destination, dtype: int64 -> out

Relative class frequency
- 40% of internal Indian flights have a destination of Delhi

print(planes['Destination'].value_counts(normalize=True)) -> in

Cochin          0.425773
Banglore        0.268884
Delhi           0.118200
New Delhi       0.086105
Hyderabad       0.065257
Kolkata         0.035780
Name: Destination, dtype: float64 -> out

- Is our sample representative of the population (Indian internal flights)?

Cross-tabulation
Call (pd.crosstab()) -> Select column for index -> select column

pd.crosstab(planes['Source'], planes['Destination']) -> in

Destination     Banglore    Cochin      Delhi   Hyderabad   Kolkata     New Delhi
Source
Banglore           0           0        1199      0           0         868  
Chennai            0           0           0      0         364           0
Delhi              0        4318           0      0           0           0
Kolkata         2720           0           0      0           0           0
Mumbai             0           0           0    662           0           0  -> out

Extending cross-tabulation
Source      Destination     Median Price (IDR)
Banglore    Delhi            4232.21
Banglore    New Delhi       12114.56
Chennai     Kolkata          3859.76
Delhi       Cochin           9987.63
Kolkata     Banglore         9654.21
Mumbai      Hyderabd         3431.97  

Aggregated values with pd.crosstab()
pd.crosstab(planes['Source'], planes['Destination'], values=planes['Price'], aggfunc='median') -> in

Destination     Banglore    Cochin      Delhi   Hyderabad   Kolkata     New Delhi
Source
Banglore        NaN         NaN         4823.0  NaN         NaN         10976.5  
Chennai         NaN         NaN         NaN     NaN         3850.0      NaN
Delhi           NaN         10262.0     NaN     NaN         NaN         NaN
Kolkata         9345.0      NaN         NaN     NaN         NaN         NaN
Mumbai          NaN         NaN         NaN     3342.0      NaN         NaN     -> out

Comparing sample to population
Source      Destination     Median Price (IDR)      Median Price (dataset)
Banglore    Delhi            4232.21                 4823.0
Banglore    New Delhi       12114.56                10976.50
Chennai     Kolkata          3859.76                 3850.0
Delhi       Cochin           9987.63                10260.0
Kolkata     Banglore         9654.21                 9345.0
Mumbai      Hyderabd         3431.97                 3342.0
'''

# Print the relative frequency of Job_Category
print(salaries['Job_Category'].value_counts(normalize=True))


# Cross-tabulate Company_Size and Experience
print(pd.crosstab(salaries["Company_Size"], salaries["Experience"]))

# Cross-tabulate Job_Category and Company_Size
print(pd.crosstab(salaries["Job_Category"], salaries["Company_Size"]))

# Cross-tabulate Job_Category and Company_Size
print(pd.crosstab(salaries["Job_Category"], salaries["Company_Size"],
      values=salaries["Salary_USD"], aggfunc="mean"))


'''
Generating new features
Correlation
sns.heatmap(planes.corr(), annot=True)
plt.show()

Viewing datatype
print(planes.dtypes) -> in

Airline                     object
date_of_Journey     datetime64[ns]
Source                      object
Destination                 object
Route                       object
Dep_Time            datetime64[ns]
Arrival_Time        datetime64[ns]
Duration                   float64
Total_Stops                 object
Additional_Info             object
Price                      float64
dtype: object -> out

Total stops
print(planes['Total_Stops'].value_counts()) -> in

1 stop          4107
non-stop        2584
2 stops         1127
3 stops           29
4 stops            1
Name: Total_Stops, dtype: int64  -> out

Cleaning total stops
planes['Total_Stops'] = planes['Total_Stops'].str.replace(' stops', '')
planes['Total_Stops'] = planes['Total_Stops'].str.replace(' stop', '')
planes['Total_Stops'] = planes['Total_Stops'].str.replace('non-stop', '0')
planes['Total_Stops'] = planes['Total_Stops'].astype(int)

Correlation
sns.heatmap(planes.corr(), annot=True)
plt.show()

Extracting month and weekday
planes['month'] = planes['Date_of_Journey'].dt.month
planes['weekday'] = planes['Date_of_Journey'].dt.weekday
print(planes[['month', 'weekday', 'Date_of_Journey']].head()) -> in

    month   weekday     Date_of_Journey
0    9      4           2019-09-06
1   12      3           2019-12-05
2    1      3           2019-01-03
3    6      0           2019-06-24
4   12      1           2019-12-03          -> out

Departure and arrival times
planes['Dep_Hour'] = planes['Dep_Time'].dt.hour
planes['Arrival_Hour'] = planes['Arrival_Time'].dt.hour

Creating categories
print(planes['Price'].describe()) -> in

count       7848.000000
mean        9035.413609
std         4429.822081
min         1759.000000
25%         5228.000000
50%         8355.000000
75%        12373.000000
max        54826.000000
Name: Price, dtype: float64 -> out

Range  Ticket Type
<= 5228  Economy
> 5228 <= 8355  Premium Economy
> 8355 <= 12373  Business Class
> 12373  First Class

Descriptive statistics
twenty_fifth = planes['Price'].quantile(0.25)
median = planes['Price'].median()
seventy_fifth = planes['Price'].quantile(0.75)
maximum = planes['Price'].max()

Labels and bins
labels = ['Economy', 'Premium Economy', 'business Class', 'First Class']
bins = [0, twenty_fifth, median, seventy_fifth, maximum]

pd.cut()
call (pd.cut()) -> pass the data -> set the labels -> provide the bins

planes['Price_Category'] = pd.cut(planes['Price'], labels=labels, bins=bins)

Price categories
    Price       Price_Category
0  13882.0      First Class
1  6218.0       Premium Economy
2  13302.0      First Class
3  3873.0       Economy
4  11087.0  Business Class -> out

Price category by airline
sns.countplot(data=planes, x='Airline', hue='Price_Category')
plt.show()
'''

#  Get the month of the response
salaries["month"] = salaries["date_of_response"].dt.month

# Extract the weekday of the response
salaries["weekday"] = salaries["date_of_response"].dt.weekday

# Create a heatmap
sns.heatmap(salaries.corr(), annot=True)
plt.show()


# Find the 25th percentile
twenty_fifth = salaries["Salary_USD"].quantile(0.25)

# Save the median
salaries_median = salaries["Salary_USD"].median()

# Gather the 75th percentile
seventy_fifth = salaries["Salary_USD"].quantile(0.75)
print(twenty_fifth, salaries_median, seventy_fifth)


# Create salary labels
salary_labels = ["entry", "mid", "senior", "exec"]

# Create the salary ranges list
salary_ranges = [0, twenty_fifth, salaries_median,
                 seventy_fifth, salaries["Salary_USD"].max()]

# Create salary_level
salaries["salary_level"] = pd.cut(
    salaries["Salary_USD"], bins=salary_ranges, labels=salary_labels)

# Plot the count of salary levels at companies of different sizes
sns.countplot(data=salaries, x="Company_Size", hue="salary_level")
plt.show()


'''
Generating hypotheses
What is true?
- Would data from a different time give the same results?
- Detecting relationships, differences and patterns:
* We use Hypothesis Testing
- Hypothesis testing requires, prior to data collection:
* Generating a hypothesis or question
* A decision on what statistical test to use

Data snooping / p-hacking
It is the act of excessive exploratory analysis, the generation of multiple hypotheses, and the execution of multiple statistical tests.

Next steps
- Design our experiment
- Involves steps such as:
* Choosing a sample
* Calculating how many data points we need
* Deciding what statistical test to run
'''

# Filter for employees in the US or GB
usa_and_gb = salaries[salaries["Employee_Location"].isin(["US", "GB"])]

# Create a barplot of salaries by location
sns.barplot(data=usa_and_gb, x="Employee_Location", y="Salary_USD")
plt.show()


# Create a bar plot of salary versus company size, factoring in employment status
sns.barplot(data=salaries, x="Company_Size",
            y="Salary_USD", hue="Employment_Status")
plt.show()
