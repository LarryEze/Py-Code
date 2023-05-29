''' Preparing the data for analysis '''

'''
Stanford Open Policing Project dataset
Introduction to the dataset
- Traffic stops by police officers
- Download data for any state: https://openpolicing.stanford.edu/

Preparing the data
- Examine the data
- Clean the data

import pandas as pd
ri = pd.read_csv('police.csv')
ri.head(3) -> in

    state   stop_date stop_time county_name driver_gender driver_race
0      RI  2005-01-04     12:55         NaN             M       White
1      RI  2005-01-23     23:15         NaN             M       White
2      RI  2005-02-17     04:15         NaN             M       White -> out

- Each row represents one traffic stop
- NaN indicates a missing value 

Locating missing values (1)
ri.isnull() -> in

    state stop_date stop_time county_name driver_gender
0   False     False     False        True         False
1   False     False     False        True         False
2   False     False     False        True         False
... -> out 

Locating missing values (2)
ri.isnull().sum() -> in

state               0
stop_date           0
stop_time           0
county_name     91741
driver_gender    5205 
... -> out

- .sum() calculates the sum of each column
- True = 1, false = 0

Dropping a column
ri.isnull().sum() -> in

state               0
stop_date           0
stop_time           0
county_name     91741
driver_gender    5205 
driver_race      5202
... -> out

ri.shape -> in

(91741, 15) -> out

- county_name column only contains missing values
- Drop county_name using the .drop() method

ri.drop('county_name', axis='columns', inplace=True) 

Dropping rows
- .dropna(): Drop rows based on the presence of missing values

ri.head() -> in

    state   stop_date stop_time driver_gender driver_race
0      RI  2005-01-04     12:55             M       White
1      RI  2005-01-23     23:15             M       White
2      RI  2005-02-17     04:15             M       White
3      RI  2005-02-20     17:15             M       White
4      RI  2005-02-24     01:20             F       White-> out

ri.dropna(subset=['stop_date', 'stop_time'], inplace=True)
'''

# Import the pandas library as pd
import pandas as pd

# Read 'police.csv' into a DataFrame named ri
ri = pd.read_csv('police.csv')

# Examine the head of the DataFrame
print(ri.head())

# Count the number of missing values in each column
print(ri.isnull().sum())


# Examine the shape of the DataFrame
print(ri.shape)

# Drop the 'county_name' and 'state' columns
ri.drop(['county_name', 'state'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(ri.shape)


# Count the number of missing values in each column
print(ri.isnull().sum())

# Drop all rows that are missing 'driver_gender'
ri.dropna(subset=['driver_gender'], inplace=True)

# Count the number of missing values in each column (again)
print(ri.isnull().sum())

# Examine the shape of the DataFrame
print(ri.shape)


'''
Using proper data types
Examining the data types
ri.dtypes -> in

stop_date           object
stop_time           object
driver_gender       object
...                    ...
stop_duration       object
drugs_related_stop    bool 
district            object -> out

- object: Python strings (or other Python objects)
- bool: True and False values
- Other types: int, float, datetime, category

Why do data types matter?
- Affects which operations you can perform
- Avoid storing data as strings (when possible)
* int, float: enables mathematical operations
* datetime: enables date-based attributes and methods
* category: uses less memory and runs faster
* bool: enables logical and mathematical operations

Fixing a data type
apple -> in

        date    time    price
0    2/13/18   16:00   164.34
1    2/14/18   16:00   167.37
2    2/15/18   16:00   172.99 -> out

apple.price.dtype -> in

dtype('o') -> out

apple['price'] = apple.price.astype('float') 

apple.price.dtype -> in

dtype('float64') -> out

- Dot notation: apple.price
- Bracket notation: apple['price']

Note: Bracket notation must be used on the left side of an assignment statement to create a new Series or overwrite an existing Series.
'''

# Examine the head of the 'is_arrested' column
print(ri.is_arrested.head())

# Change the data type of 'is_arrested' to 'bool'
ri['is_arrested'] = ri.is_arrested.astype('bool')

# Check the data type of 'is_arrested' 
print(ri.is_arrested.dtype)


'''
Creating a DatetimeIndex
Using datetime format
ri.head(3) -> in

    stop_date stop_time driver_gender driver_race
0  2005-01-04     12:55             M       White
1  2005-01-23     23:15             M       White
2  2005-02-17     04:15             M       White -> out

ri.dtypes -> in

stop_Date       object
stop_time       object
driver_gender   object
driver_race     object
... -> out

- Combine stop_date and stop_time into one column
- Convert it to datetime format

Combining object columns
apple -> in

        date    time    price
0    2/13/18   16:00   164.34
1    2/14/18   16:00   167.37
2    2/15/18   16:00   172.99 -> out

apple.date.str.replace('/'. '-') -> in

0   2-13-18
1   2-14-18
2   2-15-18 
Name: date, dtype: object -> out

combined = apple.date.str.cat(apple.time, sep = ' ') 

combined -> in

0   2/13/18 16:00
1   2/14/18 16:00
2   2/15/18 16:00
Name: date, dtype: object

Converting to datetime format
apple['date_and_time'] = pd.to_datetime(combined)
apple -> in

        date    time    price         date_and_time
0    2/13/18   16:00   164.34   2018-02-13 16:00:00
1    2/14/18   16:00   167.37   2018-02-14 16:00:00
2    2/15/18   16:00   172.99   2018-02-15 16:00:00 -> out

apple.dtypes -> in

date                    object
time                    object
price                  float64
date_and_time   datetime64[ns] -> out

Setting the index
apple.set_index('date_and_time', inplace=True)
apple

                            date    time    price 
date_and_time
2018-02-13 16:00:00      2/13/18   16:00   164.34 
2018-02-14 16:00:00      2/14/18   16:00   167.37 
2018-02-15 16:00:00      2/15/18   16:00   172.99  -> out

apple.index -> in

DatetimeIndex(['2018-02-13 16:00:00', '2018-02-14 16:00:00', '2018-02-15 16:00:00'], dtype='datetime64[ns]', name='date_and_time', freq=None) -> out
'''

# Concatenate 'stop_date' and 'stop_time' (separated by a space)
combined = ri.stop_date.str.cat(ri.stop_time, sep= ' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame
print(ri.dtypes)


# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace=True)

# Examine the index
print(ri.index)

# Examine the columns
print(ri.columns)


''' Exploring the relationship between gender and policing '''

'''
Do the genders commit different violations?
Counting unique value (1)
- .value_counts(): Couts the unique values in a Series
- Best suited for categorical data

ri.stop_outcome.value_counts() -> in

Citation            77091
Warning              5138
Arrest Driver        2735
No Action             624
N/D                   607
Arrest Passenger      343
Name: stop_outcome, dtype: int64 -> out

Counting unique value (2)
ri.stop_outcome.value_counts().sum() -> in

86536 -> out

ri.shape -> in

(86536, 13) -> out

Expressing counts as proportions
ri.stop_outcome.value_counts(normalize=True) -> in

Citation            0.890855
Warning             0.059351
Arrest Driver       0.031605
No Action           0.007211
N/D                 0.007014
Arrest Passenger    0.003964 -> out

Filtering DataFrame rows
ri.driver_race.value_counts() -> in

White       61870
Black       12285
Hispanic     9727
Asian        2389
Other         265 -> out

white = ri[ri.driver_race == 'white']

white.shape -> in

(61870, 13) -> out

Comparing stop outcomes for two groups
white.stop_outcome.value_counts(normalize = True) -> in

Citation            0.902263
Warning             0.057508
Arrest Driver       0.024018
No Action           0.007031
N/D                 0.006433
Arrest Passenger    0.002748 -> out

asian = ri[ri.driver_race == 'Asian']

asian.stop_outcome.value_counts(normalize=True) -> in

Citation            0.922980
Warning             0.045207
Arrest Driver       0.017581
No Action           0.008372
N/D                 0.004186
Arrest Passenger    0.001674 -> out
'''

# Count the unique values in 'violation'
print(ri.violation.value_counts())

# Express the counts as proportions
print(ri.violation.value_counts(normalize=True))


# Create a DataFrame of female drivers
female = ri[ri.driver_gender == 'F']

# Create a DataFrame of male drivers
male = ri[ri.driver_gender == 'M']

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize=True))

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize=True))


'''
Does gender affect who gets a ticket for speeding?
Filtering by multiple conditions (1)
female = ri[ri.driver_gender == 'F']female.shape -> in

(23774, 13) -> out

Filtering by multiple conditions (2)
female_and_arrested = ri[(ri.driver_gender == 'F') & (ri.is_arrested == True)]

* Each condition is surrounded by parentheses
* Ampersand ( & ) represents the and operator

female_and_arrested.shape -> in

(669, 13) -> out

* Only includes female drivers who were arrested

Filtering by multiple conditions (3)
female_or_arrested = ri[(ri.driver_gender == 'F') | (ri.is_arrested == True)]

* Pipe ( | ) represents the or operator

female_or_arrested.shape -> in

(26183, 13) -> out

* Includes all females
* Includes all drivers who were arrested 

Rules for filtering by multiple conditions
- Ampersand ( & ): only include rows that satisfyboth conditions
- Pipe ( | ): include rows that satisfy either condition
- Each condition must be surrounded by parentheses
- Conditions can check for equality ( == ), inequality ( != ), etc
- Can use more than two conditions

Correlation, not causation
- Analyze the relationship between gender and stop outcome
* Assess whether there is a correlation

- Not going to draw any conclusion about causation
* Would need additional data and expertise
* Exploring relationships only
'''

# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender == 'F') & (ri.violation == 'Speeding')]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender == 'M') & (ri.violation == 'Speeding')]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize = True))

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize = True))


'''
Does gender affect whose vehicle is searched?
Math with Boolean values
ri.isnull().sum() -> in

stop_date       0
stop_time       0
driver_gender   0
driver_race     0
violation_raw   0
... -> out

* True =  1, False = 0

import numpy as np
np.mean([0, 1, 0, 0]) -> in

0.25 -> out

np.mean([False, True, False, False]) -> in

0.25 -> out

* Mean of Boolean Series represents percentage of True values

Taking the mean of a Boolean Series
ri.is_arrested.value_counts(normalize = True) -> in

False   0.964431
True    0.035569 -> out

ri.is_arrested.mean() -> in

0.0355690117407784 -> out

ri.is_arrested.dtype -> in

dtype('bool') -> out

Comparing groups using groupby (1)
* Study the arrest rate by police district

ri.district.unique() -> in

array(['Zone X4', 'Zone K3', 'Zone X1', 'Zone X3', 'Zone K1', 'Zone K2'], dtype=object)

ri[ri.district == 'Zone K1'].is_arrested.mean() -> in

0.024349083895853423 -> out

Comparing groups using groupby (2)
ri[ri.district == 'Zone K2'].is_arrested.mean() -> in

0.030800588834786546 -> out

ri.groupby('district').is_arrested.mean() -> in

district
Zone K1     0.024349
Zone K2     0.030801
Zone K3     0.032311
Zone X1     0.023494
Zone X3     0.034871
Zone X4     0.048038 -> out

Grouping by multiple categories
ri.groupby(['district', 'driver_gender']).is_arrested.mean() -> in

district    driver_gender
Zone K1                 F   0.019169
                        M   0.026588
Zone K2                 F   0.022196
...                   ...        ... -> out

ri.groupby(['driver_gender', 'district']).is_arrested.mean() -> in

driver_gender  district 
F               Zone K1 0.019169
                Zone K2 0.022196
...                 ...      ... -> out
'''

# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize = True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())


# Calculate the search rate for female drivers
print(ri[ri.driver_gender == 'F'].search_conducted.mean())

# Calculate the search rate for male drivers
print(ri[ri.driver_gender == 'M'].search_conducted.mean())

# Calculate the search rate for both groups simultaneously
print(ri.groupby('driver_gender').search_conducted.mean())


# Calculate the search rate for each combination of gender and violation
print(ri.groupby(['driver_gender', 'violation']).search_conducted.mean())

# Reverse the ordering to group by violation before gender
print(ri.groupby(['violation', 'driver_gender']).search_conducted.mean())


'''
Does gender affect who is frisked during a search?
ri.search_conducted.value_counts() -> in

False   83229
True     3307 -> out

ri.search_type.value_counts(dropna = False) -> in

NaN                             83229
Incident to Arrest               1290
Probable Cause                    924
Inventory                         219
Reasonable Suspicion              214
Protective Frisk                  164
Incident to Arrest, Inventory     123
...     -> out

* .values_counts() excludes missing values by default 
* dropna = False displays missing values

Examining the search types
ri.search_type.value_counts() -> in

Incident to Arrest                  1290
Probable Cause                       924
Inventory                            219
Reasonable Suspicion                 214
Protective Frisk                     164
Incident to Arrest, Inventory        123
Incident to Arrest, Probable Cause   100
...     -> out

- Multiple values are separated by commas
- 219 searches in which 'inventory' was the only search type
- Locate 'inventory' among multiple search types

Searching for a string (1)
ri['inventory'] = ri.search_type.str.contains('Inventory', na=False)

* str.contains() returns True if string is found, False if not found
* na=False returns False when it finds a missing value

Searching for a string (2)
ri.inventory.dtype -> in

dtype('bool') -> out

* True means inventory was done, False means it was not

ri.inventory.sum() -> in

441 -> out

Calculating the inventory rate
ri.inventory.mean() -> in

0.0050961449570121106 ->  out

* 0.5% of all traffic stops resulted in an inventory

searched = ri[ri.search_conducted == True]
searched.inventory.mean() -> in

0.13335349259147264 -> out

* 13.3% of searches included an inventory
'''

# Count the 'search_type' values
print(ri.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na=False)

# Check the data type of 'frisk'
print(ri.frisk.dtype)

# Take the sum of 'frisk'
print(ri.frisk.sum())


# Create a DataFrame of stops in which a search was conducted
searched = ri[ri.search_conducted == True]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby('driver_gender').frisk.mean())


''' Visual exploratory data analysis '''

'''
Does time of day affect arrest rate?
Analyzing datetime data
apple -> in

    price       volume         date_and_time
0  174.35     20567800   2018-01-08 16:00:00
1  174.33     21584000   2018-01-09 16:00:00
2  155.15     54390500   2018-02-08 16:00:00
3  156.41     70672600   2018-02-09 16:00:00
4  176.94     23774100   2018-03-08 16:00:00
5  179.98     32185200   2018-03-09 16:00:00 -> out

Accessing datetime attributes (1)
apple.dtypes -> in

price                   float64
volume                    int64
date_and_time    datetime64[ns] -> out

apple.date_and_time.dt.month -> in

0    1
1    1
2    2
3    2
...     -> out

Accessing datetime attributes (2)
apple.set_index('date_and_time', inplace=True)
apple.index -> in

DatetimeIndex(['2018-01-08 16:00:00', '2018-01-09 16:00:00', '2018-02-08 16:00:00', '2018-02-09 16:00:00', '2018-03-08 16:00:00', '2018-03-09 16:00:00'], dtype = 'datetime64[ns]', name = 'date_and_time', freq=None) -> out

apple.index.month -> in

Int64Index([1, 1, 2, 2, 3, 3], dtype = 'int64', name = 'date_and_time') -> out

* dt accessor is not used with a DatetimeIndex

Calculating the monthly mean price
apple.price.mean() -> in

169.52666666666667 -> out

apple.groupby(apple.index.month).price.mean() -> in

date_and_time
1           174.34
2           155.78
3           178.46
Name: price, dtype: float64 -> out

monthly_price = apple.groupby(apple.index.month).price.mean()

Plotting the monthly mean price
import matplotlib.pyplot as plt

monthly_price.plot()

* Line plot: Series index on x-axis, Series values on y-axis

plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Monthly mean stock price for Apple')

plt.show()
'''

# Calculate the overall arrest rate
print(ri.is_arrested.mean())

# Calculate the hourly arrest rate
print(ri.groupby(ri.index.hour).is_arrested.mean())

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby(ri.index.hour).is_arrested.mean()


# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()


'''
Are drug-related stops on the rise?
Resampling the price
apple.groupby(apple.idex.month).price.mean() -> in

date_and_time
1       174.34
2       155.78
3       178.46 -> out

apple.price.resample('M').mean() -> in

date_and_time
2018-01-31      174.34
2018-02-28      155.78
2018-03-31      178.46 -> out

Resampling the volume
apple -> in

date_and_time           price       volume
2018-01-08 16:00:00    174.35     20567800
2018-01-09 16:00:00    174.33     21584000
2018-02-08 16:00:00    155.15     54390500
...                       ...          ... -> out

apple.volume.resample('M').mean() -> in

date_and_time
2018-01-31      21075900
2018-02-28      62531550
2018-03-31      27979650
Freq: M, Name: volume, Length: 3, dtype: float64 -> out

Concatenating price and volume
monthly_price = apple.price.resample('M').mean()
monthly_volume = apple.volume.resample('M').mean()

monthly = pd.concat([monthly_price, monthly_volume], axis='columns') -> in

date_and_time    price      volume
2018-01-31      174.34    21075900
2018-02-28      155.78    62531550
2018-03-31      178.46    27979650 -> out

Plotting price and volume (1)
monthly.plot()
plt.show()

Plotting price and volume (2)
monthly.plot(subplots = True)
plt.show()
'''

# Calculate the annual rate of drug-related stops
print(ri.drugs_related_stop.resample('A').mean())

# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()

# Display the plot
plt.show()


# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate, annual_search_rate], axis='columns')

# Create subplots from 'annual'
annual.plot(subplots = True)

# Display the subplots
plt.show()


'''
What violations are caught in each district?
Computing a frequency table 
pd.crosstab(ri.driver_race, ri.driver_gender) -> in

driver_gender     F         M
driver_race
Asian           551      1838
Black          2681      9604
Hispanic       1953      7774
Other            53       212
White         18536     43334 -> out

- Frequency table: Tally of how many times each combination of values occurs.

Selecting a DataFrame slice
- .loc[] accessor: Select from a DataFrame by label

table = table.loc['Asian' : 'Hispanic'] -> in

driver_gender     F     M
driver_race
Asian           551  1838
Black          2681  9604
Hispanic       1953  7774 -> out

Creating a line plot 
table.plot()
plt.show()

Creating a bar plot
table.plot(kind = 'bar')
plt.show()

Stacking the bars
table.plot(kind = 'bar', stacked =  True)
plt.show()
'''

# Create a frequency table of districts and violations
print(pd.crosstab(ri.district, ri.violation))

# Save the frequency table as 'all_zones'
all_zones = pd.crosstab(ri.district, ri.violation)

# Select rows 'Zone K1' through 'Zone K3'
print(all_zones.loc['Zone K1' : 'Zone K3'])

# Save the smaller table as 'k_zones'
k_zones = all_zones.loc['Zone K1' : 'Zone K3']


# Create a bar plot of 'k_zones'
k_zones.plot(kind = 'bar')

# Display the plot
plt.show()


# Create a stacked bar plot of 'k_zones'
k_zones.plot(kind = 'bar', stacked = True)

# Display the plot
plt.show()


'''
How long might you be stopped for a violation?
Analyzing an object column
apple -> in

date_and_time price volume change
2018-01-08 16:00:00 174.35 20567800 down
... ... ... ...
2018-03-09 16:00:00 179.98 32185200 up -> out

- Create a Boolean column: True if the price went up, and False otherwise
- Calculate how often the price went up by taking the column mean 

apple.change.dtype -> in

dtype('O') - out

- .astype() can't be used in this case

Mapping one set of values to another
- Dictionary maps the values you have to the values you want
mapping = {'up': True, 'down': False}
apple['is_up'] = apple.change,map(mapping)
apple -> in

date_and_time price volume change is_up
2018-01-08 16:00:00 174.35 20567800 down False
... ... ... ... ...
2018-03-09 16:00:00 179.98 32185200 up True -> out

apple.is_up.mean() -> in

0.5 -> out

Calculating the search rate
- Visualize how often searches were done after each violation type

search_rate = ri.groupby('violation').search_conducted.mean() -> in

violation
Equipment               0.064280
Moving violation        0.057014
Other                   0.045362
Registration / plates   0.093438
Seat belt               0.031513
Speeding                0.021560 -> out

Creating a bar plot
search_rate.plot(kind = 'bar')
plt.show()

Ordering the bars (1)
- Order the bars from left to right by size

search_rate.sort_values() -> in

violation
Speeding                0.021560
Seat belt               0.031513
Other                   0.045362
Moving violation        0.057014
Equipment               0.064280
Registration / plates   0.093438 
Name: search_conducted, dtype: float64-> out

Ordering the bars (2)
search_rate.sort_values().plot(kind = 'bar') 
plt.show()

Rotating the bars
search_rate.sort_values().plot(kind = 'barh') 
plt.show()
'''

# Print the unique values in 'stop_duration'
print(ri.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {'0-15 Min':8, '16-30 Min':23, '30+ Min':45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(ri.stop_minutes.unique())


# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(ri.groupby('violation_raw').stop_minutes.mean())

# Save the resulting Series as 'stop_length'
stop_length = ri.groupby('violation_raw').stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values()

# Display the plot
stop_length.sort_values().plot(kind = 'barh')
plt.show()


''' Analyzing the effect of weather on policing '''

'''
Exploring the weather dataset
Introduction to the dataset
weather = pd.read_csv('weather.csv')

weather.head(3) -> in

        STATION         DATE TAVG   TMIN    TMAX  AWND    WSF2   WT01    WT02
0   USW00014765   2005-01-01 44.0     35      53  8.95    25.1    1.0     NaN
1   USW00014765   2005-01-02 36.0     28      44  9.20    14.1    NaN     NaN
2   USW00014765   2005-01-03 49.0     44      53  6.93    17.0    1.0     NaN

    ...     WT11   WT13    WT14    WT15    WT16    WT17    WT18    WT19    WT21    WT22
0   ...      NaN    1.0     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
1   ...      NaN    NaN     NaN     NaN     1.0     NaN     1.0     NaN     NaN     NaN
2   ...      NaN    1.0     NaN     NaN     1.0     NaN     NaN     NaN     NaN     NaN -> out

* TAVG, TMIN, TMAX: Temperature
* AWND, WSF2: Wind speed
* WT01 ... WT22: Bad weather conditions

Examining the wind speed
weather[['AWND', 'WSF2']].head() -> in

    AWND    WSF2
0   8.95    25.1
1   9.40    14.1
2   6.93    17.0
3   6.93    16.1
4   7.83    17.0 -> out

weather[['AWND', 'WSF2']].describe() -> in
        
                AWND           WSF2
count    4017.000000    4017.000000
mean        8.593707      19.274782
std         3.364601       5.623866
min         0.220000       4.900000
25%         6.260000      15.000000
50%         8.050000      17.900000
75%        10.290000      21.900000
max        26.840000      48.100000 -> out 

Creating a box plot
weather[['AWND', 'WSF2']].plot(kind = 'box')
plt.show()

Creating a histogram (1)
weather['WDIFF'] = weather.WSF2 - weather.AWND
weather.WDIFF.plot(kind = 'hist')
plt.show()

Creating a histogram (2)
weather.WDIFF.plot(kind = 'hist', bins=20)
plt.show()
'''

# Read 'weather.csv' into a DataFrame named 'weather'
weather = pd.read_csv('weather.csv')

# Describe the temperature columns
print(weather[['TMIN', 'TAVG', 'TMAX']].describe())

# Create a box plot of the temperature columns
weather[['TMIN', 'TAVG', 'TMAX']].plot(kind='box')

# Display the plot
plt.show()


# Create a 'TDIFF' column that represents temperature difference
weather['TDIFF'] = weather.TMAX - weather.TMIN

# Describe the 'TDIFF' column
print(weather.TDIFF.describe())

# Create a histogram with 20 bins to visualize 'TDIFF'
weather.TDIFF.plot(kind='hist', bins=20)

# Display the plot
plt.show()


'''
Categorizing the weather
Selecting a DataFrame slice (1)
weather.shape -> in

(4017, 28) -> out

weather.columns -> in

Index(['STATION', 'DATE', 'TAVG', 'TMIN', 'TMAX', 'AWND', 'WSF2', 'WT01',  'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT07', 'WT08', 'WT09', 'WT10', 'WT11', 'WT13', 'WT14', 'WT15', 'WT16', 'WT17', 'WT18', 'WT19', 'WT21', 'WT22', 'TDIFF'], dtype = 'object') -> out

Selecting a DataFrame slice (2)
temp = weather.loc[:, 'TAVG' : 'TMAX'] 

temp.shape -> in

(4017, 3) -> out

temp.columns -> in

Index(['TAVG', 'TMIN', 'TMAX'], dtype = 'object') -> out

DataFrame operations
temp.head() -> in

    TAVG TMIN TMAX
0   44.0   35   53
1   36.0   28   44
2   49.0   44   53
3   42.0   39   45
4   36.0   28   43 -> out

temp.sum() -> in

TAVG    63884.0
TMIN   174677.0
TMAX   246116.0 -> out

temp.sum(axis='columns').head() -> in

0   132.0
1   108.0
2   146.0
3   126.0
4   107.0 -> out

Mapping one set of values to another
ri.stop_duration.unique() -> in

array(['0-15 Min', '16-30 Min', '30+ Min'], dtype=object) -> out

mapping = {'0-15 Min': 'short', '16-30 Min': 'medium', '30+ Min': 'long'}
ri['stop_length'] = ri.stop_duration.map(mapping)
ri.stop_length.dtype -> in

dtype('O') -> out

Changing data type from object to category (1)
ri.stop_length.unique() -> in

array(['short', 'medium', 'long'], dtype =  object) -> out

- Category type stores the data more efficiently
- Allows you to specify a logical order for the categories

ri.stop_length.memory_usage(deep = True) -> in

6068041 -> out

Changing data type from object to category (2)
cats = pd.CategoricalDtype(['short', 'medium', 'long'], ordered = True)

ri['stop_length'] = ri.stop_length.astype(cats)

ri.stop_length.memory_usage(deep = True) -> in

779118 -> out

Using oredered categories (1)
ri.stop_length.head() -> in

stop_datetime
2005-01-04 12:55:00     short
2005-01-23 23:15:00     short
2005-02-17 04:15:00     short
2005-02-20 17:15:00    medium
2005-02-24 01:20:00     short
Name: stop_length, dtype: category
Categories (3, object): ['short' < 'medium' < 'long'] -> out

Using ordered categories (2)
ri[ri.stop_length > 'short'].shape -> in

(16959, 16) -> out

ri.groupby('stop_length').is_arrested.mean() -> in

stop_length
short           0.013654
medium          0.093595
long            0.261572
Name: is_arrested, dtype: float64 -> out
'''

# Copy 'WT01' through 'WT22' to a new DataFrame
WT = weather.loc[: , 'WT01':'WT22']

# Calculate the sum of each row in 'WT'
weather['bad_conditions'] = WT.sum(axis = 'columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Create a histogram to visualize 'bad_conditions'
weather.bad_conditions.plot(kind = 'hist')

# Display the plot
plt.show()


# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {0:'good', 1:'bad', 2:'bad', 3:'bad', 4:'bad', 5:'worse', 6:'worse', 7:'worse', 8:'worse', 9:'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping)

# Count the unique values in 'rating'
print(weather.rating.value_counts())


# Specify the logical order of the weather ratings
cats = pd.CategoricalDtype(['good', 'bad', 'worse'], ordered=True)

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype(cats)

# Examine the head of 'rating'
print(weather.rating.head())


'''
Merging datasets
apple -> in

                            date    time    price
date_and_time
2018-02-14 09:30:00      2/14/18    9:30   163.04
2018-02-14 16:00:00      2/14/18   16:00   167.37
2018-02-15 09:30:00      2/15/18    9:30   169.79
2018-02-15 16:00:00      2/15/18   16:00   172.99 -> out

apple.reset_index(inplace  = True)
apple -> in

        date_and_time       date    time    price
0 2018-02-14 09:30:00    2/14/18    9:30   163.04
1 2018-02-14 16:00:00    2/14/18   16:00   167.37
2 2018-02-15 09:30:00    2/15/18    9:30   169.79
3 2018-02-15 16:00:00    2/15/18   16:00   172.99 -> out

Preparing the second DataFrame
high_low -> in

        DATE    HIGH       LOW
0    2/14/18  167.54    162.88
1    2/15/18  173.09    169.00
2    2/16/18  174.82    171.77 -> out

high = high_low[['DATE', 'HIGH']]
high -> in

        DATE    HIGH
0    2/14/18  167.54
1    2/15/18  173.09
2    2/16/18  174.82 -> out

Merging the DataFrames
apple_high = pd.merge(left = apple, right = high, left_on = 'date', right_on = 'DATE', how = 'left')

* left = apple : Left DataFrame
* right = high : Right DataFrame
* left_on = 'date' : Key column in left DataFrame
* right_on = 'DATE' : Key column in right DataFrame
* how = 'left' : Type of join

apple_high -> in

        date_and_time       date    time    price      DATE    HIGH
0 2018-02-14 09:30:00    2/14/18    9:30   163.04   2/14/18  167.54
1 2018-02-14 16:00:00    2/14/18   16:00   167.37   2/14/18  167.54
2 2018-02-15 09:30:00    2/15/18    9:30   169.79   2/15/18  173.09
3 2018-02-15 16:00:00    2/15/18   16:00   172.99   2/15/18  173.09 -> out

Setting the index 
apple_high.set_index('date_and_time', inplace = Ture)
apple_high -> in

                            date     time     price     DATE       HIGH
date_and_time
2018-02-14 09:30:00      2/14/18     9:30    163.04  2/14/18     167.54
2018-02-14 16:00:00      2/14/18    16:00    167.37  2/14/18     167.54
2018-02-15 09:30:00      2/15/18     9:30    169.79  2/15/18     173.09
2018-02-15 16:00:00      2/15/18    16:00    172.99  2/15/18     173.09 -> out
'''

# Reset the index of 'ri'
ri.reset_index(inplace = True)

# Examine the head of 'ri'
print(ri.head())

# Create a DataFrame from the 'DATE' and 'rating' columns
weather_rating = weather[['DATE', 'rating']]

# Examine the head of 'weather_rating'
print(weather_rating.head())


# Examine the shape of 'ri'
print(ri.shape)

# Merge 'ri' and 'weather_rating' using a left join
ri_weather = pd.merge(left=ri, right=weather_rating, left_on='stop_date', right_on='DATE', how='left')

# Examine the shape of 'ri_weather'
print(ri_weather.shape)

# Set 'stop_datetime' as the index of 'ri_weather'
ri_weather.set_index('stop_datetime', inplace=True)


'''
Does weather affect the arrest rate?
Driver gender and vehicle searches
ri.search_conducted.mean() -> in

0.0382153092354627 -> out

ri.groupby('driver_gender').search_conducted.mean() -> in

driver_gender
F               0.019181
M               0.045426 -> out

search_rate = ri.groupby(['violation', 'driver_gender']).search_conducted.mean() -> in

violation               driver_gender
Equipment                           F   0.039984
                                    M   0.071496
Moving violation                    F   0.039257
                                    M   0.061524
Other                               F   0.041018
                                    M   0.046191
Registration / plates               F   0.054924
                                    M   0.108802
Seat belt                           F   0.017301
                                    M   0.035119
Speeding                            F   0.008309
                                    M   0.027885 -> out

type(search_rate) -> in

pandas.core.series.Series -> out

type(search_rate.index) -> in

pandas.core,indexes.multi.MultiIndex -> out

search_rate.loc['Equipment'] -> in

driver_gender
F               0.039984
M               0.071496 -> out

search_rate.loc['Equipment', 'M'] -> in

0.07149643705463182 -> out

Converting a multi-indexed Series to a DataFrame
search_rate.unstack() -> in

driver_gender              F           M
violation
Equipment           0.039984    0.071496
Moving violation    0.039257    0.061524
Other               0.041018    0.046191
...                      ...         ... -> out

type(search_rate.unstack()) -> in

pandas.core.frame.DateFrame -> out

ri.pivot_table(index='violation', columns='driver_gender', values='search_conducted') -> in

driver_gender              F           M
violation
Equipment           0.039984    0.071496
Moving violation    0.039257    0.061524
Other               0.041018    0.046191
...                      ...         ... -> out

- Mean() is the default aggregation function for a pivot table
'''

# Calculate the overall arrest rate
print(ri_weather.is_arrested.mean())

# Calculate the arrest rate for each 'rating'
print(ri_weather.groupby('rating').is_arrested.mean())

# Calculate the arrest rate for each 'violation' and 'rating'
print(ri_weather.groupby(['violation', 'rating']).is_arrested.mean())


# Save the output of the groupby operation from the last exercise
arrest_rate = ri_weather.groupby(['violation', 'rating']).is_arrested.mean()

# Print the 'arrest_rate' Series
print(arrest_rate)

# Print the arrest rate for moving violations in bad weather
print(arrest_rate.loc['Moving violation', 'bad'])

# Print the arrest rates for speeding violations in all three weather conditions
print(arrest_rate.loc['Speeding'])


# Unstack the 'arrest_rate' Series into a DataFrame
print(arrest_rate.unstack())

# Create the same DataFrame using a pivot table
print(ri_weather.pivot_table(index='violation', columns='rating', values='is_arrested'))