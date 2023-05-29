# Import the course packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the four datasets
avocados = pd.read_csv("Data Manipulation with pandas/avocado.csv")
homelessness = pd.read_csv("Data Manipulation with pandas\homelessness.csv")
temperatures = pd.read_csv("Data Manipulation with pandas/temperatures.csv", parse_dates=['date'])
sales = pd.read_csv("Data Manipulation with pandas\walmart.csv")


''' TRANSFORMING DATAFRAMES '''

'''
Introducing DataFrames
- Pandas is built on NumPy and Matplotlib
- Attribute contains no parentheses ( i.e () e.g  table_name.shape )
- Method contains parentheses (i.e () e.g table_name.head() )
- 'count' is the number of non-missing values in each column

table_name.head() - returns the first few rows (the “head” of the DataFrame)
table_name.info() - shows information on each of the columns, such as the data type and number of missing values
table_name.describe() - computes a few summary statistics for numerical columns, like mean and median
table_name.shape - contains a tuple that holds the number of rows and columns of the DataFrame
table_name.values - contains the data values in a 2-dimensional NumPy array
table_name.columns - An index of columns: the column names
table_name.index - An index for the rows: either row numbers or row names

Pandas Philosophy
There should be one -- and preferably only one -- obvious way to do it.
- The Zen of Python by Tim Peters, Item 13
'''

# Print the head of the homelessness data
import numpy as np
import pandas as pd
print(homelessness.head())

# Print information about homelessness
print(homelessness.info())

# Print the shape of homelessness
print(homelessness.shape)

# Print a description of homelessness
print(homelessness.describe())


# Import pandas using the alias pd

# Print the values of homelessness
print(homelessness.values)

# Print the column index of homelessness
print(homelessness.columns)

# Print the row index of homelessness
print(homelessness.index)


''' Sorting and Subsetting '''

'''
Sorting
table_name.sort_values('column_name') -  Arrange in ascending order
table_name.sort_values('column_name', ascending=False) -  Arrange in descending order
table_name.sort_values( ['column_name', 'column_name'], ascending=[True, False] ) -  Arrange multiple variables in ascending and descending order

Subsetting columns
table_name['column_name'] - subset / select just one column
table_name[ ['column_name', 'column_name'] ] - select multiple columns
e.g
cols_to_subset = ['column_name', 'column_name']
table_name[cols_to_subset]

Subsetting rows
The most common way to subset rows is by creating a logical condition to filter against
e.g
table_name[table_name['column_name'] > int]  

table_name[table_name['column_name'] == 'text'] - This will subset the rows based on text data  

table_name[table_name['column_name'] > '2020-01-01'] - This will subset the rows based on dates  

Subsetting based on multiple conditions
is_lab = table_name['column_name'] == 'input_text'
is_brown = table_name['column_name'] == 'input_text'
table_name[is_lab & is_brown]

OR
table_name[ (table_name['column_name'] == 'input_text') & (table_name['column_name'] == 'input_text') ]

OR
is_black_or_brown = table_name['column_name'].isin( ['text', 'text'] )
table_name[is_black_or_brown]
'''

# Sort homelessness by individuals
homelessness_ind = homelessness.sort_values("individuals")

# Print the top few rows
print(homelessness_ind.head())

# Sort homelessness by descending family members
homelessness_fam = homelessness.sort_values("family_members", ascending=False)

# Print the top few rows
print(homelessness_fam.head())

# Sort homelessness by region, then descending family members
homelessness_reg_fam = homelessness.sort_values(['region', 'family_members'], ascending=[True, False])

# Print the top few rows
print(homelessness_reg_fam.head())


# Select the individuals column
individuals = homelessness['individuals']

# Print the head of the result
print(individuals.head())

# Select the state and family_members columns
state_fam = homelessness[['state', 'family_members']]

# Print the head of the result
print(state_fam.head())

# Select only the individuals and state columns, in that order
ind_state = homelessness[['individuals', 'state']]

# Print the head of the result
print(ind_state.head())


# Filter for rows where individuals is greater than 10000
ind_gt_10k = homelessness[homelessness['individuals'] > 10000]

# See the result
print(ind_gt_10k)

# Filter for rows where region is Mountain
mountain_reg = homelessness[homelessness['region'] == 'Mountain']

# See the result
print(mountain_reg)

# Filter for rows where family_members is less than 1000 and region is Pacific
fam_lt_1k_pac = homelessness[(homelessness['family_members'] < 1000) & (homelessness['region'] == 'Pacific')]

# See the result
print(fam_lt_1k_pac)


# Subset for rows in South Atlantic or Mid-Atlantic regions
south_mid_atlantic = homelessness[(homelessness['region'] == 'South Atlantic') | (homelessness['region'] == 'Mid-Atlantic')]

# See the result
print(south_mid_atlantic)

# The Mojave Desert states
canu = ["California", "Arizona", "Nevada", "Utah"]

# Filter for rows in the Mojave Desert states
mojave_homelessness = homelessness[homelessness['state'].isin(canu)]

# See the result
print(mojave_homelessness)


'''
New Columns

Adding a new column
table_name['column_name'] = table_name['column_name'] / x
e.g
dogs['bmi'] = dogs['weight_kg'] / dogs['height_m'] ** 2
'''

# Add total col as sum of individuals and family_members
homelessness['total'] = homelessness['individuals'] + \
    homelessness['family_members']

# Add p_individuals col as proportion of total that are individuals
homelessness['p_individuals'] = homelessness['individuals'] / \
    homelessness['total']

# See the result
print(homelessness)


# Create indiv_per_10k col as homeless individuals per 10k state pop
homelessness["indiv_per_10k"] = 10000 * \
    homelessness['individuals'] / homelessness['state_pop']

# Subset rows for indiv_per_10k greater than 20
high_homelessness = homelessness[homelessness['indiv_per_10k'] > 20]

# Sort high_homelessness by descending indiv_per_10k
high_homelessness_srt = high_homelessness.sort_values(
    ['indiv_per_10k'], ascending=False)

# From high_homelessness_srt, select the state and indiv_per_10k cols
result = high_homelessness_srt[['state', 'indiv_per_10k']]

# See the result
print(result)


''' Aggregating DataFrames '''

'''
Summary statistics

Summarizing numerical / dates data
table_name['column_name'].mean() - mean
table_name['column_name'].median() - median
table_name['column_name'].mode() - mode
table_name['column_name'].min() - minimum
table_name['column_name'].max() - maximum
table_name['column_name'].var() - variance
table_name['column_name'].std() - standard deviation
table_name['column_name'].sum() - sum
table_name['column_name'].quantile() - quantiles
table_name['column_name'].cumsum() - cummulative sum
table_name['column_name'].cummax() - cummulative maximum
table_name['column_name'].cummin() - cummulative minimum
table_name['column_name'].cumprod() - cummulative product

table_name['column_name'].agg(function_name)
e.g
def pct30(column):
    return column.quantile(0.3)

table_name['column_name'].agg(pct30)
table_name[['column_name', 'column_name']].agg(pct30)

###
def pct40(column):
    return column.quantile(0.4)

table_name['column_name'].agg([pct30, pct40])
'''

# Print the head of the sales DataFrame
print(sales.head())

# Print the info about the sales DataFrame
print(sales.info())

# Print the mean of weekly_sales
print(sales['weekly_sales'].mean())

# Print the median of weekly_sales
print(sales['weekly_sales'].median())


# Print the maximum of the date column
print(sales['date'].max())

# Print the minimum of the date column
print(sales['date'].min())


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)


# Print IQR of the temperature_c column
print(sales['temperature_c'].agg(iqr))


# A custom IQR function
def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)


# Update to print IQR of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg(iqr))


# Import NumPy and create custom IQR function
# import numpy as np

def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)


# Update to print IQR and median of temperature_c, fuel_price_usd_per_l, & unemployment
print(sales[["temperature_c", "fuel_price_usd_per_l", "unemployment"]].agg([iqr, np.median]))


sales_1_1 = sales[(sales['store'] == 1) & (sales['department'] == 1)]
# Sort sales_1_1 by date
sales_1_1 = sales_1_1.sort_values('date')

# Get the cumulative sum of weekly_sales, add as cum_weekly_sales col
sales_1_1['cum_weekly_sales'] = sales_1_1['weekly_sales'].cumsum()

# Get the cumulative max of weekly_sales, add as cum_max_sales col
sales_1_1['cum_max_sales'] = sales_1_1['weekly_sales'].cummax()

# See the columns you calculated
print(sales_1_1[["date", "weekly_sales", "cum_weekly_sales", "cum_max_sales"]])


'''
Counting
table_name['column_name'].value_counts(sort = True)
table_name['column_name'].value_counts(normalize = True) - Count as proportions of the total

Dropping duplicate names
table_name.drop_duplicates(subset=['column_name', 'column_name'])
'''

# Drop duplicate store/type combinations
store_types = sales.drop_duplicates(subset=['store', 'type'])
print(store_types.head())

# Drop duplicate store/department combinations
store_depts = sales.drop_duplicates(subset=['store', 'department'])
print(store_depts.head())

# Subset the rows where is_holiday is True and drop duplicate dates
holiday_dates = sales[sales['is_holiday']].drop_duplicates(subset='date')

# Print date col of holiday_dates
print(holiday_dates['date'])


# Count the number of stores of each type
store_counts = store_types['type'].value_counts()

print(store_counts)

# Get the proportion of stores of each type
store_props = store_types['type'].value_counts(normalize=True)
print(store_props)

# Count the number of each department number and sort
dept_counts_sorted = store_depts['department'].value_counts(sort=True)
print(dept_counts_sorted)

# Get the proportion of departments of each number and sort
dept_props_sorted = store_depts['department'].value_counts(sort=True, normalize=True)
print(dept_props_sorted)


'''
Grouped summary statistics
e.g
table_name.groupby('column_name')['column_name'].mean()
table_name.groupby(['column_name', 'column_name'])['column_name'].mean()
table_name.groupby(['column_name', 'column_name'])[['column_name', 'column_name']].mean()
table_name.groupby('column_name')['column_name'].agg([min, max, sum])
'''

# Calc total weekly sales
sales_all = sales["weekly_sales"].sum()

# Subset for type A stores, calc total weekly sales
sales_A = sales[sales["type"] == "A"]["weekly_sales"].sum()

# Subset for type B stores, calc total weekly sales
sales_B = sales[sales["type"] == "B"]["weekly_sales"].sum()

# Subset for type C stores, calc total weekly sales
sales_C = sales[sales["type"] == "C"]["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = [sales_A, sales_B, sales_C] / sales_all
print(sales_propn_by_type)


# Group by type; calc total weekly sales
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Get proportion for each type
sales_propn_by_type = sales_by_type / sum(sales_by_type)
print(sales_propn_by_type)

# From previous step
sales_by_type = sales.groupby("type")["weekly_sales"].sum()

# Group by type and is_holiday; calc total weekly sales
sales_by_type_is_holiday = sales.groupby(['type', 'is_holiday'])['weekly_sales'].sum()
print(sales_by_type_is_holiday)


# Import numpy with the alias np
# import numpy as np

# For each store type, aggregate weekly_sales: get min, max, mean, and median
sales_stats = sales.groupby('type')['weekly_sales'].agg([min, max, np.mean, np.median])

# Print sales_stats
print(sales_stats)

# For each store type, aggregate unemployment and fuel_price_usd_per_l: get min, max, mean, and median
unemp_fuel_stats = sales.groupby('type')[['unemployment', 'fuel_price_usd_per_l']].agg([min, max, np.mean, np.median])

# Print unemp_fuel_stats
print(unemp_fuel_stats)


'''
Pivot tables
e.g
Group by to pivot table
table_name.pivot_table( values = 'column_name', index = 'column_name' )
# This takes the mean values for each column (group) by default

Pivot on two variables
table_name.pivot_table( values = 'column_name', index = 'column_name', columns = 'column_name', fill_value = 0, margins = True )
# fill_value is used to replace NaN values in the pivot table
# margins is used to set the last row and column to contain the mean of all the values in the column or row 

Different statistics
table_name.pivot_table( values = 'column_name', index = 'column_name', aggfunc = np.median )

Multiple statistics
table_name.pivot_table( values = 'column_name', index = 'column_name', aggfunc = [np.mean, np.median] )
'''

# Pivot for mean weekly_sales for each store type
mean_sales_by_type = sales.pivot_table(values='weekly_sales', index='type')

# Print mean_sales_by_type
print(mean_sales_by_type)

# Import NumPy as np

# Pivot for mean and median weekly_sales for each store type
mean_med_sales_by_type = sales.pivot_table(
    values='weekly_sales', index='type', aggfunc=[np.mean, np.median])

# Print mean_med_sales_by_type
print(mean_med_sales_by_type)

# Pivot for mean weekly_sales by store type and holiday
mean_sales_by_type_holiday = sales.pivot_table(
    values='weekly_sales', index='type', columns='is_holiday')

# Print mean_sales_by_type_holiday
print(mean_sales_by_type_holiday)


# Print mean weekly_sales by department and type; fill missing values with 0
print(sales.pivot_table(values='weekly_sales',
    index='department', columns='type', fill_value=0))

# Print the mean weekly_sales by department and type; fill missing values with 0s; sum all rows and cols
print(sales.pivot_table(values="weekly_sales", index="department",
    columns="type", fill_value=0, margins=True))


''' SLICING AND INDEXING DATAFRAMES '''

'''
EXPLICIT INDEXES

setting a column as the index
table_name_ind = table_name.set_index('column_name')

Subsetting with index
table_name[table_name[ ['column_name'].isin(['column_index', 'column_index']) ]
table_name_ind.loc[ ['column_index', 'column_index'] ]
e.g
dogs[ dogs['name'].isin(['Bella', 'Stella']) ]
dogs_ind.loc[ ['Bella', 'Stella'] ]

Multi-level indexes a.k.a Hierarchical indexes
table_name_ind = table_name.set_index( ['column_name', 'column_name'] )

Subsetting Multi_level with index
table_name_ind.loc[ ['column_index', 'column_index'] ]
table_name_ind.loc[ [('column_index', 'column_index'), ('column_index', 'column_index')] ]
e.g
dogs_ind.loc[ ['Labrador', 'Chihuahua'] ] - Subset outer level with a list
dogs_ind.loc[ [('Labrador', 'Brown'), ('Chihuahua', 'Tan')] ] - Subset inner levels with a list of tuples

sorting by index values
table_name.sort_index()

Controlling sort_index
table_name.sort_index(level = ['column_name', 'column_name'], ascending = [True, True] )

Removing an index
table_name_ind.reset_index() - This will undo the index that was set

Dropping an index
table_name_ind.reset_index(drop = True) - This will remove / delete the index column

*Note index are left aligned while table contents are right aligned
'''

# Look at temperatures
print(temperatures)

# Set the index of temperatures to city
temperatures_ind = temperatures.set_index('city')

# Look at temperatures_ind
print(temperatures_ind)

# Reset the temperatures_ind index, keeping its contents
print(temperatures_ind.reset_index())

# Reset the temperatures_ind index, dropping its contents
print(temperatures_ind.reset_index(drop=True))


# Make a list of cities to subset on
cities = ["Moscow", "Saint Petersburg"]

# Subset temperatures using square brackets
print(temperatures[temperatures['city'].isin(cities)])

# Subset temperatures_ind using .loc[]
print(temperatures_ind.loc[cities])


# Index temperatures by country & city
temperatures_ind = temperatures.set_index(['country', 'city'])

# List of tuples: Brazil, Rio De Janeiro & Pakistan, Lahore
rows_to_keep = [('Brazil', 'Rio De Janeiro'), ('Pakistan', 'Lahore')]

# Subset for rows to keep
print(temperatures_ind.loc[rows_to_keep])


# Sort temperatures_ind by index values
print(temperatures_ind.sort_index())

# Sort temperatures_ind by index values at the city level
print(temperatures_ind.sort_index(level='city'))

# Sort temperatures_ind by country then descending city
print(temperatures_ind.sort_index(level=['country', 'city'], ascending=[True, False]))


'''
Slicing and subsetting with .loc and .iloc

Sort the index before slicing a DataFrame
table_name = table_name.set_index( ['column_name', 'column_name'] ).sort_index()

Slicing the outer index level
table_name.loc['column_value':'column_value']

Slicing the outer index level
table_name.loc['column_value':'column_value']   # Note that the last value will be included in the result when using loc to slice pandas dataframes

Slicing the inner index level
table_name.loc[ ('column_value', 'column_value') : ('column_value', 'column_value') ]

Slicing columns
table_name.loc[ :, 'column_name':'column_name')

Slicing twice (i.e rows and columns at same time)
table_name.loc[ ('column_value', 'column_value'):('column_value', 'column_value'), 'column_name':'column_name' ]

Subsetting Days / Dates
table_name = table_name.set_index('date_column').sort_index()
table_name.loc[ 'date_value' : 'date_value' ]

e.g
dogs = dogs.set_index('date_of_birth').sort_index()
print(dogs)

dogs.loc[ '2014-08-25' : '2016-09-16' ]
OR 
dogs.loc[ '2014' : '2016' ] # Slicing by partial dates

Subsetting by row / column number
print( dogs.iloc[ 2:5, 1:4 ] )  # Note that the last value will not be included in the result when using iloc to slice pandas dataframes
'''

# Sort the index of temperatures_ind
temperatures_srt = temperatures_ind.sort_index()

# Subset rows from Pakistan to Russia
print(temperatures_srt.loc['Pakistan': 'Russia'])

# Try to subset rows from Lahore to Moscow
print(temperatures_srt.loc['Lahore': 'Moscow'])

# Subset rows from Pakistan, Lahore to Russia, Moscow
print(temperatures_srt.loc[('Pakistan', 'Lahore'): ('Russia', 'Moscow')])


# Subset rows from India, Hyderabad to Iraq, Baghdad
print(temperatures_srt.loc[('India', 'Hyderabad'): ('Iraq', 'Baghdad')])

# Subset columns from date to avg_temp_c
print(temperatures_srt.loc[:, 'date': 'avg_temp_c'])

# Subset in both directions at once
print(temperatures_srt.loc[('India', 'Hyderabad'): ('Iraq', 'Baghdad'), 'date':'avg_temp_c'])


# Use Boolean conditions to subset temperatures for rows in 2010 and 2011
temperatures_bool = temperatures[(
    temperatures['date'] >= '2010-01-01') & (temperatures['date'] <= '2011-12-31')]
print(temperatures_bool)

# Set date as the index and sort the index
temperatures_ind = temperatures.set_index('date').sort_index()

# Use .loc[] to subset temperatures_ind for rows in 2010 and 2011
print(temperatures_ind.loc['2010': '2011'])

# Use .loc[] to subset temperatures_ind for rows from Aug 2010 to Feb 2011
print(temperatures_ind.loc['2010-08': '2011-02'])


# Get 23rd row, 2nd column (index 22, 1)
print(temperatures.iloc[22, 1])

# Use slicing to get the first 5 rows
print(temperatures.iloc[0:5])

# Use slicing to get columns 3 to 4
print(temperatures.iloc[:, 2:4])

# Use slicing in both directions at once
print(temperatures.iloc[0:5, 2:4])


'''
Working with pivot tables
Pivot tables have a default mean aggregation function and are just DataFrames with sorted indexes

table_name = table_name.pivot_table( 'column_to_aggregate', index = 'column_to_groupby_in_rows', columns = 'column_to_groupby_in_columns' )

Slicing pivot tables
table_name.loc[ 'column_value' : 'column_value' ]

The axis argument
table_name.mean(axis = 'index')

Calculating summary statistics across columns
table_name.mean(axis = 'columns')

# You can access the components of a date (year, month and day) using code of the form dataframe["column"].dt.component.
For example, 
    the month component is dataframe["column"].dt.month, and the year component is dataframe["column"].dt.year
'''

# Add a year column to temperatures
temperatures['year'] = temperatures['date'].dt.year

# Pivot avg_temp_c by country and city vs year
temp_by_country_city_vs_year = temperatures.pivot_table(
    'avg_temp_c', index=['country', 'city'], columns='year')

# See the result
print(temp_by_country_city_vs_year)


# Subset for Egypt to India
temp_by_country_city_vs_year.loc['Egypt': 'India']

# Subset for Egypt, Cairo to India, Delhi
temp_by_country_city_vs_year.loc[('Egypt', 'Cairo'): ('India', 'Delhi')]

# Subset for Egypt, Cairo to India, Delhi, and 2005 to 2010
temp_by_country_city_vs_year.loc[(
    'Egypt', 'Cairo'):('India', 'Delhi'), 2005:2010]


# Get the worldwide mean temp by year
mean_temp_by_year = temp_by_country_city_vs_year.mean()

# Filter for the year that had the highest mean temp
print(mean_temp_by_year.loc[mean_temp_by_year >= mean_temp_by_year.max()])

# Get the mean temp by city
mean_temp_by_city = temp_by_country_city_vs_year.mean(axis='columns')

# Filter for the city that had the lowest mean temp
print(mean_temp_by_city.loc[mean_temp_by_city <= mean_temp_by_city.min()])


'''  Creating and visualizing DataFrames '''

'''
Visualizing your data

Histograms
import matplotlib.pyplot as plt
table_name['column_name'].hist( bins = int )
plt.show()

Bar plots
table_name_2 = table_name_1.groupby('column_name')['column_name'].mean
print(table_name_2)
table_name_2.plot( kind = 'bar', title = 'plot_title' )
plt.show()

Line plots
table_name.head()
table_name.plot( x = 'column_name', y = 'column_name', kind = 'line', rot = 45)
plt.show()

# rot = rotate the x-axis label

Scatter plots
table_name.plot( x = 'column_name', y = 'column_name', kind = 'scatter')

Layering plots
table_name[table_name['column_name'] == 'column_value_1']['column_name'].hist(alpha = float)
table_name[table_name['column_name'] == 'column_value_2 ']['column_name'].hist(alpha = float)
plt.legend([ 'column_value_1', 'column_value_2' ])
plt.show()
'''

# Import matplotlib.pyplot with alias plt

# Look at the first few rows of data
print(avocados.head())

# Get the total number of avocados sold of each size
nb_sold_by_size = avocados.groupby('size')['nb_sold'].sum()

# Create a bar plot of the number of avocados sold by size
nb_sold_by_size.plot(kind='bar')

# Show the plot
plt.show()


# Import matplotlib.pyplot with alias plt

# Get the total number of avocados sold on each date
nb_sold_by_date = avocados.groupby('date')['nb_sold'].sum()

# Create a line plot of the number of avocados sold by date
nb_sold_by_date.plot(kind='line')

# Show the plot
plt.show()


# Scatter plot of avg_price vs. nb_sold with title
avocados.plot(x='nb_sold', y='avg_price',
              title='Number of avocados sold vs. average price', kind='scatter')

# Show the plot
plt.show()


# Histogram of conventional avg_price
avocados[avocados['type'] == 'conventional']['avg_price'].hist(
    alpha=0.5, bins=20)

# Histogram of organic avg_price
avocados[avocados['type'] == 'organic']['avg_price'].hist(alpha=0.5, bins=20)

# Add a legend
plt.legend(['conventional', 'organic'])

# Show the plot
plt.show()


'''
Missing Values

Detecting missing values
table_name.isna()

Detecting any missing values 
table_name.isna().any()
# This tells if there is at least a missing value in any column and also what column it is

Counting missing values
table_name.isna().sum()

Plotting missing values
import matplotlib.pyplot as plt
table_name.isna().sum().plot( kind = 'plot_type' )
plt.show()

Removing missing values
table_name.dropna()

Replacing missing values
table_name.fillna(0)
'''

# Import matplotlib.pyplot with alias plt
avocados_2016 = avocados[avocados['year'] == 2016]

# Check individual values for missing values
print(avocados_2016.isna())

# Check each column for missing values
print(avocados_2016.isna().any())

# Bar plot of missing values by variable
avocados_2016.isna().sum().plot(kind='bar')

# Show plot
plt.show()


# Remove rows with missing values
avocados_complete = avocados_2016.dropna()

# Check if any columns contain missing values
print(avocados_complete.isna().any())


# List the columns with missing values
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]

# Create histograms showing the distributions cols_with_missing
avocados_2016[cols_with_missing].hist()

# Show the plot
plt.show()


# From previous step
cols_with_missing = ["small_sold", "large_sold", "xl_sold"]
avocados_2016[cols_with_missing].hist()
plt.show()

# Fill in missing values with 0
avocados_filled = avocados_2016.fillna(0)

# Create histograms of the filled columns
avocados_filled[cols_with_missing].hist()

# Show the plot
plt.show()


'''
Creating DataFrames

Dictionaries
dict_name = { 'key1': value1, 'key2': value2, 'key3': value3 }

2 ways to create a DataFrame
From a list of dictionaries
# Constructed row by row
list_ of_dicts_name = [ { 'key1': value1, 'key2': value2, 'key3': value3 }, { 'key4': value4, 'key5': value5, 'key6': value6 } ]
DataFrame_name = pd.DataFrame(list_ of_dicts_name)
print(DataFrame_name)

From a dictionary of lists
# Constructed column by column
dict_of_lists = { 'key1': ['value1', 'value1'], 'key2': ['value2', 'value2'], 'key3': ['value3', 'value3'], 'key4': ['value4', 'value4'], 'key5': ['value5', 'value5'], }
DataFrame_name = pd.DataFrame(dict_of_lists)
print(DataFrame_name)
'''

# Create a list of dictionaries with new data
avocados_list = [
    {'date': "2019-11-03", 'small_sold': 10376832, 'large_sold': 7835071},
    {'date': "2019-11-10", 'small_sold': 10717154, 'large_sold': 8561348},
]

# Convert list into DataFrame
avocados_2019 = pd.DataFrame(avocados_list)

# Print the new DataFrame
print(avocados_2019)


# Create a dictionary of lists with new data
avocados_dict = {
    "date": ['2019-11-17', '2019-12-01'],
    "small_sold": [10859987, 9291631],
    "large_sold": [7674135, 6238096]
}

# Convert dictionary into DataFrame
avocados_2019 = pd.DataFrame(avocados_dict)

# Print the new DataFrame
print(avocados_2019)


'''
Reading and writing CSVs
CSV = Comma Separated values
- Designed for DataFrame-like data
- Most database and spreadsheet programs can use them or create them

CSV to DataFrame
file_name.csv    -    File path /  url
import pandas as pd
Dataframe_name = pd.read_csv('file_name.csv')
print(Dataframe_name)

DataFrame to CSV
Dataframe_name.to_csv('file_name.csv')
'''

# Read CSV as DataFrame called airline_bumping
airline_bumping = pd.read_csv('airline_bumping.csv')

# Take a look at the DataFrame
print(airline_bumping.head())

# For each airline, select nb_bumped and total_passengers and sum
airline_totals = airline_bumping.groupby(
    'airline')[['nb_bumped', 'total_passengers']].sum()

# Create new col, bumps_per_10k: no. of bumps per 10k passengers for each airline
airline_totals["bumps_per_10k"] = airline_totals["nb_bumped"] / \
    airline_totals["total_passengers"] * 10000

# Print airline_totals
print(airline_totals)


# Create airline_totals_sorted
airline_totals_sorted = airline_totals.sort_values(
    'bumps_per_10k', ascending=False)

# Print airline_totals_sorted
print(airline_totals_sorted)

# Save as airline_totals_sorted.csv
airline_totals_sorted.to_csv("airline_totals_sorted.csv")