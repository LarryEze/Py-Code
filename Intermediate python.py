# Import the course packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the two datasets
gapminder = pd.read_csv("Intermediate python/gapminder.csv")

year = np.array(gapminder['year'])
pop = np.array(gapminder['population'])
gdp_cap = np.array(gapminder['gdp_cap'])
life_exp = np.array(gapminder['life_exp'])
continent = np.array(gapminder['cont'])

''' MatPlotLib '''

'''
The most basic plot is the line plot. A general recipe is given here.

import matplotlib.pyplot as plt
plt.plot(x , y)
plt.show()

To create scatter plots
plt.scatter(x , y) 

To create histograms (default no of bins is 10 if its not specified)
plt.hist(x, bins) 

s is the argument used to specify the size of the plot
c is the argument used to specify the colour of the plot
alpha is argument used to change the opacity of the plot ( range = 0 to 1 )

plt.xlabel('x') to label the x-axis
plt.ylabel('y') to label the y-axis
plt.title('Lorem ipsum') to give the plot a title
plt.yticks([0, 2, 4, 6, 8, 10], [0, '2B', '4B', '6B', '8B', '10B']) to rescale the y-axis and also rename the scale digits

plt.text(x, y, 'text') to add text inside the plot
e.g plt.text(5700, 80, 'China') 

plt.grid(True) to add grid lines to the plot

To put the x-axis on a logarithmic scale
plt.xscale('log')
'''

# Print the last item from year and pop
print(year[-1])
print(pop[-1])

# Import matplotlib.pyplot as plt

# Make a line plot: year on the x-axis, pop on the y-axis
plt.plot(year, pop)

# Display the plot with plt.show()
plt.show()


# Print the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Make a line plot, gdp_cap on the x-axis, life_exp on the y-axis
plt.plot(gdp_cap, life_exp)

# Display the plot
plt.show()


# Change the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Put the x-axis on a logarithmic scale
plt.xscale('log')

# Show plot
plt.show()


# import matplotlib.pyplot as plt

# Build Scatter plot
plt.scatter(pop, life_exp)

# Show plot
plt.show()


''' Histogram '''

# Create histogram of life_exp data
plt.hist(life_exp)

# Display histogram
plt.show()


# Build histogram with 5 bins
plt.hist(life_exp, bins=5)

# Show and clean up plot
plt.show()
plt.clf()

# Build histogram with 20 bins
plt.hist(life_exp, bins=20)

# Show and clean up again
plt.show()
plt.clf()


# Histogram of life_exp, 15 bins
plt.hist(life_exp, bins=15)

# Show and clear plot
plt.show()
plt.clf()


life_exp1950 = np.array([28.8, 55.23, 43.08, 30.02, 62.48, 69.12, 66.8, 50.94, 37.48, 68.0, 38.22, 40.41, 53.82, 47.62, 50.92, 59.6, 31.98, 39.03, 39.42, 38.52, 68.75, 35.46, 38.09, 54.74, 44.0, 50.64, 40.72, 39.14, 42.11, 57.21, 40.48, 61.21, 59.42, 66.87, 70.78, 34.81, 45.93, 48.36, 41.89, 45.26, 34.48, 35.93, 34.08, 66.55, 67.41, 37.0, 30.0, 67.5, 43.15, 65.86, 42.02, 33.61, 32.5, 37.58, 41.91, 60.96, 64.03, 72.49, 37.37, 37.47, 44.87, 45.32, 66.91, 65.39, 65.94, 58.53, 63.03, 43.16, 42.27, 50.06,
                        47.45, 55.56, 55.93, 42.14, 38.48, 42.72, 36.68, 36.26, 48.46, 33.68, 40.54, 50.99, 50.79, 42.24, 59.16, 42.87, 31.29, 36.32, 41.72, 36.16, 72.13, 69.39, 42.31, 37.44, 36.32, 72.67, 37.58, 43.44, 55.19, 62.65, 43.9, 47.75, 61.31, 59.82, 64.28, 52.72, 61.05, 40.0, 46.47, 39.88, 37.28, 58.0, 30.33, 60.4, 64.36, 65.57, 32.98, 45.01, 64.94, 57.59, 38.64, 41.41, 71.86, 69.62, 45.88, 58.5, 41.22, 50.85, 38.6, 59.1, 44.6, 43.58, 39.98, 69.18, 68.44, 66.07, 55.09, 40.41, 43.16, 32.55, 42.04, 48.45])

# Histogram of life_exp1950, 15 bins
plt.hist(life_exp1950, bins=15)

# Show and clear plot again
plt.show()
plt.clf()


''' Customization '''

# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log')

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Add title
plt.title(title)

# After customizing, display the plot
plt.show()


# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000, 10000, 100000]
tick_lab = ['1k', '10k', '100k']

# Adapt the ticks on the x-axis
plt.xticks(tick_val, tick_lab)

# After customizing, display the plot
plt.show()


# Store pop as a numpy array: np_pop
np_pop = pop

# Double np_pop
np_pop = np_pop * 2

# Update: set s argument to np_pop
plt.scatter(gdp_cap, life_exp, s=np_pop)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Display the plot
plt.show()


dict = {
    'Asia': 'red',
    'Europe': 'green',
    'Africa': 'blue',
    'Americas': 'yellow',
    'Oceania': 'black'
}

col = [dict[item] for item in continent]

# Specify c and alpha inside plt.scatter()
plt.scatter(x=gdp_cap, y=life_exp, s=(pop * 2), c=col, alpha=0.8)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Show the plot
plt.show()


# Scatter plot
plt.scatter(x=gdp_cap, y=life_exp, s=(pop * 2), c=col, alpha=0.8)

# Previous customizations
plt.xscale('log')
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000], ['1k', '10k', '100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Add grid() call
plt.grid(True)

# Show the plot
plt.show()


''' Dictionaries and Pandas '''

'''
To create a Dictionary
Dictionary = { key : value }
'''

# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# Get index of 'germany': ind_ger
ind_ger = countries.index('germany')

# Use ind_ger to print out capital of Germany
print(capitals[ind_ger])


# Definition of countries and capital
countries = ['spain', 'france', 'germany', 'norway']
capitals = ['madrid', 'paris', 'berlin', 'oslo']

# From string in countries and capitals, create dictionary europe
europe = {'spain': 'madrid', 'france': 'paris',
          'germany': 'berlin', 'norway': 'oslo'}

# Print europe
print(europe)


# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris',
          'germany': 'berlin', 'norway': 'oslo'}

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])


# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris',
          'germany': 'berlin', 'norway': 'oslo'}

# Add italy to europe
europe['italy'] = 'rome'

# Print out italy in europe (Assert italy in europe)
print('italy' in europe)

# Add poland to europe
europe['poland'] = 'warsaw'

# Print europe
print(europe)


# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'bonn',
          'norway': 'oslo', 'italy': 'rome', 'poland': 'warsaw',
          'australia': 'vienna'}

# Update capital of germany
europe['germany'] = 'berlin'

# Remove australia
del (europe['australia'])

# Print europe
print(europe)


# Dictionary of dictionaries
europe = {'spain': {'capital': 'madrid', 'population': 46.77},
          'france': {'capital': 'paris', 'population': 66.03},
          'germany': {'capital': 'berlin', 'population': 80.62},
          'norway': {'capital': 'oslo', 'population': 5.084}}


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital': 'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)


''' Pandas '''

# Pre-defined lists
names = ['United States', 'Australia', 'Japan',
         'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country': names, 'drives_right': dr, 'cars_per_cap': cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)


# Build cars DataFrame
names = ['United States', 'Australia', 'Japan',
         'India', 'Russia', 'Morocco', 'Egypt']
dr = [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = {'country': names, 'drives_right': dr, 'cars_per_cap': cpc}
cars = pd.DataFrame(cars_dict)
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)


# Import the cars.csv data: cars
cars = pd.read_csv('Intermediate python/cars.csv')

# Print out cars
print(cars)


# Fix import by including index_col
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out cars
print(cars)


'''
# Square brackets: limited functionality

# Ideally 2D NumPy arrays
my_array[rows, columns]

Column access
e.g brics[ ["country", "capital"] ]

Row access: only through slicing
e.g brics[1 : 4]

Pandas DataFrame
pandas.DataFrame.query()    -   To query rows based on column value (condition).
pandas.DataFrame.filter()   -   To filter rows by index and columns by name.
pandas.DataFrame.loc[]      -   To select rows by indices label and column by name.
pandas.DataFrame.iloc[]     -   To select rows by index and column by position.
pandas.DataFrame.apply()    -   To custom select using lambda function.

# loc (label-based)
Row access
e.g brics.loc[ ["RU", "IN", "CH"] ]

Column access
e.g brics.loc[ :, ["country", "capital"] ]

Row & Column access
e.g brics.loc[ ["RU", "IN", "CH"], ["country", "capital"] ]

# iloc (integer position-based)
Row access
e.g brics.iloc[ [1, 2, 3] ]

Column access
e.g brics.iloc[ : , [0, 1] ]

Row & Column access
e.g brics.iloc[ [1, 2, 3], [0, 1] ]

Note: The only difference between loc and iloc is that, row_labels are used to select using loc while the position or index is used for iloc.
'''

# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out country column as Pandas Series
print(cars['country'])

# Print out country column as Pandas DataFrame
print(cars[['country']])

# Print out DataFrame with country and drives_right columns
print(cars[['country', 'drives_right']])


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out first 3 observations
print(cars[0: 3])

# Print out fourth, fifth and sixth observation
print(cars[3: 6])


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out observation for Japan as pandas series
print(cars.loc['JPN'])  # or print(cars.iloc[2])

# Print out observations for Australia and Egypt as pandas dataframe
print(cars.loc[['AUS', 'EG']])  # or print(cars.iloc[[1, -1]])


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out drives_right value of Morocco
print(cars.loc[['MOR'], ['drives_right']])

# Print sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])


# Import cars data

cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Print out drives_right column as Series
print(cars.loc[:, 'drives_right'])

# Print out drives_right column as DataFrame
print(cars.loc[:, ['drives_right']])

# Print out cars_per_cap and drives_right as DataFrame
print(cars.loc[:, ['cars_per_cap', 'drives_right']])


'''
LOGIC, CONTROL FLOW & FILTERING
'''

'''
COMPARISON OPERATORS
comparator - Meaning
<    -   Strictly less than
<=   -   Less than or equal to
>    -   Strictly greater than
>=   -   Greater than or equal to
==   -   Equal 
!=   -   Not equal
'''

# Comparison of booleans
print(True == False)

# Comparison of integers
print(-5 * 15 != 75)

# Comparison of strings
print("pyscript" == "PyScript")

# Compare a boolean with an integer
print(True == 1)


# Comparison of integers
x = -3 * 6
print(x >= -10)

# Comparison of strings
y = "test"
print("test" <= y)

# Comparison of booleans
print(True > False)


# Create arrays
# import NumPy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than or equal to 18
print(my_house >= 18)

# my_house less than your_house
print(my_house < your_house)


'''
BOOLEAN OPERATORS
comparator  -   Meaning
AND         -   Both are True
OR          -   At least one  is True
NOT         -   Both are False

NumPy array equivalent
logical_and()
e.g np.logical_and(x > 5, x < 10)

logical_or()
e.g np.logical_or(x > 5, x < 10)

logical_not()
e.g np.logical_not(x > 5, x < 10)
'''

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
print(my_kitchen > 10 and my_kitchen < 18)

# my_kitchen smaller than 14 or bigger than 17?
print(my_kitchen < 14 or my_kitchen > 17)

# Double my_kitchen smaller than triple your_kitchen?
print(my_kitchen * 2 < your_kitchen * 3)


# Create arrays
# import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house > 18.5, my_house < 10))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house < 11, your_house < 11))


'''
CONDITIONAL STATEMENTS
if, elif,  else

Note* The Python code has to be indented with four (4) spaces ( or a tab ) to tell Python what to do in the case the condition succeeds.
To exit the if statement, simply continue with some Python code without the indentation.

if condition :
    expression
e.g  
z = 4
if z % 2 == 0 :          # True
    print("checking " + str(z))
    print("z is even")


if condition : 
    expression
else : 
    expression
e.g  
z = 5
if z % 2 == 0 :          # False
    print("z is even")
else :
    print("z is odd")


if condition : 
    expression
elif condition :
    expression
else : 
    expression
e.g
z = 3
if z % 2 == 0 :
    print("z is divisible by 2")    # False
elif z % 3 == 0 :
    print("z is divisible by 3")     # True
else :
    print("z is neither divisible by 2 nor by 3")
'''

# Define variables
room = "kit"
area = 14.0

# if statement for room
if room == "kit":
    print("looking around in the kitchen.")

# if statement for area
if area > 15:
    print("big place!")


# Define variables
room = "kit"
area = 14.0

# if-else construct for room
if room == "kit":
    print("looking around in the kitchen.")
else:
    print("looking around elsewhere.")

# if-else construct for area
if area > 15:
    print("big place!")
else:
    print("pretty small.")


# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit":
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else:
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15:
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else:
    print("pretty small.")


'''
FILTERING PANDAS DATAFRAMES

Comparison operations
where X is a pandas DataFrame
Y = X["column_name"] > 8
X[Y]
# OR  
X[X["column_name"] > 8]

e.g
is_huge = brics["area"] > 8
brics[is_huge]
# OR
brics[brics["area"] > 8] 


Boolean operations
X[np.logical_and(X["column_name"] > 8, X["column_name"] < 10)]
e.g
import numpy as np
brics[np.logical_and(brics["area"] > 8, brics["area"] < 10)]
'''

# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Extract drives_right column as Series: dr
dr = cars["drives_right"]

# Use dr to subset cars: sel
sel = cars[dr]

# Print sel
print(sel)


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Convert code to a one-liner
sel = cars[cars['drives_right']]

# Print sel
print(sel)


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars["cars_per_cap"]
many_cars = cpc > 500
car_maniac = cars[many_cars]

# Print car_maniac
print(car_maniac)


# Import cars data
cars = pd.read_csv('Intermediate python/cars.csv', index_col=0)

# Import numpy, you'll need this

# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 99, cpc < 501)
medium = cars[between]

# Print medium
print(medium)


''' LOOPS '''

'''
Note* The major difference between if, elif and else statements and the while loop is that, the statements run only once till it meets a condition while the loop runs continuously until a condition is met.
To terminate a while loop while running, use CTRL + C

while loop
while condition :
    expression

e.g
error = 50
while error > 1 :
    error = error / 4
    print(error)
'''


# Initialize offset
offset = 8

# Code the while loop
while offset != 0:
    print("correcting...")
    offset = offset - 1
    print(offset)


# Initialize offset
offset = -6

# Code the while loop
while offset != 0:
    print("correcting...")
    if offset > 0:
        offset = offset - 1
    else:
        offset = offset + 1
    print(offset)


'''
for loop
for var in seq :
    expression

e.g
for c in "family" :
    print(c.capitalize())

fam = [1.73, 1.68, 1.71, 1.89]
for height in fam :
    print(height) # this is without indexing the results
# OR
for index, height in enumerate(fam) :
    print("index " + str(index) + ": " + str(height)) 
'''

# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for a in areas:
    print(a)
# OR
# Change for loop to use enumerate() and update print()
for index, a in enumerate(areas):
    print("room " + str(index) + ": " + str(a))


# house list of lists
house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]

# Build a for loop from scratch
for [a, b] in house:
    print("the " + str(a) + " is " + str(b) + " sqm")


'''
LOOP DATA STRUCTURES

Dictionary
for var in seq :
    expression

e.g
world = { "afghanistan":30.55, "albania":2.77, "algeria":39.21 }
for k, v in world.items() :
    print(k + " -- " + str(v))

NumPy array
in 1D array
e.g
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
bmi = np_weight / np_height ** 2 

for v in bmi :
    print(v)   # will work

in 2D array
e.g
np_height = np.array([1.73, 1.68, 1.71, 1.89, 1.79])
np_weight = np.array([65.4, 59.2, 63.6, 88.4, 68.7])
meas = np.array([np_height, np_weight])

for val in np.nditer(meas) :
    print(val)

Note* Dictionary uses Methods while NumPy 2D array uses Functions to loop over them.
'''

# Definition of dictionary
europe = {'spain': 'madrid', 'france': 'paris', 'germany': 'berlin',
          'norway': 'oslo', 'italy': 'rome', 'poland': 'warsaw', 'austria': 'vienna'}

# Iterate over europe
for k, v in europe.items():
    print("the capital of " + k + " is " + v)


# Import numpy as np
baseball = pd.read_csv('Introduction to python/baseball.csv')

np_height = np.array(baseball['Height'])
np_weight = np.array(baseball['Weight'])
np_baseball = np.column_stack((np_height, np_weight))

# For loop over np_height
for h in np_height:
    print(str(h) + " inches")

# For loop over np_baseball ( 2D NumPy array)
for b in np.nditer(np_baseball):
    print(b)


'''
Pandas DataFrame

e.g
import pandas as pd
brics = pd.read_csv("brics.csv", index_col = 0)

for label, row in brics.iterrows() :
    print(label + ": " + row["capital"])

import pandas as pd
brics = pd.read_csv("brics.csv", index_col = 0)

for label, row in brics.iterrows() :
    # - Creating Series on every iteration
    print(brics.loc[label, "name_length"] = len(row["country"])  # To iterate over dataframes

# OR

import pandas as pd
brics = pd.read_csv("brics.csv", index_col = 0)
brics["name_length"] = brics["country"].apply(len)
print(brics)   # To perform more efficient vectorized calculations
'''

# Import cars data
cars = pd.read_csv('Intermediate python\cars.csv', index_col=0)

# Iterate over rows of cars
for lab, row in cars.iterrows():
    print(lab)
    print(row)


# Import cars data
cars = pd.read_csv('Intermediate python\cars.csv', index_col=0)

# Adapt for loop
for lab, row in cars.iterrows():
    print(lab + ": " + str(row["cars_per_cap"]))


# Import cars data
cars = pd.read_csv('Intermediate python\cars.csv', index_col=0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = row["country"].upper()

# Print cars
print(cars)


# Import cars data
cars = pd.read_csv('Intermediate python\cars.csv', index_col=0)

# Use .apply(str.upper)
cars["COUNTRY"] = cars["country"].apply(str.upper)
print(cars)


''' CASE STUDY: HACKER STATISTICS '''

''' 
Random Numbers

import numpy as np
np.random.rand()    # Pseudo-random numbers

np.random.seed(123)    # Starting from a seed
np.random.rand()    # ensures reproducibility

e.g
import numpy as np
np.random.seed(123)
coin = np.random.randint(0, 2)    # Randomly generate 0 or 1
print(coin) 

if coin == 0 :
    print("heads")
else :
    print("tails")

np.random.randint(start, end)    # The starting number is included while the end number isn't included in the generation just like during list / array subsetting.
'''

# Set the seed
np.random.seed(123)

# Generate and print random float
print(np.random.rand())


# Import numpy and set seed
np.random.seed(123)

# Use randint() to simulate a dice
print(np.random.randint(1, 7))

# Use randint() again
print(np.random.randint(1, 7))


# NumPy is imported, seed is set
np.random.seed(123)

# Starting step
step = 50

# Roll the dice
dice = np.random.randint(1, 7)

# Finish the control construct
if dice <= 2:
    step = step - 1
elif dice <= 5:
    step = step + 1
else:
    step = step + np.random.randint(1, 7)

# Print out dice and step
print(dice, step)


''' 
Random Walk 

Random Walk
e.g
import numpy as np
np.random.seed(123)
outcomes = []

for x in range(10) :
    coin = np.random.randint(0, 2)
    if coin == 0 :
        outcomes.append("heads")
    else :
        outcomes.append("tails")
print(outcomes)    # This will generate a random step

import numpy as np
np.random.seed(123)
tails = [0]

for x in range(10) :
    coin = np.random.randint(0, 2)
    tails.append(tails[x] + coin)
print(tails)    # This will generate a random walk

If you pass max() two arguments, the biggest one gets returned. For example, to make sure that a variable x never goes below 10 when you decrease it. 
Use: x = max(10, x - 1)
'''

# NumPy is imported, seed is set
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

# Complete the for loop
for x in range(100):
    # Set step: last element in random_walk
    step = random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1, 7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    # append next_step to random_walk
    random_walk.append(step)

# Print random_walk
print(random_walk)


# NumPy is imported, seed is set
np.random.seed(123)

# Initialize random_walk
random_walk = [0]

for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        # Replace below: use max to make sure step can't go below 0
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

print(random_walk)


# NumPy is imported, seed is set
np.random.seed(123)

# Initialization
random_walk = [0]

for x in range(100):
    step = random_walk[-1]
    dice = np.random.randint(1, 7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1, 7)

    random_walk.append(step)

# Import matplotlib.pyplot as plt

# Plot random_walk
plt.plot(random_walk)

''' 
The first list you pass is mapped onto the x axis and the second list is mapped onto the y axis.
If you pass only one argument, Python will know what to do and will use the index of the list to map onto the x axis, and the values in the list onto the y axis.
'''

# Show the plot
plt.show()


'''
Distribution

e.g
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
final_tails = []

for x in range(100) :
    tails = [0]
    for x in range(10) :
        coin = np.random.int(0, 2)
        tails.append(tails[x] + coin)
    final_tails.append(tails[-1])
plt.hist(final_tails, bins = 10)
plt.show()
'''

# NumPy is imported; seed is set
np.random.seed(123)

# Initialize all_walks (don't change this line)
all_walks = []

# Simulate random walk 10 times
for i in range(10):

    # Code from before
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)


# numpy and matplotlib imported, seed set.
np.random.seed(123)

'''
import numpy as np
import matplotlib.pyplot as plt
'''

# initialize and populate all_walks
all_walks = []

for i in range(10):
    random_walk = [0]

    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
        random_walk.append(step)
    all_walks.append(random_walk)

# Convert all_walks to NumPy array: np_aw
np_aw = np.array(all_walks)

# Plot np_aw and show
plt.plot(np_aw)
plt.show()

# Clear the figure
plt.clf()

# Transpose np_aw: np_aw_t
np_aw_t = np.transpose(np_aw)

# Plot np_aw_t and show
plt.plot(np_aw_t)
plt.show()


# numpy and matplotlib imported, seed set
np.random.seed(123)

# Simulate random walk 250 times
all_walks = []

for i in range(250):
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)

        # Implement clumsiness
        if np.random.rand() <= 0.001:
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()


# numpy and matplotlib imported, seed set
np.random.seed(123)

# Simulate random walk 500 times
all_walks = []
for i in range(500):
    random_walk = [0]
    for x in range(100):
        step = random_walk[-1]
        dice = np.random.randint(1, 7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1, 7)
        if np.random.rand() <= 0.001:
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1, :]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()
