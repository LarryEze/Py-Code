# Importing course packages; you can add more too!
import numpy
import math
from math import radians
import math  # import math
from math import radians  # from math import radians
import numpy as np


# Import columns as numpy arrays
baseball_names = np.genfromtxt(
    fname="Introduction to python/baseball.csv",  # This is the filename
    delimiter=",",  # The file is comma-separated
    usecols=[0],  # Use the first column
    skip_header=1,  # Skip the first line
    dtype=str,  # This column contains strings
)
baseball_teams = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[1], skip_header=1
)
baseball_positions = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[2], skip_header=1
)
baseball_heights = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[3], skip_header=1
)
baseball_weights = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[4], skip_header=1
)
baseball_ages = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[5], skip_header=1
)
baseball_posCategory = np.genfromtxt(
    fname="Introduction to python/baseball.csv", delimiter=",", usecols=[6], skip_header=1
)


soccer_names = np.genfromtxt(
    fname="Introduction to python/soccer.csv",
    delimiter=",",
    usecols=[1],
    skip_header=1,
    dtype=str,
    encoding="utf",
)
soccer_ratings = np.genfromtxt(
    fname="Introduction to python/soccer.csv",
    delimiter=",",
    usecols=[2],
    skip_header=1,
    encoding="utf",
)
soccer_positions = np.genfromtxt(
    fname="Introduction to python/soccer.csv",
    delimiter=",",
    usecols=[3],
    skip_header=1,
    encoding="utf",
    dtype=str,
)
soccer_heights = np.genfromtxt(
    fname="Introduction to python/soccer.csv",
    delimiter=",",
    usecols=[4],
    skip_header=1,
    encoding="utf",
)
soccer_shooting = np.genfromtxt(
    fname="Introduction to python/soccer.csv",
    delimiter=",",
    usecols=[8],
    skip_header=1,
    encoding="utf",
)


""" PYTHON BASICS """

# In-line comments

""" 
Multi-line comments 
"""

# Print the sum of 7 and 10
print(7 + 10)

# Addition, subtraction
print(5 + 5)
print(5 - 5)

# Multiplication, division, modulo, and exponentiation
print(3 * 5)
print(10 / 2)
print(18 % 7)
print(4**2)

# How much is your $100 worth after 7 years?
print(100 * 1.1**7)


""" 
To check the type of a variable
type(variable name)

DATA TYPES
float   -   They are real numbers with integer (whole number) and fractional (decimals) parts
int     -   They are integers (whole numbers)
str     -   They are used to represent strings (text)
bool    -   They can either be True or False (boolean)

list    -   They are a compound data type (contain one or more different data types) which can be used to group values together
"""


# Create a variable savings
savings = 100

# Create a variable growth_multiplier
growth_multiplier = 1.1

# Calculate result
result = savings * growth_multiplier**7

# Print out result
print(result)


savings = 100
growth_multiplier = 1.1
desc = "compound interest"

# Assign product of savings and growth_multiplier to year1
year1 = savings * growth_multiplier

# Print the type of year1
print(type(year1))

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

# Print out doubledesc
print(doubledesc)


# Definition of savings and result
savings = 100
result = 100 * 1.10**7

# Fix the printout
print(
    "I started with $" + str(savings) + " and now have $" +
    str(result) + ". Awesome!"
)

# Definition of pi_string
pi_string = "3.1415926"

# Convert pi_string into float: pi_float
pi_float = float(pi_string)


""" PYTHON LIST """

# area variables (in square meters)
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]

# Print areas
print(areas)

# Adapt list areas
areas = [
    "hallway",
    hall,
    "kitchen",
    kit,
    "living room",
    liv,
    "bedroom",
    bed,
    "bathroom",
    bath
]

# Print areas
print(areas)

# house information as list of lists
house = [
    ["hallway", hall],
    ["kitchen", kit],
    ["living room", liv],
    ["bedroom", bed],
    ["bathroom", bath]
]

# Print out house
print(house)

# Print out the type of house
print(type(house))


"""
list slicing
    [start : end]
inclusive    exclusive
"""

# Create the areas list
areas = [
    "hallway",
    11.25,
    "kitchen",
    18.0,
    "living room",
    20.0,
    "bedroom",
    10.75,
    "bathroom",
    9.50,
]

# Print out second element from areas
print(areas[1])

# Print out last element from areas
print(areas[-1])

# Print out the area of the living room
print(areas[5])

# Sum of kitchen and bedroom area: eat_sleep_area
eat_sleep_area = areas[3] + areas[7]

# Print the variable eat_sleep_area
print(eat_sleep_area)

# Use slicing to create downstairs
downstairs = areas[0:6]

# Use slicing to create upstairs
upstairs = areas[6:10]

# Print out downstairs and upstairs
print(downstairs, upstairs)

# Alternative slicing to create downstairs
downstairs = areas[:6]

# Alternative slicing to create upstairs
upstairs = areas[-4:]


# Create the areas list
areas = [
    "hallway",
    11.25,
    "kitchen",
    18.0,
    "living room",
    20.0,
    "bedroom",
    10.75,
    "bathroom",
    9.50,
]

# Correct the bathroom area
areas[-1] = 10.50

# Change "living room" to "chill zone"
areas[4] = "chill zone"


# Create the areas list and make some changes
areas = [
    "hallway",
    11.25,
    "kitchen",
    18.0,
    "chill zone",
    20.0,
    "bedroom",
    10.75,
    "bathroom",
    10.50,
]

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# Add garage data to areas_1, new list is areas_2
areas_2 = areas_1 + ["garage", 15.45]


"""
To delete list elements
e.g x = ["a", "b", "c", "d"]
del(x[1])

The ; sign is used to place commands on the same line. The following two code chunks are equivalent:
# Same line
command1; command2

# Separate lines
command1
command2
"""

# Create the areas list
areas = [
    "hallway",
    11.25,
    "kitchen",
    18.0,
    "chill zone",
    20.0,
    "bedroom",
    10.75,
    "bathroom",
    10.50,
    "poolhouse",
    24.5,
    "garage",
    15.45,
]

# Delete the poolhouse and its dimension
del areas[-4:-2]


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Create areas_copy
areas_copy = areas[:]  # or can also code as areas_copy = list(areas)

# Change areas_copy
areas_copy[0] = 5.0

# Print areas
print(areas)


""" FUNCTIONS AND PACKAGES """

"""
Functions are piece of reusable codes used to perform specific / particular task

e.g 
type()                                      -   To check the variable type
max()                                       -   Maximum value
min()                                       -   Minimum value
len()                                       -   Length of / count Variable
round(number, ndigits = None)               -   Approximate value
sorted(iterable, reverse = True / False)    -   Return a new list containing all items from the iterable in ascending order.
help()                                      -   See documentation of function

The general recipe for calling functions and saving the result to a variable is thus:
output = function_name(input)

* Iterable as being any collection of objects, e.g., a List.
"""

# Create variables var1 and var2
var1 = [1, 2, 3, 4]
var2 = True

# Print out type of var1
print(type(var1))

# Print out length of var1
print(len(var1))

# Convert var2 to an integer: out2
out2 = int(var2)


# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse=True)

# Print out full_sorted
print(full_sorted)


"""
Methods are functions that belong to (python variable) objects and they are called using dot notation

str     -   capitalize(), upper(), count(), replace(), index()
float   -   bit_length(), conjugate()
list    -   index(), count(), append(), reverse()
"""

# string to experiment with: place
place = "poolhouse"

# Use upper() on place: place_up
place_up = place.upper()

# Print out place and place_up
print(place, place_up)

# Print out the number of o's in place
print(place.count("o"))


# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 9.50 appears in areas
print(areas.count(9.50))

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)


"""
Packages are directory of python scripts where each script represents modules which specify functions, methods and new python types aimed at solving particular problems

http://pip.readthedocs.org/en/stable/installing/

TensorFlow
SciPy                   -   (scientific computing)
NumPy                   -   (Numeric Python)  
Pandas                  -   (data analysis)
Matplotlib              -   (data visualization)
Keras
Scikit-learn            -   (machine learning)
PyTorch
ScraPy
BeautifulSoup
"""

# Definition of radius
r = 0.43

# Import the math package

# Calculate Circumference of a Circle
C = 2 * math.pi * r

# Calculate Area of a Circle
A = math.pi * r**2

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))


# Definition of radius
r = 192500

# Import radians function of math package

# Travel distance of Moon over 12 degrees. Store in dist.
dist = r * radians(12)

# Print out dist
print(dist)


""" NumPy """

"""
To install numpy
pip3 install numpy

To import numpy
import numpy as np

* NumPy array can only contain objects of 1 data type
"""

# Create list baseball
baseball = [180, 215, 210, 210, 188, 176, 209, 200]

# Create a numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out type of np_baseball
print(type(np_baseball))


# height_in is available as a regular list
height_in = baseball_heights

# Create a numpy array from height_in: np_height_in
np_height_in = np.array(height_in)

# Print out np_height_in
print(np_height_in)

# Convert np_height_in to m: np_height_m
np_height_m = np_height_in * 0.0254

# Print np_height_m
print(np_height_m)


# height_in and weight_lb are available as regular lists
height_in = baseball_heights
weight_lb = baseball_weights

# Create array from height_in with metric units: np_height_m
np_height_m = np.array(height_in) * 0.0254

# Create array from weight_lb with metric units: np_weight_kg
np_weight_kg = np.array(weight_lb) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg / np_height_m ** 2

# Print out bmi
print(bmi)


# Calculate the BMI: bmi
np_height_m = np.array(height_in) * 0.0254
np_weight_kg = np.array(weight_lb) * 0.453592
bmi = np_weight_kg / np_height_m ** 2

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])


# Store weight and height lists as numpy arrays
np_weight_lb = np.array(weight_lb)
np_height_in = np.array(height_in)

# Print out the weight at index 50
print(np_weight_lb[50])

# Print out sub-array of np_height_in: index 100 up to and including index 110
print(np_height_in[100:111])


# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the type of np_baseball
print(type(np_baseball))

# Print out the shape of np_baseball
print(np_baseball.shape)


# baseball is available as a regular list of lists
baseball = np.column_stack((height_in, weight_lb))

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the shape of np_baseball
print(np_baseball.shape)


# Create np_baseball (2 cols)
np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball
print(np_baseball[49, :])

# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb = np_baseball[:, 1]

# Print out height of 124th player
print(np_baseball[123, 0])


# baseball is available as a regular list of lists
age = baseball_ages

# Column binding the arrays
baseball = np.column_stack((height_in, weight_lb, age))

# Finding the minimum and maximum values
min_val = min(min(row) for row in baseball)
max_val = max(max(row) for row in baseball)

# Applying min-max normalization
updated = [[(val - min_val) / (max_val - min_val) for val in row]
           for row in baseball]

# Import numpy package

updated = np.array(updated)

# Create np_baseball (3 cols)
np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball + updated)

# Create numpy array: conversion
conversion = np.array([0.0254, 0.453592, 1])

# Print out product of np_baseball and conversion
print(np_baseball * conversion)


"""
To get the Mean / average of your data, use np.mean()
To get the Median / middle of your data, use np.median()
To get the Standard Deviation of your data, use np.std()
To check if your data is correlated, use np.corrcoef()

To Generate data, use np.random.normal()

Arguments for np.random.normal()
-     Distribution mean
-     Distribution Standard Deviation
-     Number of samples

e.g 
height = np.round(np.random.normal(1.75, 0.20, 5000), 2)
weight = np.round(np.random.normal(60.32, 15, 5000), 2)
np_city = np.column_stack((height, weight))

np.column_stack() is used to stack arrays as columns 
e.g np_city = np.column_stack((height, weight))
"""

# Create np_height_in from np_baseball
np_height_in = np_baseball[:, 0]

# Print out the mean of np_height_in
print(np.mean(np_height_in))

# Print out the median of np_height_in
print(np.median(np_height_in))


# Print mean height (first column)
avg = np.mean(np_baseball[:, 0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:, 0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:, 0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:, 0], np_baseball[:, 1])
print("Correlation: " + str(corr))


# Convert positions and heights to numpy arrays: np_positions, np_heights
np_heights = np.array(soccer_heights)
np_positions = np.array(soccer_positions)

# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions == "GK"]

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != "GK"]

# Print out the median height of goalkeepers. Replace 'None'
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Print out the median height of other players. Replace 'None'
print("Median height of other players: " + str(np.median(other_heights)))
