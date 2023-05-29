# Importing numpy
import numpy as np

# Importing matplotlib
import matplotlib.pyplot as plt

# Importing the data
with open("Introduction to NumPy/rgb_array.npy", "rb") as f:
    rgb_array = np.load(f)
with open("Introduction to NumPy/tree_census.npy", "rb") as f:
    tree_census = np.load(f)
with open("Introduction to NumPy\monthly_sales.npy", "rb") as f:
    monthly_sales = np.load(f)
with open("Introduction to NumPy\sudoku_game.npy", "rb") as f:
    sudoku_game = np.load(f)
with open("Introduction to NumPy\sudoku_solution.npy", "rb") as f:
    sudoku_solution = np.load(f)


''' UNDERSTANDING NumPy ARRAYS '''

'''
INTRODUCING ARRAYS
importing NumPy
import numpy as np

Creating 1Dimensional arrays from lists
python_list = [3, 2, 5, 8, 4]
array = np.array(python_list)
array
type(array)

type() - This is used to check the data type

python_list_of_lists = [3, 2, 5], [9, 8, 4], [6, 3, 4]]
array = np.array(python_list_of_lists)
array

# Python lists can contain many different data types
e.g
python_list = ['beep', False, 56, .945, [3, 2, 5]]

# NumPy arrays can contain only a single data type
e.g
numpy_boolean_array = [[True, False], [True, True], [False, True]]
numpr_float_array = [1.9, 5.4, 8.8, 3.6, 3.2]

Creating arrays from scratch
There are many NumPy functions used to create arrays from scratch, including:
# np.zeros()
# np.random.random()
# np.arange()

Creating arrays: np.zeros()
This will create an array full of zeros of 5 rows and 3 columns
in: np.zeros((5, 3))
out: array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

Creating arrays: np.random.random()
This will create an array made up of random floats between zero and one of 2 rows and 4 columns
in: np.random.random((2, 4))
out: array([[0.74634, 0.45686, 0.23458, 0.78543], [0.83642, 0.83562, 089365, 0.08265]])

Creating arrays: np.arange()
This will create an evenly-spaced array of numbers based on given start and stop values
np.arange(start_value, stop_value)

start_value will be included while the stop_value will not be included
in: np.arange(-3, 4)
out: array([-3, -2, -1, 0, 1, 2, 3])

start_value can be omitted if the range begins with zero (0)
in: np.arange(4)
out: array([0, 1, 2, 3])

The third argument is interpreted as the step value
in: np.arange(-3, 4, 3)
out: array([-3, 0, 3])

np.arange is particularly useful for plotting
from matplotlib import pyplot as plt
plt.scatter(np.arange(0, 7), np.arange(-3, 4))
plt.show()
'''

# Import NumPy

sudoku_list = [[0, 0, 4, 3, 0, 0, 2, 0, 9],
               [0, 0, 5, 0, 0, 9, 0, 0, 1],
               [0, 7, 0, 0, 6, 0, 0, 4, 3],
               [0, 0, 6, 0, 0, 2, 0, 8, 7],
               [1, 9, 0, 0, 0, 7, 4, 0, 0],
               [0, 5, 0, 0, 8, 3, 0, 0, 0],
               [6, 0, 0, 0, 0, 0, 1, 0, 5],
               [0, 0, 3, 5, 0, 8, 6, 9, 0],
               [0, 4, 2, 9, 1, 0, 3, 0, 0]]

# Convert sudoku_list into an array
sudoku_array = np.array(sudoku_list)

# Print the type of sudoku_array
print(type(sudoku_array))


# Create an array of zeros which has four columns and two rows
zero_array = np.zeros((2, 4))
print(zero_array)

# Create an array of random floats which has six columns and three rows
random_array = np.random.random((3, 6))
print(random_array)


doubling_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

# Create an array of integers from one to ten
one_to_ten = np.arange(1, 11)

# Create your scatterplot
plt.scatter(one_to_ten, doubling_array)
plt.show()


'''
ARRAY DIMENSIONALITY
3D arrays
array_1_2D = np.array([[1, 3], [5, 7]])
array_2_2D = np.array([[8, 9], [5, 7]])
array_3_2D = np.array([[1, 2], [5, 7]])
array_3D = np.array([array_1_2D , array_2_2D, array_3_2D])

4D arrays
This are 2D array filled with 3D arrays
array_4D = np.array([array_A_3D, array_B_3D, array_C_3D, array_D_3D, array_E_3D, array_F_3D, array_G_3D, array_H_3D, array_I_3D])

# A Vector refers to an array with one dimension
# A Matrix refers to an array with two dimensions
# A Tensor refers to an array with three or more dimensions

Rows are the first dimension
Columns are the second dimension

Shape shifting
Array attribute:
These are properties of an instance of an array, such as
# .shape - describes the shape of an array and returns a tuple of the length of each dimension
e.g
array = np.zeros((3, 5))
array.shape

Array methods:
These are called directly on the array object itself, such as
# .flatten() - It takes all array elements and puts them in just one dimension inside a 1D array
# .reshape() - Allows to redefine the shape of an array without changing the elements that make up the array
e.g
array = np.array([[1, 2], [5, 7], [6, 6]])
array.flatten()

array = np.array([[1, 2], [5, 7], [6, 6]])
array.reshape((2, 3))
'''

# Create the game_and_solution 3D array
game_and_solution = np.array([sudoku_game, sudoku_solution])

# Print game_and_solution
print(game_and_solution)


new_sudoku_game = np.array([[8, 9, 0, 2, 0, 0, 6, 7, 0],
                            [7, 0, 0, 9, 0, 0, 0, 5, 0],
                            [5, 0, 0, 0, 0, 8, 1, 4, 0],
                            [0, 7, 0, 0, 3, 2, 0, 6, 0],
                            [6, 0, 0, 0, 0, 1, 3, 0, 8],
                            [0, 0, 1, 7, 5, 0, 9, 0, 0],
                            [0, 0, 5, 0, 4, 0, 0, 1, 2],
                            [9, 8, 0, 0, 0, 6, 0, 0, 5]])


new_sudoku_solution = np.array([[8, 9, 3, 2, 1, 5, 6, 7, 4],
                                [7, 1, 6, 9, 8, 4, 2, 5, 3],
                                [5, 3, 2, 6, 9, 8, 1, 4, 7],
                                [1, 7, 8, 4, 3, 2, 5, 6, 9],
                                [6, 4, 9, 5, 7, 1, 3, 2, 8],
                                [4, 2, 1, 7, 5, 3, 9, 8, 6],
                                [3, 6, 5, 8, 4, 9, 7, 1, 2],
                                [9, 8, 7, 1, 2, 6, 4, 3, 5]])

# Create a second 3D array of another game and its solution
new_game_and_solution = np.array([new_sudoku_game, new_sudoku_solution])

# Create a 4D array of both game and solution 3D arrays
games_and_solutions = np.array([game_and_solution, new_game_and_solution])

# Print the shape of your 4D array
print(games_and_solutions.shape)


# Flatten sudoku_game
flattened_game = sudoku_game.flatten()

# Print the shape of flattened_game
print(flattened_game.shape)

# Reshape flattened_game back to a nine by nine array
reshaped_game = flattened_game.reshape(8, 9)

# Print sudoku_game and reshaped_game
print(sudoku_game)
print(reshaped_game)


'''
NumPy data types

NumPy vs Python data types
Sample Python data types:
int
float

Sample NumPy data types:
np.int64
np.int32
np.float64
np.float32

The .dtype attribute
This is used to find the data type of elements in an array
e.g
np.array([1.32, 5.78, 175.55]).dtype

dtype as an argument
This is used to declare a data type when creating the array
e.g
float32_array = np.array([1.32, 5.78, 175.55], dtype= np.float32)
float32_array.dtype

A keyword argument is an argument preceded by an identifying word in a function or method call

Type conversion
boolean_array = np.array([[True, False], [False, False]], dtype= np.bool_)
boolean_array.astype(np.int32)

Type coercion
This is when numpy changes all the data to one data type
e.g
np.array([True, 'Boop', 42, 42.42])

Type coercion hierarchy
string > float > int > bool
'''

# Create an array of zeros with three rows and two columns
zero_array = np.zeros((3, 2))

# Print the data type of zero_array
print(zero_array.dtype)

# Create a new array of int32 zeros with three rows and two columns
zero_int_array = np.zeros((3, 2), dtype=np.int32)

# Print the data type of zero_int_array
print(zero_int_array.dtype)


# Print the data type of sudoku_game
print(sudoku_game.dtype)

# Change the data type of sudoku_game to int8
small_sudoku_game = sudoku_game.astype(np.int8)

# Print the data type of small_sudoku_game
print(small_sudoku_game.dtype)


''' Selecting and Updating Data '''

'''
Indexing and slicing arrays

Indexing
This is an order based method for accessing data.
NumPy indexing is zero-based, meaning the first index is index zero.

Indexing 1D arrays
array = np.array([2, 4, 6, 8, 10])
array[3]

Indexing elements in 2D
When indexing a 2D array, give NumPy both a row and column index in order to return a single element
array_name[2, 4] - returns a single element
array_name[0] - returns all the row elements (Row indexing)
array_name[:, 3] - returns all the column elements (Column indexing)

Slicing
This extracts a subset of data based on given indices from one array and creates a new array with the sliced data.
Elements of the start index is included in the results while that of the stop is not.

Slicing 1D arrays
array = np.array([2, 4, 6, 8, 10])
array[2:4] 

Slicing 2D arrays
array_name[3:6, 3:6]

Slicing with steps
We can give NumPy a third number: step value
array_name[3:6:2, 3:6:2]

Axis order
NumPy axis labels in a 2D array, the direction along rows is axis zero (0) while the direction along columns is axis one (1)
ROWS - AXIS 0
COLUMNS - AXIS 1

Sorting arrays
np.sort(array_name) - sorting (default = columns in 2D arrays)
np.sort(array_name, axis=0) - sorting by rows
np.sort(array_name, axis=1) - sorting by columns
'''

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Print the first five block_ids
print(block_ids[0:5])

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Select the tenth block ID from block_ids
tenth_block_id = block_ids[9]
print(tenth_block_id)

# Select all rows of block ID data from the second column
block_ids = tree_census[:, 1]

# Select five block IDs from block_ids starting with the tenth ID
block_id_slice = block_ids[9:14]
print(block_id_slice)


# Create an array of the first 100 trunk diameters from tree_census
hundred_diameters = tree_census[0:100, 2]
print(hundred_diameters)

# Create an array of trunk diameters with even row indices from 50 to 100 inclusive
every_other_diameter = tree_census[50:101:2, 2]
print(every_other_diameter)


# Extract trunk diameters information and sort from smallest to largest
sorted_trunk_diameters = np.sort(tree_census[:, 2])
print(sorted_trunk_diameters)


'''
Filtering arrays
Two ways to filter
-   Masks and fancy indexing ( Returns array of elements )
-   np.where() ( Returns array of indices )

Masks and fancy indexing
Boolean masks
one_to_five = np,arange(1, 6)
one_to_five
#   To get even numbers
mask = one_to_five % 2 == 0
mask
Filtering with fancy indexing
one_to_five[mask]

2D fancy indexing
classroom_ids_and_sizes = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
classroom_ids_and_sizes
#   To get ids with even sizes
classroom_ids_and_sizes[:, 0][classroom_ids_and_sizes[:, 1] % 2 == 0]


Filtering with np.where()
classroom_ids_and_sizes
np.where( classroom_ids_and_sizes[:, 1] % 2 == 0 )

A tuple of indices
sudoku_game
#   To get all rows and columns with zero(0) as value
row_ind, column_ind = np.where(sudoku_game == 0)
row_ind, column_ind

Find and replace
np.where(sudoku_game == 0, "", sudoku_game)
'''

# Create an array which contains row data on the largest tree in tree_census
largest_tree_data = tree_census[tree_census[:, 2] == 51]
print(largest_tree_data)

# Slice largest_tree_data to get only the block ID
largest_tree_block_id = largest_tree_data[:, 1]
print(largest_tree_block_id)

# Create an array which contains row data on all trees with largest_tree_block_id
trees_on_largest_tree_block = tree_census[tree_census[:, 1]
                                          == largest_tree_block_id]
print(trees_on_largest_tree_block)


# Create the block_313879 array containing trees on block 313879
block_313879 = tree_census[tree_census[:, 1] == 313879]
print(block_313879)

# Create an array of row_indices for trees on block 313879
row_indices = np.where(tree_census[:, 1] == 313879)

# Create an array which only contains data for trees on block 313879
block_313879 = tree_census[row_indices]
print(block_313879)


# Create and print a 1D array of tree and stump diameters
trunk_stump_diameters = np.where(
    tree_census[:, 2] == 0, tree_census[:, 3], tree_census[:, 2])
print(trunk_stump_diameters)


'''
Adding and removing data
Concatenating in NumPy
classroom_ids_and_sizes = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
new_classrooms = np.array([[5, 30], [5, 17]])
np.concatenate((classroom_ids_and_sizes, new_classrooms))

Concatenating columns
classroom_ids_and_sizes = np.array([[1, 22], [2, 21], [3, 27], [4, 26]])
grade_levels_and_teachers = np.array([[1, 'James'], [1, .George'], [3, 'Amy'], [3, 'Meehir']])
np.concatenate((classroom_ids_and_sizes, grade_levels_and_teachers), axis = 1)

Shape compatibility
ValueError: all the input array dimensions for the concatenation axis must match exactly

Creating compatibility
array_1D = np.array([1, 2, 3])
column_array_2D = array_1D.reshape((3, 1))
column_array_2D
row_array_2D = array_1D.reshape((1, 3))
row_array_2D

Deleting with np.delete()
array_name
np.delete(array_name, slice / index, axis)
e.g
classroom_data
np.delete(classroom_data, 1, axis=0)
'''

new_trees = np.array([[1211, 227386, 20, 0],
                      [1212, 227386, 8, 0]])

# Print the shapes of tree_census and new_trees
print(tree_census.shape, new_trees.shape)

# Add rows to tree_census which contain data for the new trees
updated_tree_census = np.concatenate((tree_census, new_trees))
print(updated_tree_census)


# Print the shapes of tree_census and trunk_stump_diameters
print(trunk_stump_diameters.shape, tree_census.shape)

# Reshape trunk_stump_diameters
reshaped_diameters = trunk_stump_diameters.reshape((1000, 1))

# Concatenate reshaped_diameters to tree_census as the last column
concatenated_tree_census = np.concatenate(
    (tree_census, reshaped_diameters), axis=1)
print(concatenated_tree_census)


# Delete the stump diameter column from tree_census
tree_census_no_stumps = np.delete(tree_census, 3, axis=1)

# Save the indices of the trees on block 313879
private_block_indices = np.where(tree_census[:, 1] == 313879)

# Delete the rows for trees on block 313879 from tree_census_no_stumps
tree_census_clean = np.delete(tree_census_no_stumps, [921, 922], axis=0)

# Print the shape of tree_census_clean
print(tree_census_clean.shape)


''' Array Mathematics! '''

'''
Summarizing data
Aggregating methods
- .sum()
- .min()
- .max()
- .mean()
- .cumsum()

Summing data
array_name.sum()
Aggregating rows
array_name.sum(axis=0)
Aggregating columns
array_name.sum(axis=1)

Minimum and maximum values
array_name.min() 
array_name.min(axis=1) - Min for each column
array_name.min(axis=0) - Min for each row
array_name.max()

Finding the mean
array_name.mean() 
array_name.mean(axis=1) - Mean for each column
array_name.mean(axis=0) - Mean for each row

The keepdims argument
.sum(), .min(), .max(), .mean() all have an optional keepdims keyword argument
e.g 
array_name.sum(axis=1)
array_name.sum(axis=1, keepdims=True)

#   If keepdims = True, then the dimensions that were collapsed when aggregating are left in the output array and set to one. (helpful for dimension compatibility)

Cumulative sums
This will return the cumulative sum of elements along a given axis
array_name.cumsum(axis = 0) - for rows 

Graphing summary values
e.g
array_name1 = array_name.cumsum(axis = 0)
plt.plot(np.arange(1, 6), array_name1[:, 0], label='text_label1')
plt.plot(np.arange(1, 6), array_name1.mean(axis = 1), label='text_label2')
plt.legend()
plt.show()
array_name.max()
'''

# Create a 2D array of total monthly sales across industries
monthly_industry_sales = monthly_sales.sum(axis=1, keepdims=True)
print(monthly_industry_sales)

# Add this column as the last column in monthly_sales
monthly_sales_with_total = np.concatenate(
    (monthly_sales, monthly_industry_sales), axis=1)
print(monthly_sales_with_total)


# Create the 1D array avg_monthly_sales
avg_monthly_sales = monthly_sales.mean(axis=1)
print(avg_monthly_sales)

# Plot avg_monthly_sales by month
plt.plot(np.arange(1, 13), avg_monthly_sales,
         label="Average sales across industries")

# Plot department store sales by month
plt.plot(np.arange(1, 13), monthly_sales[:, 2], label="Department store sales")
plt.legend()
plt.show()


# Find cumulative monthly sales for each industry
cumulative_monthly_industry_sales = monthly_sales.cumsum(axis=0)
print(cumulative_monthly_industry_sales)

# Plot each industry's cumulative sales by month as separate lines
plt.plot(np.arange(1, 13),
         cumulative_monthly_industry_sales[:, 0], label="Liquor Stores")
plt.plot(np.arange(1, 13),
         cumulative_monthly_industry_sales[:, 1], label="Restaurants")
plt.plot(np.arange(1, 13),
         cumulative_monthly_industry_sales[:, 2], label="Department stores")
plt.legend()
plt.show()


'''
Vectorized operations
Vectorized python code
array = np.array(['NumPy', 'is', 'awesome'])
vectorized_len = np.vectorize(len)
vectorized_len(array) > 2
'''

# Create an array of tax collected by industry and month
tax_collected = monthly_sales * 0.05
print(tax_collected)

# Create an array of sales revenue plus tax collected by industry and month
total_tax_and_revenue = monthly_sales + tax_collected
print(total_tax_and_revenue)


monthly_industry_multipliers = np.array([[0.98, 1.02, 1.],
                                         [1., 1.01, 0.97],
                                         [1.06, 1.03, 0.98],
                                         [1.08, 1.01, 0.98],
                                         [1.08, 0.98, 0.98],
                                         [1.1, 0.99, 0.99],
                                         [1.12, 1.01, 1.],
                                         [1.1, 1.02, 1.],
                                         [1.11, 1.01, 1.01],
                                         [1.08, 0.99, 0.97],
                                         [1.09, 1., 1.02],
                                         [1.13, 1.03, 1.02]])

# Create an array of monthly projected sales for all industries
projected_monthly_sales = monthly_sales * monthly_industry_multipliers
print(projected_monthly_sales)

# Graph current liquor store sales and projected liquor store sales by month
plt.plot(np.arange(1, 13),
         monthly_sales[:, 0], label="Current liquor store sales")
plt.plot(np.arange(1, 13),
         projected_monthly_sales[:, 0], label="Projected liquor store sales")
plt.legend()
plt.show()


names = np.array([['Izzy', 'Monica', 'Marvin'],
                  ['Weber', 'Patel', 'Hernandez']])

# Vectorize the .upper() string method
vectorized_upper = np.vectorize(str.upper)

# Apply vectorized_upper to the names array
uppercase_names = vectorized_upper(names)
print(uppercase_names)


'''
Broadcasting
Compatibility rules
#   NumPy compares sets of array dimensions from right to left
#   Two dimensions are compatible when
- One of them has a length of one or
- They are of equal lengths
#   All dimension sets must be compatible
e.g 
shape(10, 5) & shape(10, 1)

array = np.arange(10).reshape((2, 5))
array + np.array([0, 1, 2, 3, 4])

array = np.arange(10).reshape((2, 5))
array + np.array([0, 1]).reshape((2, 1))
'''

monthly_growth_rate = np.array(
    [1.01, 1.03, 1.03, 1.02, 1.05, 1.03, 1.06, 1.04, 1.03, 1.04, 1.02, 1.01])

# Convert monthly_growth_rate into a NumPy array
monthly_growth_1D = np.array(monthly_growth_rate)

# Reshape monthly_growth_1D
monthly_growth_2D = monthly_growth_1D.reshape((12, 1))

# Multiply each column in monthly_sales by monthly_growth_2D
print(monthly_sales * monthly_growth_2D)


# Find the mean sales projection multiplier for each industry
mean_multipliers = monthly_industry_multipliers.mean(axis=0)
print(mean_multipliers)

# Print the shapes of mean_multipliers and monthly_sales
print(mean_multipliers.shape, monthly_sales.shape)

# Multiply each value by the multiplier for that industry
projected_sales = monthly_sales * mean_multipliers
print(projected_sales)


''' Array Transformations '''

'''
RGB arrays
rgb = np.array([ [[255, 0, 0], [255, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 255, 0], [0, 255, 0]], [[0, 0, 255], [0, 0, 255], [0, 0, 255]] ])
plt.imshow(rgb)
plt.show()

Saving and loading arrays
Save arrays in many formats:
- .csv
- .txt
- .pkl
- .npy - fastest and most efficient for numpy

Loading .npy files
with open('file_name', 'open_mode') as alias:
rb - Read binary
e.g
with open('logo.npy', 'rb') as f:
    logo_rgb_array = np.load(f)
plt.imshow(logo_rgb_array)
plt.show()

Examining RGB data
red_array = logo_rgb_array[:, :, 0]
blue_array = logo_rgb_array[:, :, 1]
green_array = logo_rgb_array[:, :, 2]

red_array[1], green_array[1], blue_array[1]

updating RGB data
dark_logo_array = np.where(logo_rgb_array == 255, 50, logo_rgb_array)
plt.imshow(dark_logo_array)
plt.show()

Saving arrays as .npy files
with open('file_name', 'open_mode') as alias:
wb - Write binary
e.g
with open('dark_logo.npy', 'wb') as f:
np.save(f, dark_logo_array)

If we need help()
e.g
help(np.unique) - For function
help(np.ndarray.flatten) - For method
'''

# Load the mystery_image.npy file
with open('mystery_image.npy', 'rb') as f:
    rgb_array = np.load(f)

plt.imshow(rgb_array)
plt.show()


# Display the documentation for .astype()
help(np.ndarray.astype)


# Reduce every value in rgb_array by 50 percent
darker_rgb_array = rgb_array * 0.5

# Convert darker_rgb_array into an array of integers
darker_rgb_int_array = darker_rgb_array.astype(np.int8)
plt.imshow(darker_rgb_int_array)
plt.show()

# Save darker_rgb_int_array to an .npy file called darker_monet.npy
with open('darker_monet.npy', 'wb') as f:
    np.save(f, darker_rgb_int_array)


'''
Array acrobatics
In machine learning, data augmentation is the process of adding additional data by performing small manipulations on data that is already available.

Flipping an array
array_name1 = np.flip(array_name)
plt.imshow(array_name1)
plt.show()

Flipping along an axis
array_name1 = np.flip(array_name, axis = 0 ) - 1st axis

array_name1 = np.flip(array_name, axis = 1 ) - 2nd axis

array_name1 = np.flip(array_name, axis = 2 ) - 3rd axis

array_name1 = np.flip(array_name, axis = (0, 1) ) - 1st and 2nd axis

plt.imshow(array_name1)
plt.show()

Transposing an array
This flips axis order while keeping the element order within each axis the same
e.g
array_name1 = np.array(array_name)
np.transpose(array_name1)

The default behaviour for np.transpose is to reverse the axis order

Setting transposed axis order
e.g
array_name1 = np.transpose(array_name, axes=(1, 0, 2) )
plt.imshow(array_name1)
plt.show()
'''

# Flip rgb_array so that it is upside down
upside_down_monet = np.flip(rgb_array, axis=(0, 1))
plt.imshow(upside_down_monet)
plt.show()

# Flip rgb_array so that it is upside down
upside_down_monet = np.flip(rgb_array, axis=(0, 1))
plt.imshow(upside_down_monet)
plt.show()


# Transpose rgb_array
transposed_rgb = np.transpose(rgb_array, axes=(1, 0, 2))
plt.imshow(transposed_rgb)
plt.show()


'''
Stacking and splitting

Splitting arrays
e.g
rgb = np.array([ [[255, 0, 0], [255, 255, 0], [255, 255, 255]], [[255, 0, 255], [0, 255, 0], [0, 255, 255]], [[0, 0, 0], [0, 255, 255], [0, 0, 255]] ])
red_array = logo_rgb_array[:, :, 0]
blue_array = logo_rgb_array[:, :, 1]
green_array = logo_rgb_array[:, :, 2]
red_array, green_array, blue_array = np.split(rgb, 3, axis=2)
red_array

Trailing dimensions
red_array_2D = red_array.reshape((3, 3))
red_array_2D

Stacking arrays
Stacking 2D arrays
red_array = np.zeros((1001, 1001)).astype(np.int32)
green_Array = green_array.reshape((1001, 1001))
blue_array = blue_array.reshape((1001, 1001))
stacked_rgb = np.stack([red_Array, green_array, blue_array], axis = 2)
plt.imshow(stacked_rgd)
plt.show()
'''

# Split monthly_sales into quarterly data
q1_sales, q2_sales, q3_sales, q4_sales = np.split(monthly_sales, 4)

# Print q1_sales
print(q1_sales)

# Stack the four quarterly sales arrays
quarterly_sales = np.stack([q1_sales, q2_sales, q3_sales, q4_sales])
print(quarterly_sales)


# Split rgb_array into red, green, and blue arrays
red_array, green_array, blue_array = np.split(rgb_array, 3, axis=2)

# Create emphasized_blue_array
emphasized_blue_array = np.where(
    blue_array > blue_array.mean(), 255, blue_array)

# Print the shape of emphasized_blue_array
print(emphasized_blue_array.shape)

# Remove the trailing dimension from emphasized_blue_array
emphasized_blue_array_2D = emphasized_blue_array.reshape((675, 844))


# Print the shapes of blue_array and emphasized_blue_array_2D
print(blue_array.shape, emphasized_blue_array_2D.shape)

# Reshape red_array and green_array
red_array_2D = red_array.reshape((675, 844))
green_array_2D = green_array.reshape((675, 844))

# Stack red_array_2D, green_array_2D, and emphasized_blue_array_2D
emphasized_blue_monet = np.stack(
    [red_array_2D, green_array_2D, emphasized_blue_array_2D], axis=2)
plt.imshow(emphasized_blue_monet)
plt.show()
