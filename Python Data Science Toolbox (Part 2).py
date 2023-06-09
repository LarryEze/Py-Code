# Import the course packages
import pandas as pd
import matplotlib.pyplot as plt

# Import the course datasets
world_ind = pd.read_csv(
    'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv')

tweets = pd.read_csv('Python Data Science Toolbox (Part 2)/tweets.csv')


''' Using iterators in PythonLand '''

'''
Introduction to iterators

Iterating with a for loop
# We can iterate over a list using a for loop
e.g 
letters = ['A', 'B', 'C']
for letter in letters :
    print(letter)

# We can iterate over a string using a for loop
e.g 
for letter in 'DataCamp' :
    print(letter)

# We can iterate over a range object using a for loop
e.g 
for i in range(4) :
    print(i)

Iterators vs. Iterables
# Iterable
- E.g : lists, strings, dictionaries, file connections
- An object with an associated iter() method.
- Applying iter() to an iterable creates an iterator
# Iterator
- Produces next value with next() method.

Iterating over iterables: next()
e.g
word = 'Da'
it = iter(word)
next(it) <- in
'D' <- out
next(it) <- in
'a' <- out

Iterating at once with *
e.g
word = 'Data'
it = iter(word)
print(*it) <- in
D a t a <- out

Iterating over dictionaries
e.g
pythonistas = { 'hugo': 'bowne', 'francis': 'castro'}
for key, value in pythonistas.items() :
    print(key, value) <- in
francis castro <- out
hugo bowne

Iterating over file connections
e.g
file = open('file.txt')
it = iter(file)
print(next(it))
'''

# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash:
    print(person)

# Create an iterator for flash: superhero
superhero = iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))


# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))


# Create a range object: values
values = range(10, 21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)


'''
Playing with iterators
Using enumerate()
This is a function that takes any iterable as argument, such as a list, and returns a special enumerate object, which consists of pairs containing the elements of the original iterable, along with their index within the iterable.
e.g
avengers = ['hawkeye', 'iron man', 'thor']
e = enumerate(avengers)
print(type(e)) <- in
<class 'enumerate'> <- out

e_list = list(e) 
print(e_list) <- in
[(0, 'hawkeye'), (1, 'iron man'), (2, 'thor')] <- out

enumerate() and unpack
e.g
avengers = ['hawkeye', 'iron man', 'thor']
for index, value in enumerate(avengers) :
    print(index, value) <- in
0 hawkeye <- out
1 iron man
2 thor

Using zip()
This accepts an arbitrary number of iterables and returns an iterator of tuples.
e.g
avengers = ['hawkeye', 'iron man', 'thor']
names = ['barton', 'stark', 'odinson']
z = zip(avengers, names)
print(type(z)) <- in
<class 'zip'> <- out

z_list = list(z)
print(z_list) <- in
[('hawkeye', 'barton'), ('iron man', 'stark'), ('thor', odinson')] <- out

zip() and unpack
e.g
avengers = ['hawkeye', 'iron man', 'thor']
names = ['barton', 'stark', 'odinson']
for z1, z2 in zip(avengers, names) :
    print(z1, z2) <- in
hawkeye barton <- out
iron man stark
thor odinson

print zip with *
e.g
avengers = ['hawkeye', 'iron man', 'thor']
names = ['barton', 'stark', 'odinson']
z = zip(avengers, names)
print(*z) <- in
('hawkeye', 'barton') ('iron man', 'stark') ('thor', odinson') <- out

# ( * ) is referred to as the splat operator and its used to print all the elements!
'''

# Create a list of strings: mutants
mutants = ['charles xavier',
           'bobby drake',
           'kurt wagner',
           'max eisenhardt',
           'kitty pryde']

# Create a list of tuples: mutant_list
mutant_list = list(enumerate(mutants))

# Print the list of tuples
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)

# Change the start index
for index2, value2 in enumerate(mutants, start=1):
    print(index2, value2)


# Create a list of strings: aliases and powers
aliases = ['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
powers = ['telepathy', 'thermokinesis',
          'teleportation', 'magnetokinesis', 'intangibility']

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants, aliases, powers))

# Print the list of tuples
print(mutant_data)

# Create a zip object using the three lists: mutant_zip
mutant_zip = zip(mutants, aliases, powers)

# Print the zip object
print(mutant_zip)

# Unpack the zip object and print the tuple values
for value1, value2, value3 in mutant_zip:
    print(value1, value2, value3)


# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)


'''
Using iterators to load large files into memory
Loading data in chunks
-   pandas function: read_csv()
specify the chunk: chunksize

Iterating over data
e.g
import pandas as pd
result = []
for chunk in pd.read_csv('data.csv', chunksize=1000) :
    result.append(sum(chunk['column_name']))
total = sum(result)
print(total)
### OR
import pandas as pd
total = 0
for chunk in pd.read_csv('data.csv', chunksize=1000) :
    total += sum(chunk['column_name'])
print(total)
'''

# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Iterate over the file chunk by chunk
for chunk in pd.read_csv('Python Data Science Toolbox (Part 2)/tweets.csv', chunksize=10):

    # Iterate over the column in DataFrame
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

# Print the populated dictionary
print(counts_dict)


# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    return counts_dict


# Call count_entries(): result_counts
result_counts = count_entries(
    'Python Data Science Toolbox (Part 2)/tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)


''' List comprehensions and generators '''

'''
List comprehensions:
-   Create lists from other lists, DataFrame columns, etc.
-   Collapse for loops for building lists into a single line of code
-   More efficient than using a for loop
-   Components: 
Iterable
Iterator variable (represent members of iterable)
Output expression

Populate a list with a for loop (i.e add 1 to each element]
e.g
nums = [12, 8, 21]
new_nums = []
for num in nums :
    new_nums.append(num + 1)
print(new_nums) <- in
[13, 9, 22] <- out

A list comprehension
e.g
nums = [12, 8, 21]
new_nums = [num + 1 for num in nums]
print(new_nums) <- in
[13, 9, 22] <- out

List comprehension with range()
e.g
result = [num for num in range(11)]
print(result) <- in
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] <- out

Nested Loops
pairs_1 = []
for num1 in range(0, 2) :
    for num2 in range(6, 8) :
        pairs_1.append((num1, num2))
print(pairs_1) <- in
[(0, 6), (0, 7), (1, 6), (1, 7)] <- out

Nested Loops Comprehension
pairs_2 = [(num1, num2) for num1 in range(0, 2) for num2 in range(6, 8)]
print(pairs_2) <- in
[(0, 6), (0, 7), (1, 6), (1, 7)] <- out

Note: 'int' object is not iterable
'''

# Create list comprehension: squares
squares = [i ** 2 for i in range(10)]


# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)


'''
Advanced comprehensions

Conditionals in comprehensions
-   Conditionals on the iterable
e.g
[num ** 2 for num in range(10) if num % 2 == 0] <- in
[0, 4, 16, 36, 64] <- out

-   Conditionals on the output expression
e.g
[num ** 2 if num % 2 == 0 else 0 for num in range(10)] <- in
[0, 0, 4, 0, 16, 0, 36, 0, 64, 0] <- out

-   Dictionary comprehensions
-- Use curly braces {} instead of brackets 
e.g
pos_neg = {num: -num for num in range(9)} 
print(pos_neg) <- in
{0: 0, 1: -1, 2: -2, 3: -3, 4: -4, 5: -5, 6: -6, 7: -7, 8: -8} <- out
'''

# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry',
              'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member for member in fellowship if len(member) >= 7]

# Print the new list
print(new_fellowship)


# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry',
              'aragorn', 'legolas', 'boromir', 'gimli']

# Create list comprehension: new_fellowship
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

# Print the new list
print(new_fellowship)


# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry',
              'aragorn', 'legolas', 'boromir', 'gimli']

# Create dict comprehension: new_fellowship
new_fellowship = {member: len(member) for member in fellowship}

# Print the new dictionary
print(new_fellowship)


'''
Introduction to generator expressions
List comprehensions vs generators
-   List comprehension - returns a list and uses '[]'
-   Generators -  returns a generator object and uses '()'
-   Both can be iterated over

Generator functions
-   Produces generator objects when called
-   Defined like a regular function - def
-   Yields a sequence of values instead of returning a single value
-   Generates a value with yield keyword
e.g
-   sequence.py
def num_sequence(n) :
    """Generate values from 0 to n."""
    i = 0
    while i < n:
        yield i
        i += 1

result = num_sequence(5)
print(type(result)) <- in
<class 'generator'>
for item in result :
    print(item) <- in
0 <- out
1
2
3
4 
'''

# Create generator object: result
result = (num for num in range(31))

# Print the first 5 values
print(next(result))
print(next(result))
print(next(result))
print(next(result))
print(next(result))

# Print the rest of the values
for value in result:
    print(value)


# Create a list of strings: lannister
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Create a generator object: lengths
lengths = (len(person) for person in lannister)

# Iterate over and print the values in lengths
for value in lengths:
    print(value)


# Create a list of strings
lannister = ['cersei', 'jaime', 'tywin', 'tyrion', 'joffrey']

# Define generator function get_lengths


def get_lengths(input_list):
    """Generator function that yields the length of the strings in input_list."""

    # Yield the length of a string
    for person in input_list:
        yield len(person)


# Print the values generated by get_lengths()
for value in get_lengths(lannister):
    print(value)


'''
Wrapping up comprehensions and generators.
Re-cap: list comprehensions
- Basic
[ output expression for iterator variable in iterable ]
-Advanced
[ output expression + conditional on output for iterator variable in iterable + conditional on iterable ]
'''

df = tweets

# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19] for entry in tweet_time]

# Print the extracted times
print(tweet_clock_time)


# Extract the created_at column from df: tweet_time
tweet_time = df['created_at']

# Extract the clock time: tweet_clock_time
tweet_clock_time = [entry[11:19]
                    for entry in tweet_time if entry[17:19] == '19']

# Print the extracted times
print(tweet_clock_time)


''' Bringing it all together! '''

''' Welcome to the case study! '''

feature_names = ['CountryName', 'CountryCode',
                 'IndicatorName', 'IndicatorCode', 'Year', 'Value']
row_vals = ['Arab World', 'ARB',
            'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298']

# Zip lists: zipped_lists
zipped_lists = zip(feature_names, row_vals)

# Create a dictionary: rs_dict
rs_dict = dict(zipped_lists)

# Print the dictionary
print(rs_dict)


# Define lists2dict()
def lists2dict(list1, list2):
    """Return a dictionary where list1 provides the keys and list2 provides the values."""

    # Zip lists: zipped_lists
    zipped_lists = zip(list1, list2)

    # Create a dictionary: rs_dict
    rs_dict = dict(zipped_lists)

    # Return the dictionary
    return rs_dict


# Call lists2dict: rs_fxn
rs_fxn = lists2dict(feature_names, row_vals)

# Print rs_fxn
print(rs_fxn)


row_lists = [['Arab World', 'ARB', 'Adolescent fertility rate (births per 1,000 women ages 15-19)', 'SP.ADO.TFRT', '1960', '133.56090740552298'], ['Arab World', 'ARB', 'Age dependency ratio (% of working-age population)', 'SP.POP.DPND', '1960', '87.7976011532547'], ['Arab World', 'ARB', 'Age dependency ratio, old (% of working-age population)', 'SP.POP.DPND.OL', '1960', '6.634579191565161'], ['Arab World', 'ARB', 'Age dependency ratio, young (% of working-age population)', 'SP.POP.DPND.YG', '1960', '81.02332950839141'], ['Arab World', 'ARB', 'Arms exports (SIPRI trend indicator values)', 'MS.MIL.XPRT.KD', '1960', '3000000.0'], ['Arab World', 'ARB', 'Arms imports (SIPRI trend indicator values)', 'MS.MIL.MPRT.KD', '1960', '538000000.0'], ['Arab World', 'ARB', 'Birth rate, crude (per 1,000 people)', 'SP.DYN.CBRT.IN', '1960', '47.697888095096395'], ['Arab World', 'ARB', 'CO2 emissions (kt)', 'EN.ATM.CO2E.KT', '1960', '59563.9892169935'], ['Arab World', 'ARB', 'CO2 emissions (metric tons per capita)', 'EN.ATM.CO2E.PC', '1960', '0.6439635478877049'], ['Arab World', 'ARB', 'CO2 emissions from gaseous fuel consumption (% of total)', 'EN.ATM.CO2E.GF.ZS', '1960', '5.041291753975099'], [
    'Arab World', 'ARB', 'CO2 emissions from liquid fuel consumption (% of total)', 'EN.ATM.CO2E.LF.ZS', '1960', '84.8514729446567'], ['Arab World', 'ARB', 'CO2 emissions from liquid fuel consumption (kt)', 'EN.ATM.CO2E.LF.KT', '1960', '49541.707291032304'], ['Arab World', 'ARB', 'CO2 emissions from solid fuel consumption (% of total)', 'EN.ATM.CO2E.SF.ZS', '1960', '4.72698138789597'], ['Arab World', 'ARB', 'Death rate, crude (per 1,000 people)', 'SP.DYN.CDRT.IN', '1960', '19.7544519237187'], ['Arab World', 'ARB', 'Fertility rate, total (births per woman)', 'SP.DYN.TFRT.IN', '1960', '6.92402738655897'], ['Arab World', 'ARB', 'Fixed telephone subscriptions', 'IT.MLT.MAIN', '1960', '406833.0'], ['Arab World', 'ARB', 'Fixed telephone subscriptions (per 100 people)', 'IT.MLT.MAIN.P2', '1960', '0.6167005703199'], ['Arab World', 'ARB', 'Hospital beds (per 1,000 people)', 'SH.MED.BEDS.ZS', '1960', '1.9296220724398703'], ['Arab World', 'ARB', 'International migrant stock (% of population)', 'SM.POP.TOTL.ZS', '1960', '2.9906371279862403'], ['Arab World', 'ARB', 'International migrant stock, total', 'SM.POP.TOTL', '1960', '3324685.0']]

# Print the first two lists in row_lists
print(row_lists[0])
print(row_lists[1])

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Print the first two dictionaries in list_of_dicts
print(list_of_dicts[0])
print(list_of_dicts[1])


# Import the pandas package

# Turn list of lists into list of dicts: list_of_dicts
list_of_dicts = [lists2dict(feature_names, sublist) for sublist in row_lists]

# Turn list of dicts into a DataFrame: df
df = pd.DataFrame(list_of_dicts)

# Print the head of the DataFrame
print(df.head())


''' Using Python generators for streaming data '''

# Open a connection to the file
with open('Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv') as file:

    # Skip the column names
    file.readline()

    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Process only the first 1000 rows
    for j in range(1000):

        # Split the current line into a list: line
        line = file.readline().split(',')

        # Get the value for the first column: first_col
        first_col = line[0]

        # If the column value is in the dict, increment its value
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1

        # Else, add to the dict and set value to 1
        else:
            counts_dict[first_col] = 1

# Print the resulting dictionary
print(counts_dict)


# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data


# Open a connection to the file
with open('Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))


# Initialize an empty dictionary: counts_dict
counts_dict = {}

# Open a connection to the file
with open('Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv') as file:

    # Iterate over the generator from read_large_file()
    for line in read_large_file(file):

        row = line.split(',')
        first_col = row[0]

        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

# Print
print(counts_dict)


''' Using pandas' read_csv iterator for streaming data '''

# Import the pandas package

# Initialize reader object: df_reader
df_reader = pd.read_csv(
    'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv', chunksize=10)

# Print two chunks
print(next(df_reader))
print(next(df_reader))


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv(
    'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv', chunksize=1000)

# Get the first DataFrame chunk: df_urb_pop
df_urb_pop = next(urb_pop_reader)

# Check out the head of the DataFrame
print(df_urb_pop.head())

# Check out specific country: df_pop_ceb
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

# Zip DataFrame columns of interest: pops
pops = zip(df_pop_ceb['Total Population'],
           df_pop_ceb['Urban population (% of total)'])

# Turn zip object into list: pops_list
pops_list = list(pops)

# Print pops_list
print(pops_list)


# Code from previous exercise
urb_pop_reader = pd.read_csv(
    'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'],
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [
    int(a * b * 0.01) for a, b in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv(
    'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
               df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [
        int(tup[0] * tup[1] * 0.01) for tup in pops_list]

    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


# Define plot_pop()
def plot_pop(filename, country_code):

    # Initialize reader object: urb_pop_reader
    urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    # Initialize empty DataFrame: data
    data = pd.DataFrame()

    # Iterate over each DataFrame chunk
    for df_urb_pop in urb_pop_reader:
        # Check out specific country: df_pop_ceb
        df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        # Zip DataFrame columns of interest: pops
        pops = zip(df_pop_ceb['Total Population'],
                   df_pop_ceb['Urban population (% of total)'])

        # Turn zip object into list: pops_list
        pops_list = list(pops)

        # Use list comprehension to create new DataFrame column 'Total Urban Population'
        df_pop_ceb['Total Urban Population'] = [
            int(tup[0] * tup[1] * 0.01) for tup in pops_list]

        # Append DataFrame chunk to data: data
        data = data.append(df_pop_ceb)

    # Plot urban population data
    data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()


# Set the filename: fn
fn = 'Python Data Science Toolbox (Part 2)\world_ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop(fn, 'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn, 'ARB')
