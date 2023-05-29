# Import the course packages
import pandas as pd
from functools import reduce

# Import the dataset
tweets = pd.read_csv('Python Data Science Toolbox (Part 1)/tweets.csv')


''' Writing your own functions '''

'''
User-defined functions

Defining a function
e.g
def square() :              #   <-  Function header
    new_value = 4 ** 2      #   <-  Function body
    print(new_value)
square()

Function with parameters
e.g
def square(value) :         #   square(value)   <-  Signature of the function
    new_value = value ** 2  
    print(new_value)
square(4)

Parameters  -   They are written in the function header when defining them
Arguments   -   They are passed into already defined functions

Return values from functions
def square(value) :     
    new_value = value ** 2  
    return new_value
num = square(4)
print(num)

Docstrings
#   They describe what the function does, such as the computations it performs or its return values.
#   They also serve as documentation for the function
#   They are placed in the immediate line after the function header in between triple double quotes """
e.g
def square(value) :     
    """Return the square of a value."""
    new_value = value ** 2  
    return new_value
num = square(4)
'''

# Define the function shout


def shout():
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = 'congratulations' + '!!!'

    # Print shout_word
    print(shout_word)


# Call shout
shout()


# Define shout with the parameter, word
def shout(word):
    """Print a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Print shout_word
    print(shout_word)


# Call shout with the string 'congratulations'
shout('congratulations')


# Define shout with the parameter, word
def shout(word):
    """Return a string with three exclamation marks"""
    # Concatenate the strings: shout_word
    shout_word = word + '!!!'

    # Replace print with return
    return shout_word


# Pass 'congratulations' to shout: yell
yell = shout('congratulations')

# Print yell
print(yell)


'''
Multiple parameters and return values

Multiple function parameters
e.g
def raise_to_power(value1, value2) :
    """Raise value1 to the power of value2."""
    new_value = value1 ** value2
    return new_value

result = raise_to_power(2, 3)
print(result)

#   Call function: no of arguments = no of parameters

Returning multiple values
e.g
def raise_to_power(value1, value2) :
    """Raise value1 to the power of value2 and vice versa."""
    
    new_value1 = value1 ** value2
    new_value2 = value2 ** value1

    new_tuple = (new_value1, new_value2)

    return new_tuple

result = raise_to_power(2, 3)
print(result)
'''

# Define shout with parameters word1 and word2


def shout(word1, word2):
    """Concatenate strings with three exclamation marks"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'

    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'

    # Concatenate shout1 with shout2: new_shout
    new_shout = shout1 + shout2

    # Return new_shout
    return new_shout


# Pass 'congratulations' and 'you' to shout(): yell
yell = shout('congratulations', 'you')

# Print yell
print(yell)


nums = (3, 4, 6)

# Unpack nums into num1, num2, and num3
num1, num2, num3 = nums

# Construct even_nums
even_nums = (2, num2, num3)


# Define shout_all with parameters word1 and word2
def shout_all(word1, word2):

    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'

    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'

    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words


# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)


# Import pandas
# import pandas as pd

# Import Twitter data as DataFrame: df
df = pd.read_csv('Python Data Science Toolbox (Part 1)/tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1
    if entry in langs_count.keys():
        langs_count[entry] += 1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry] = 1

# Print the populated dictionary
print(langs_count)


# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]

    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry] += 1
        # Else add the language to langs_count, set the value to 1
        else:
            langs_count[entry] = 1

    # Return the langs_count dictionary
    return langs_count


tweets_df = df

# Call count_entries(): result
result = count_entries(tweets_df, 'lang')

# Print the result
print(result)


''' Default arguments, variable-length arguments and scope '''

'''
Scope and user-defined functions

Crash course on scope in functions
Scope   -   part of the program where an object or name may be accessible
There are 3 types of scope
#   Global scope    -   defined in the main body of a script
#   Local scope -   defined inside a function
#   Built-in scope -   names in the pre-defined built-ins module
e.g
def square(value) :
    """Returns the square of a number."""
    new_val = value ** 2
    return new_val
square(3)   <- in
9 <- out
new_val <- in
NameError.....  <- out

new_val = 10

def square(value) :
    """Returns the square of a number."""
    new_val = value ** 2
    return new_val
square(3)   <- in
9   <- out
new_val <- in
10  <- out

new_val = 10

def square(value) :
    """Returns the square of a number."""
    new_value2 = new_val ** 2
    return new_value2
square(3)   <- in
100 <- out
new_val = 20 
square(new_val) <- in
400 <- out

new_val = 10

def square(value) :
    """Returns the square of a number."""
    global new_val
    new_val = new_val ** 2
    return new_val
square(3)   <- in
100 <- out
new_val <- in
100 <- out
'''

# Create a string: team
team = "teen titans"

# Define change_team()


def change_team():
    """Change the value of the global variable team."""

    # Use team in global scope
    global team

    # Change the value of team in global: team
    team = "justice league"


# Print team
print(team)

# Call change_team()
change_team()

# Print team
print(team)


'''
Nested functions
e.g
def mod2plus5(x1, x2, x3) :
    """Returns the remainder plus 5 of the three values."""

    new_x1 = x1 % 2 + 5
    new_x2 = x2 % 2 + 5
    new_x3 = x3 % 2 + 5
   
    return (new_x1, new_x2, new_x3)

OR

def mod2plus5(x1, x2, x3) :
    """Returns the remainder plus 5 of the three values."""
    
    def inner(x) :
    """Returns the remainder plus 5 of a value."""
        return  x % 2 + 5
    
    return (inner(x1), inner(x2), inner(x3))

Returning functions
e.g
def raise_val(n) :
    """Returns the inner function."""
    
    def inner(x) :
    """Raise x to the power of n."""
        raised = x ** n
        return  raised
    
    return inner

square = raise_val(2)
cube = raise_val(3)
print(square(2), cube(4)) <- in
4, 64 <- out

Using nonlocal
e.g
def outer() :
    """Prints the value of n."""
    n = 1
    
    def inner() :
        nonlocal n
        n = 2
        print(n)
    
    inner()
    print(n)

outer() <- in
2
2 <- out

Scopes searched
-   Local scope
-   Enclosing functions
-   Global scope
-   Built-in
LEGB rule of function

import builtins -   Import built-in functions
dir(builtins)   -   View the available built-in functions
'''

# Define three_shouts


def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))


# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))


# Define echo
def echo(n):
    """Return the inner_echo function."""

    # Define inner_echo
    def inner_echo(word1):
        """Concatenate n copies of word1."""
        echo_word = word1 * n
        return echo_word

    # Return inner_echo
    return inner_echo


# Call echo: twice
twice = echo(2)

# Call echo: thrice
thrice = echo(3)

# Call twice() and thrice() then print
print(twice('hello'), thrice('hello'))


# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""

    # Concatenate word with itself: echo_word
    echo_word = word * 2

    # Print echo_word
    print(echo_word)

    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""
        # Use echo_word in nonlocal scope
        nonlocal echo_word

        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word + '!!!'

    # Call function shout()
    shout()

    # Print echo_word
    print(echo_word)


# Call function echo_shout() with argument 'hello'
echo_shout('hello')


'''
Default and flexible arguments
Add a default argument
e.g
def power(number, pow=1) :
    """Raise number to the power of pow."""
    new_value = number ** pow
    return new_value

power(9, 2) <- in
81 <- out 
power(9, 1) <- in
9 <- out 
power(9) <- in
9 <- out 

Flexible arguments: *args
*args   -   This turns all the arguments passed to a function call into a tuple (called args) in the function body
e.g
def add_all(*args) :
    """Sum all values in *args together."""

    # Initialize sum
    sum_all = 0

    # Accumulate the sum
    for num in args:
        sum_all += num
    
    return sum_all

add_all(1) <- in
1 <- out
add_all(1, 2) <- in
3 <- out
add_all(5, 10, 15, 20) <- in
50 <- out

Flexible arguments: **kwargs
**  -    Used to represent keyword arguments i.e arguments preceded by identifiers.

print_all(name="Hugo Bowne-Anderson", employer="Datacamp")
e.g
def print_all(**kwargs) :
    """Print out key-value pairs in **kwargs."""

    # Print out the key-value pairs
    for key, value in kwargs.items() :
        print(key + ": " + value)

print_all(name="dumbledore", job="headmaster") <- in
job: headmaster
name: dumbledore <- out
'''

# Define shout_echo


def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word


# Call shout_echo() with "Hey": no_echo
no_echo = shout_echo("Hey")

# Call shout_echo() with "Hey" and echo=5: with_echo
with_echo = shout_echo("Hey", 5)

# Print no_echo and with_echo
print(no_echo)
print(with_echo)


# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    """Concatenate echo copies of word1 and three exclamation marks at the end of the string."""

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Make echo_word uppercase if intense is True
    if intense is True:
        # Make uppercase and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'

    # Return echo_word_new
    return echo_word_new


# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey", echo=5, intense=True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)


# Define gibberish
def gibberish(*args):
    """Concatenate strings in *args together."""

    # Initialize an empty string: hodgepodge
    hodgepodge = ""

    # Concatenate the strings in args
    for word in args:
        hodgepodge += word

    # Return hodgepodge
    return hodgepodge


# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)


# Define report_status
def report_status(**kwargs):
    """Print out the status of a movie character."""

    print("\nBEGIN: REPORT\n")

    # Iterate over the key-value pairs of kwargs
    for key, value in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + value)

    print("\nEND REPORT")


# First call to report_status()
report_status(name="luke", affiliation="jedi", status="missing")

# Second call to report_status()
report_status(name="anakin", affiliation="sith lord", status="deceased")


# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]

    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1

        # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'source')

# Print result1 and result2
print(result1)
print(result2)


# Define count_entries()
def count_entries(df, *args):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Iterate over column names in args
    for col_name in args:

        # Extract column from DataFrame: col
        col = df[col_name]

        # Iterate over the column in DataFrame
        for entry in col:

            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1

            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

    # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

# Print result1 and result2
print(result1)
print(result2)


''' Lambda functions and error-handling '''

'''
Lambda functions
They are used to write functions faster
e.g
raise_to_power = lambda x, y: x ** y

raise_to_power(2, 3) <- in
8 <- out

Anonymous functions
- Function map takes two arguments: map(func, seq)
- map() applies the function to ALL elements in the sequence
e.g
nums = [48, 6, 9, 21, 1]

square_all = map(lambda num: num ** 2, nums)

print(list(square_all))
'''

add_bangs = (lambda a: a + '!!!')
add_bangs('hello')


# Define echo_word as a lambda function: echo_word
echo_word = (lambda word1, echo: word1 * echo)

# Call echo_word: result
result = echo_word('hey', 5)

# Print result
print(result)


# Create a list of strings: spells
spells = ["protego", "accio", "expecto patronum", "legilimens"]

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item: item + '!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Print the result
print(shout_spells_list)


# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin',
              'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']

# Use filter() to apply a lambda function over fellowship: result
result = filter(lambda member: len(member) > 6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Print result_list
print(result_list)


# Import reduce from functools
# from functools import reduce

# Create a list of strings: stark
stark = ['robb', 'sansa', 'arya', 'brandon', 'rickon']

# Use reduce() to apply a lambda function over stark: result
result = reduce(lambda item1, item2: item1 + item2, stark)

# Print the result
print(result)


'''
Introduction to error handling
Errors and exceptions
#   Exceptions  -   caught during execution
#   Catch exceptions with try-except clause
-   Runs the code following try
-   If there's an exception, run the code following except
e.g
def sqrt(x) :
    "Returns the square root of a number."""
    try :
        return x ** 0.5
    except :
        print('x must be an int or float')

OR

def sqrt(x) :
    "Returns the square root of a number."""
    try :
        return x ** 0.5
    except TypeError :
        print('x must be an int or float')

def sqrt(x) :
    "Returns the square root of a number."""
    if x < 0 :
        raise ValueError('x must be non-negative')
    try :
        return x ** 0.5
    except :
        print('x must be an int or float')     
'''

# Define shout_echo


def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three exclamation marks at the end of the string."""

    # Initialize empty strings: echo_word, shout_words
    echo_Word = ""
    shout_words = ""

    # Add exception handling with try-except
    try:
        # Concatenate echo copies of word1 using *: echo_word
        echo_word = word1 * echo

        # Concatenate '!!!' to echo_word: shout_words
        shout_words = echo_word + '!!!'
    except:
        # Print error message
        print("word1 must be a string and echo must be an integer.")

    # Return shout_words
    return shout_words


# Call shout_echo
shout_echo("particle", echo="accelerator")


# Define shout_echo
def shout_echo(word1, echo=1):
    """Concatenate echo copies of word1 and three exclamation marks at the end of the string."""

    # Raise an error with raise
    if echo < 0:
        raise ValueError('echo must be greater than or equal to 0')

    # Concatenate echo copies of word1 using *: echo_word
    echo_word = word1 * echo

    # Concatenate '!!!' to echo_word: shout_word
    shout_word = echo_word + '!!!'

    # Return shout_word
    return shout_word


# Call shout_echo
shout_echo("particle", echo=5)


# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2] == 'RT', tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)


# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Add try block
    try:
        # Extract column from DataFrame: col
        col = df[col_name]

        # Iterate over the column in DataFrame
        for entry in col:

            # If entry is in cols_count, add 1
            if entry in cols_count.keys():
                cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
            else:
                cols_count[entry] = 1

        # Return the cols_count dictionary
        return cols_count

    # Add except block
    except:
        print('The DataFrame does not have a ' + col_name + ' column.')


# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Print result1
print(result1)


# Define count_entries()
def count_entries(df, col_name='lang'):
    """Return a dictionary with counts of occurrences as value for each key."""

    # Raise a ValueError if col_name is NOT in DataFrame
    if col_name not in df.columns:
        raise ValueError('The DataFrame does not have a ' +
                         col_name + ' column.')

    # Initialize an empty dictionary: cols_count
    cols_count = {}

    # Extract column from DataFrame: col
    col = df[col_name]

    # Iterate over the column in DataFrame
    for entry in col:

        # If entry is in cols_count, add 1
        if entry in cols_count.keys():
            cols_count[entry] += 1
            # Else add the entry to cols_count, set the value to 1
        else:
            cols_count[entry] = 1

        # Return the cols_count dictionary
    return cols_count


# Call count_entries(): result1
result1 = count_entries(tweets_df, col_name='lang')

# Print result1
print(result1)
