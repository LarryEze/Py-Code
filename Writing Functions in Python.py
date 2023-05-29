''' Best Practices '''

'''
Docstrings
A complex function
def split_and_stack(df, new_names):
    half = int(len(df.columns) / 2)
    left = df.iloc[:, :half]
    right = df.iloc[:, half:]
    return pd.DataFrame(
        data =  np.vstack([left.values, right.values]),
        columns = new_names
    )

def split_and_stack(df, new_names):
    """ Split a DataFrames's columns into two halves and then stack them vertically, returning a new DataFrame with 'new_names' as the column names.

    Args:
        df (DataFrame): The DataFrame to split.
        new_names (iterable of str): The column names for the new DataFrame.

    Returns:
        DataFrame
    """
    half = int(len(df.columns) / 2)
    left = df.iloc[:, :half]
    right = df.iloc[:, half:]
    return pd.DataFrame(
        data =  np.vstack([left.values, right.values]),
        columns = new_names
    )

Anatomy of a docstring
def function_name(arguments):
    """
    Description of what the function does.

    Description pf the arguments, if any.

    Description of the return value(s), if any.

    Description of errors raised, if any.

    Optional extra notes or examples of usage.
    """

- A docstring is a string written as the first line of a function

Docstring formats
- Google Style
- Numpydoc
- reStructuredText
- EpyText


Google style - (Description, Arguments, Return value(s)
def function(arg_1, arg_2=42):
    """ Description of what the function does.

    Args:
        arg_1 (str): Description of arg_1 that can break onto the next line if needed.
        arg_2 (int, optional): Write optional when an argument has a default value.
    
    Returns:
        bool: Optional decription of the return value
        Extra lines are not indented

    Raises:
        ValueError: Include any error types that the function intentionally raises.

    Notes:
        See https://www.datacamp.com/community/tutorials/docstrings-python
    for more info.
    """

Numpydoc
def function(arg_1, arg_2=42):
    """ 
    Description of what the function does.

    Parameters
    -------------
    arg_1 : expected type of arg_1
        Description of arg_1.
    arg_2 : int, optional
        Write optional when an argument has a default value.
        Default = 42.
    
    Returns
    ---------
    The type of the return value
        Can include a description of the return value.
        Replace "Returns" with "Yields" if this function is a generator.
    """

Retrieving docstrings
def the_answer():
    """ Return the answer to life, 
    the universe, and everything.

    Returns:
        int
    """
    return 42
print(the_answer.__doc__) -> in

Return the answer to life,
    the universe, and everything.
    
    Return:
        int -> out

# OR
import inspect
print(inspect.getdoc(the_answer)) -> in

Return the answer to life,
the universe, and everything.
    
Return:
    int -> out
'''


def count_letter(content, letter):
    """Count the number of times `letter` appears in `content`.

    Args:
        content (str): The string to search.
        letter (str): The letter to search for.

    Returns:
        int

    # Add a section detailing what errors might be raised
    Raises:
        ValueError: If `letter` is not a one-character string.
    """
    if (not isinstance(letter, str)) or len(letter) != 1:
        raise ValueError('`letter` must be a single character string.')
    return len([char for char in content if char == letter])


# Get the "count_letter" docstring by using an attribute of the function
docstring = count_letter.__doc__

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))


# Inspect the count_letter() function to get its docstring
docstring = inspect.getdoc(count_letter)

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))


def build_tooltip(function):
    """Create a tooltip for any function that shows the
    function's docstring.

    Args:
        function (callable): The function we want a tooltip for.

    Returns:
        str
    """

    # Get the docstring for the "function" argument by using inspect
    docstring = inspect.getdoc(function)
    border = '#' * 28
    return '{}\n{}\n{}'.format(border, docstring, border)


print(build_tooltip(count_letter))
print(build_tooltip(range))
print(build_tooltip(print))


'''
DRY and "Do One Thing"
- DRY (also known as 'don't repeat yourself') and the 'Do One Thing' principle are good ways to ensure that functions are well designed and easy to test.

Don't repeat yourself (DRY)
train = pd.read_csv('train.csv')
train_y = train['labels'].values
train_x = train[col for col in train.columns if col != 'labels'].values
train_pca =  PCA(n_components = 2).fit_transform(train_x)
plt.scatter(train_pca[:, 0], train_pca[:, 1])

val = pd.read_csv('validation.csv')
val_y = val['labels'].values
val_x = val[col for col in val.columns if col != 'labels'].values
val_pca =  PCA(n_components = 2).fit_transform(val_x)
plt.scatter(val_pca[:, 0], val_pca[:, 1])

test = pd.read_csv('test.csv')
test_y = test['labels'].values
test_x = test[col for col in test.columns if col != 'labels'].values
test_pca =  PCA(n_components = 2).fit_transform(test_x)
plt.scatter(test_pca[:, 0], test_pca[:, 1])

Use functions to avoid repetition
def load_and_plot(path):
    """Load a dataset and plot the first two principal components.

    Args:
        path (str): The location of a CSV file.

    Returns:
        tuple of ndarray: (features, labels)
    """
    # load the data
    data = pd.read_csv(path)
    y = data['label'].values
    x = data[col for col in data.columns if col != 'label'].values

    # plot the first two principle components
    pca = PCA(n_components = 2).fit_transform(x)
    plt.scatter(pca[:, 0], pca[:, 1])
    
    # return loaded data
    return x, y

train_x, train_y = load_and_plot('train.csv')
val_x, val_y = load_and_plot('validation.csv')
test_x, test_y = load_and_plot('test.csv')

Do One Thing
def load_data(path):
    """Load a dataset and plot the first two principal components.

    Args:
        path (str): The location of a CSV file.

    Returns:
        tuple of ndarray: (features, labels)
    """
    data = pd.read_csv(path)
    y = data['label'].values
    x = data[col for col in data.columns if col != 'label'].values
    return x, y

def plot_data(x):
    """Plot the first two principal components of a matrix.

    Args:
        x (numpy.ndarray): The data to plot.
    """
    pca = PCA(n_components = 2).fit_transform(x)
    plt.scatter(pca[:, 0], pca[:, 1])

Advantages of doing one thing
The code becomes:
- More flexible
- More easily understood
- Simpler to test
- Simpler to debug
- Easier to change

Code smells and refactoring
- Repeated code and functions that do more than one thing are examples of 'code smells', which are indications that you may need to refactor.
- Refactoring is the process of improving code by changing it a little bit at a time.
'''

# Standardize the GPAs for each year
df['y1_z'] = (df.y1_gpa - df.y1_gpa.mean()) / df.y1_gpa.std()
df['y2_z'] = (df.y2_gpa - df.y2_gpa.mean()) / df.y2_gpa.std()
df['y3_z'] = (df.y3_gpa - df.y3_gpa.mean()) / df.y3_gpa.std()
df['y4_z'] = (df.y4_gpa - df.y4_gpa.mean()) / df.y4_gpa.std()


def standardize(column):
    """Standardize the values in a column.

    Args:
        column (pandas Series): The data to standardize.

    Returns:
        pandas Series: the values as z-scores
    """
    # Finish the function so that it returns the z-scores
    z_score = (column - column.mean()) / column.std()
    return z_score


# Use the standardize() function to calculate the z-scores
df['y1_z'] = standardize(df.y1_gpa)
df['y2_z'] = standardize(df.y2_gpa)
df['y3_z'] = standardize(df.y3_gpa)
df['y4_z'] = standardize(df.y4_gpa)


def mean_and_median(values):
    """Get the mean and median of a sorted list of `values`

    Args:
        values (iterable of float): A list of numbers

    Returns:
        tuple (float, float): The mean and median
    """
    mean = sum(values) / len(values)
    midpoint = int(len(values) / 2)
    if len(values) % 2 == 0:
        median = (values[midpoint - 1] + values[midpoint]) / 2
    else:
        median = values[midpoint]

    return mean, median


def mean(values):
    """Get the mean of a sorted list of values

    Args:
        values (iterable of float): A list of numbers

    Returns:
        float
    """
    # Write the mean() function
    mean = sum(values) / len(values)
    return mean


def median(values):
    """Get the median of a sorted list of values

    Args:
        values (iterable of float): A list of numbers

    Returns:
        float
    """
    # Write the median() function
    midpoint = int(len(values) / 2)
    if len(values) % 2 == 0:
        median = (values[midpoint - 1] + values[midpoint]) / 2
    else:
        median = values[midpoint]
    return median


'''
Pass by assignment
A surprising example
def foo(x):
    x[0] = 99

my_list = [1, 2, 3]
foo(my_list)
print(my_list) -> in

[99, 2, 3] -> out (lists are mutable i.e can be changed)

def bar(x):
    x = x + 90

my_var = 3
bar(my_var)
print(my_var) -> in

3 -> out (integers are immutable i.e can't be changed)

Immutable or Mutable?
Immutable
- int
- float
- bool
- string
- bytes
- tuple
- frozenset
- None

Mutable
- list
- dict
- set
- bytearray
- objects
- functions
- almost everything else!

Mutable default arguments are dangerous!
def foo(var=[]):
    var.append(1)
    return var

foo() -> in

[1] -> out

foo() -> in

[1, 1] -> out

def foo(var=None):
    if var is None:
        var = []
    var.append(1)
    return var

foo() -> in

[1] -> out

foo() -> in

[1] -> out
'''


def store_lower(_dict, _string):
    """Add a mapping between `_string` and a lowercased version of `_string` to `_dict`

    Args:
        _dict (dict): The dictionary to update.
        _string (str): The string to add.
    """
    orig_string = _string
    _string = _string.lower()
    _dict[orig_string] = _string


d = {}
s = 'Hello'

store_lower(d, s)

''' (d = {'Hello': 'hello'}, s = 'Hello') -> output '''


def add_column(values, df=pandas.DataFrame()):
    """Add a column of `values` to a DataFrame `df`.
    The column will be named "col_<n>" where "n" is
    the numerical index of the column.

    Args:
        values (iterable): The values of the new column
        df (DataFrame, optional): The DataFrame to update.
            If no DataFrame is passed, one is created by default.

    Returns:
        DataFrame
    """
    df['col_{}'.format(len(df.columns))] = values
    return df

# OR

    # Use an immutable variable for the default argument


def better_add_column(values, df=None):
    """Add a column of `values` to a DataFrame `df`.
    The column will be named "col_<n>" where "n" is
    the numerical index of the column.

    Args:
        values (iterable): The values of the new column
        df (DataFrame, optional): The DataFrame to update.
            If no DataFrame is passed, one is created by default.

    Returns:
        DataFrame
    """
    # Update the function to create a default DataFrame
    if df is None:
        df = pandas.DataFrame()
    df['col_{}'.format(len(df.columns))] = values
    return df


''' Context Managers '''

'''
Using context managers
A context manager:
- Sets up a context
- Runs your code
- Removes the context

A real-world example
with open('my_file.txt') as my_file:
    text = my_file.read()
    length = len(text)

print('The file is {} characters long'.format(length))

open() does three things:
- Sets up a context by opening a file
- Lets you run any code you want on that file
- Removes the context by closing the file

Using a context manager
with <context-manager>(<args>) as <variable-name>:
    # Run your code here
    # This code is running 'inside the context'

# This code runs after the context is removed
'''

# Open "alice.txt" and assign the file to "file"
with open('alice.txt') as file:
    text = file.read()

n = 0
for word in text.split():
    if word.lower() in ['cat', 'cats']:
        n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))


image = get_image_from_instagram()

# Time how long process_with_numpy(image) takes to run
with timer():
    print('Numpy version')
    process_with_numpy(image)

# Time how long process_with_pytorch(image) takes to run
with timer():
    print('Pytorch version')
    process_with_pytorch(image)


'''
Writing context managers
Two ways to define a context manager
- Class-based
- Function-based

How to create a context manager
- There are five parts to creating a context manager

@contextlib.contextmanager
def my_context():
    # Add any set up code you need
    yield
    # Add any teardown code you need

1. Define a function.
2. (optional) Add any set up code your context needs.
3. Use the 'yield' keyword.
4. (optional) Add any teardown code your context needs.
5. Add the '@contextlib.contextmanager' decorator

The 'yield' keyword
@contextlib.contextmanager
def my_context():
    print('hello')
    yield 42
    print('goodbye')

with my_context() as foo:
    print('foo is {}'.format(foo)) -> in

hello
foo is 42
goodbye     -> out

Setup and teardown
@contextlib.contextmanager
def database(url):
    # set up database connection
    db = postgres.connect(url)

    yield db

    # tear down database connection
    db.disconnect()

url = 'http://datacamp.com/date'
with database(url) as my_db:
    course_list = my_db.execute(
        'SELECT * FROM courses'
    ) 

@contextlib.contextmanager
def in_dir(path):
    # save current working directory
    old_dir = os.getcwd()

    # switch to new working directory
    os.chdir(path)

    yield

    # change back to previous working directory
    os.chdir(old_dir)

with in_dir('/data/project_1/'):
    project_files = os.listdir() 


- The regular open() context manager:
* takes a filename and a mode ('r' for read, 'w' for write, or 'a' for append)
* opens the file for reading, writing, or appending
* yields control back to the context, along with a reference to the file
* waits for the context to finish
* and then closes the file before exiting
'''

# Add a decorator that will make timer() a context manager


@contextlib.contextmanager
def timer():
    """Time the execution of a context block.

    Yields:
        None
    """
    start = time.time()
    # Send control back to the context block
    yield
    end = time.time()
    print('Elapsed: {:.2f}s'.format(end - start))


with timer():
    print('This should take approximately 0.25 seconds')
    time.sleep(0.25)


@contextlib.contextmanager
def open_read_only(filename):
    """Open a file in read-only mode.

    Args:
        filename (str): The location of the file to read

    Yields:
        file object
    """
    read_only_file = open(filename, mode='r')
    # Yield read_only_file so it can be assigned to my_file
    yield read_only_file
    # Close read_only_file
    read_only_file.close()


with open_read_only('my_file.txt') as my_file:
    print(my_file.read())


'''
Advanced topics
Nested contexts
def copy(src, dst):
    """ Copy the contents of one file to another.

    Args:
        src (str): File name of the file to be copied.
        dst (str): Where to write the new file.
    """
    # Open the cource file and read in the contents
    with open(src) as f_src:
        contents = f_src.read()

def copy(src, dst):
    """ Copy the contents of one file to another.

    Args:
        src (str): File name of the file to be copied.
        dst (str): Where to write the new file.
    """
    # Open both files
    with open(src) as f_src:
        with open(dst, 'w') as f_dst:
            # Read and write each line, one at a time
            for line in f_src:
                f_dst.write(line)

Handling errors
def get_printer(ip):
    p = connect_to_printer(ip)
    
    try:
        yield

    # This MUST be called or no one else will be able to connect to the printer
    finally:
        p.disconnect()
        print('disconnected from printer')

doc = {'text': 'This is my text.'}

with get_printer('10.0.34.111') as printer:
    printer.print_page(doc['txt'] -> in

disconnected from printer
Traceback (most recent call last):
    File'<stdin>', line 1, in <module>
        printer.print_page(doc['txt'])
keyError: 'txt'     -> out

try:
    # code that might raise an error
except:
    # do something about the error
finally:
    # this code runs no matter what 

Context manager patterns
Open        :   Close
Lock        :   Release
Change      :   Reset
Enter       :   Exit
Start       :   Stop
Setup       :   Teardown
Connect     :   Disconnect
'''

# Use the "stock('NVDA')" context manager
# and assign the result to the variable "nvda"
with stock('NVDA') as nvda:
    # Open "NVDA.txt" for writing as f_out
    with open('NVDA.txt', 'w') as f_out:
        for _ in range(10):
            value = nvda.price()
            print('Logging ${:.2f} for NVDA'.format(value))
            f_out.write('{:.2f}\n'.format(value))


def in_dir(directory):
    """Change current working directory to `directory`, allow the user to run some code, and change back.

    Args:
        directory (str): The path to a directory to work in.
    """
    current_dir = os.getcwd()
    os.chdir(directory)

    # Add code that lets you handle errors
    try:
        yield
    # Ensure the directory is reset, whether there was an error or not
    finally:
        os.chdir(current_dir)


''' Decorators '''

'''
Functions are objects
Functions are just another type of object
Python objects:
def x():
    pass
x = [1, 2, 3]
x = {'foo': 42}
x = pandas.DataFrame()
x = 'This is a sentence.'
x = 3
x = 71.2
import x

Functions as variables
def my_function():
    print('Hello')
x = my_function
type(x) -> in

<type 'function'> -> out

x() -> in

Hello -> out

PrintMcPrintface = print
PrintMcPrintface('Python is awesome!') -> in

Python is awesome! -> out

Lists and dictionaries of functions
list_of_functions = [my_function, open, print]
list_of_functions[2]('I am printing with an element of a list!') -> in

I am printing with an element of a list! -> out

dict_of_functions = { 'func1': my_function, 'func2': open, 'func3': print }
dict_of_functions['func3']('I am printing with a value of a dict!') -> in

I am printing with a value of a dict! -> out

Referencing a function
def my_function():
    return 42

x = my_function
my_function() -> in

42 -> out

my_function -> in

<function my_function at 0x7f475332a730> -> out

Functions as arguments
def has_docstring(func):
    """Check to see if the function 'func' has a docstring.

    Args:
        func (callable): A function.

    Returns:
        bool
    """
    return func.__doc__ is not None

def no():
    return 42

has_docstring(no) -> in

False -> out

def yes():
    """ Return the value 42 """
    Return 42 

has_docstring(yes) -> in

True -> out

Defining a function inside another function
def foo():
    x = [3, 6, 9]

    def bar(y):
        print(y)

    for value in x:
        bar(x)

def foo(x, y):
    if x > 4 and x < 10 and y > 4 and y < 10:
        print(x * y)

# OR

def foo(x, y):
    def in_range(v):
        return v > 4 and v < 10

    if in_range(x) and in_range(y):
        print(x * y)

Functions as return values
def get_function():
    def print_me(s):
        print(s)

    return print_me

new_func = get_function()
new_func('This is a sentence.')

This is a sentence. -> out
'''

# Add the missing function references to the function map
function_map = {
    'mean': mean,
    'std': std,
    'minimum': minimum,
    'maximum': maximum
}

data = load_data()
print(data)

func_name = get_user_input()

# Call the chosen function and pass "data" as an argument
function_map[func_name](data)


def has_docstring(func):
    """Check to see if the function `func` has a docstring.

    Args:
        func (callable): A function.

    Returns:
        bool
    """
    return func.__doc__ is not None


# Call has_docstring() on the load_and_plot_data() function
ok = has_docstring(load_and_plot_data)

if not ok:
    print("load_and_plot_data() doesn't have a docstring!")
else:
    print("load_and_plot_data() looks ok")


# Call has_docstring() on the as_2D() function
ok = has_docstring(as_2D)

if not ok:
    print("as_2D() doesn't have a docstring!")
else:
    print("as_2D() looks ok")


# Call has_docstring() on the log_product() function
ok = has_docstring(log_product)

if not ok:
    print("log_product() doesn't have a docstring!")
else:
    print("log_product() looks ok")


def create_math_function(func_name):
    if func_name == 'add':
        def add(a, b):
            return a + b
        return add
    elif func_name == 'subtract':
        # Define the subtract() function
        def subtract(a, b):
            return a - b
        return subtract
    else:
        print("I don't know that one")


add = create_math_function('add')
print('5 + 2 = {}'.format(add(5, 2)))

subtract = create_math_function('subtract')
print('5 - 2 = {}'.format(subtract(5, 2)))


'''
Scope
x = 7
y = 200
print(x) -> in

7 -> out

def foo():
    x = 42
    print(x)
    print(y)

foo() -> in

42
200 -> out

print(x) -> in

7 -> out

Local scope
def foo():
    x = 42
    print(x)

which x?

Local scope -> NonLocal scope (Parent function) -> Global scope -> Builtin scope (e.g print)

The global keyword
x = 7

def foo():
    x = 42
    print(x)

foo() -> in 

42 -> out

print(x) -> in

7 -> out

# OR

x = 7

def foo():
    global x
    x = 42
    print(x)

foo() -> in 

42 -> out

print(x) -> in

42 -> out

The nonlocal keyword
def foo():
    x = 10

    def bar():
        x = 200
        print(x)

    bar()
    print(x)

foo() -> in

200
10 -> out

# OR

def foo():
    x = 10

    def bar():
        nonlocal x
        x = 200
        print(x)

    bar()
    print(x)

foo() -> in

200
200 -> out
'''

call_count = 0


def my_function():
    # Use a keyword that lets us update call_count
    global call_count
    call_count += 1

    print("You've called my_function() {} times!".format(
        call_count
    ))


for _ in range(20):
    my_function()


def read_files():
    file_contents = None

    def save_contents(filename):
        # Add a keyword that lets us modify file_contents
        nonlocal file_contents
        if file_contents is None:
            file_contents = []
        with open(filename) as fin:
            file_contents.append(fin.read())

    for filename in ['1984.txt', 'MobyDick.txt', 'CatsEye.txt']:
        save_contents(filename)

    return file_contents


print('\n'.join(read_files()))


def wait_until_done():
    def check_is_done():
        # Add a keyword so that wait_until_done() doesn't run forever
        global done
        if random.random() < 0.1:
            done = True

    while not done:
        check_is_done()


done = False
wait_until_done()

print('Work done? {}'.format(done))


'''
Closures
- A closure is a tuple of variables that are no longer in scope, but that a function needs in order to run.

Attaching nonlocal variables to nested functions
def foo():
    a = 5
    def bar():
        print(a)
    return bar

func = foo()

func() -> in

5 -> out

type(func.__closure__) -> in
<class 'tuple'> -> out

len(func.__closure__) -> in
1 -> in

func.__closure__[0].cell_contents -> in
5 -> out

Closures and deletion
x = 25

def foo(value):
    def bar():
        print(value)
    return bar

my_func = foo(x)
my_func() -> in

25 -> out

del(x)
my_func() -> in

25 -> out

len(my_func.__closure__) -> in
1 -> in

my_func.__closure__[0].cell_contents -> in
25 -> out

Closures and overwriting
x = 25

def foo(value):
    def bar():
        print(value)
    return bar

x = foo(x)
x() -> in

25 -> out

len(x.__closure__) -> in
1 -> in

x.__closure__[0].cell_contents -> in
25 -> out

Definitions - nested function
- Nested function: A function defined inside another function.

# outer function
def parent():
    # nested function
    def child():
        pass
    return child

Definitions - nonlocal variables
Nonlocal variables: Variables defined in the parent function that are used by the child function,

def parent(arg_1, arg_2):
    # From child()'s point of view, 'value' and 'my_dict' are nonlocal variables, as are 'arg_1' and 'arg_2'.
    value = 22
my_dict = {'chocolate': 'yummy'}

    def child():
        print(2 * value)
        print(my_dict['chocolate'])
        print(arg_1 + arg_2)

    return child

Closure: Nonlocal variables attached to a returned function 
new_function = parent(3, 4)

print([cell.cell_contents for cell in new_function.__closure__])

Why does all of this matter?
Decorator use:
- Functions as objects
- Nested functions
- Nonlocal scope
- Closures
'''


def return_a_func(arg1, arg2):
    def new_func():
        print('arg1 was {}'.format(arg1))
        print('arg2 was {}'.format(arg2))
    return new_func


my_func = return_a_func(2, 17)

print(my_func.__closure__ is not None)
print(len(my_func.__closure__) == 2)

# Get the values of the variables in the closure
closure_values = [my_func.__closure__[i].cell_contents for i in range(2)]
print(closure_values == [2, 17])


def my_special_function():
    print('You are running my_special_function()')


def get_new_func(func):
    def call_func():
        func()
    return call_func


new_func = get_new_func(my_special_function)

# Redefine my_special_function() to just print "hello"


def my_special_function():
    print('hello')


new_func()


def my_special_function():
    print('You are running my_special_function()')


def get_new_func(func):
    def call_func():
        func()
    return call_func


new_func = get_new_func(my_special_function)

# Delete my_special_function()
del (my_special_function)

new_func()


def my_special_function():
    print('You are running my_special_function()')


def get_new_func(func):
    def call_func():
        func()
    return call_func


# Overwrite `my_special_function` with the new function
my_special_function = get_new_func(my_special_function)

my_special_function()


'''
Decorators
- This is a wrapper that can be placed around a function that changes the function's behaviour.

what does a decorator look like?
@double_args
def multiply(a, b):
    return a * b
multiply(1, 5) -> in

20 -> out

The double_args decorator
def multiply(a, b):
    return a * b 
def double_args(func):
    return func

new_multiply = double_args(multiply)
new_multiply(1, 5) -> in

5 -> out

multiply(1, 5) -> in

5 -> out

def multiply(a, b):
    return a * b 
def double_args(func):
    # Define a new function that we can modify
    def wrapper(a, b):
        # For now, just call the unmodified function
        return func(a, b)
    # Return the new function
    return wrapper

new_multiply = double_args(multiply)
new_multiply(1, 5) -> in

5 -> out

def multiply(a, b):
    return a * b 
def double_args(func):
    def wrapper(a, b):
        # Call the passed in function, but double each argument
        return func(a * 2, b * 2)
    return wrapper

new_multiply = double_args(multiply)
new_multiply(1, 5) -> in

20 -> out

def multiply(a, b):
    return a * b 
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper

multiply = double_args(multiply)
multiply(1, 5) -> in

20 -> out

multiply.__closure__[0].cell_contents -> in

<function multiply at 0x7f0060c9e620> -> out

Decorator syntax
def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper

def multiply(a, b):
    return a * b 

multiply = double_args(multiply)
multiply(1, 5) -> in

20 -> out

# OR

def double_args(func):
    def wrapper(a, b):
        return func(a * 2, b * 2)
    return wrapper

@double_args
def multiply(a, b):
    return a * b 

multiply(1, 5) -> in

20 -> out 
'''


def my_function(a, b, c):
    print(a + b + c)


# Decorate my_function() with the print_args() decorator
my_function = print_args(my_function)

my_function(1, 2, 3)


# Decorate my_function() with the print_args() decorator
@print_args
def my_function(a, b, c):
    print(a + b + c)


my_function(1, 2, 3)


def print_before_and_after(func):
    def wrapper(*args):
        print('Before {}'.format(func.__name__))
        # Call the function being decorated with *args
        func(*args)
        print('After {}'.format(func.__name__))
    # Return the nested function
    return wrapper


@print_before_and_after
def multiply(a, b):
    print(a * b)


multiply(5, 10)


''' More on Decorators '''

'''
Real-world examples
Time a function
import time

def timer(func):
    """ A decorator that prints how long a function took to run.

    Args:
        func (callable): The function being decorated.

    Returns:
        callable: The decorated function.
    """
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # When wrapper() is called, get the current time.
        t_start = time.time()
        # Call the decorated function and store the result.
        result = func(*args, **kwargs)
        # Get the total time it took to run, and print it.
        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))
        return result
    return wrapper

Using timer()
@timer
def sleep_n_seconds(n):
    time.sleep(n)

sleep_n_seconds(5) -> in

sleep_n_seconds took 5.0050950050354s -> out 

sleep_n_seconds(10) -> in

sleep_n_seconds took 10.0100677013397222s -> out 

- Memoizing is the process of storing the results of a function so that the next time the function is called with the same arguments; you can just look up the answer.

def memoize(func):
    """Store the results of the decorated function for fast lookup"""
    # Store results in a dict that maps arguments to results
    cache = {}
    # Define the wrapper function to return.
    def wrapper(*args, **kwargs):
        # If these arguments haven't been seen before,
        if *args, kwargs) not in cache:
            # Call func() and store the result.
            cache[(args, kwargs)] = fuc(*args, **kwargs)
        return cache[(args, kwargs)]
    return wrapper

@memoize
def slow_function(a, b):
    print('Sleeping...')
    time.sleep(5)
    return a + b

slow_function(3, 4) -> in

Sleeping...
7 -> out

slow_function(3, 4) -> in

7 -> out

When to use decorators
- Add common behaviour to multiple functions

@timer
def foo():
    # do some computation

@timer
def bar():
    # do some computation

@timer
def baz():
    # do some computation
'''


def print_return_type(func):
    # Define wrapper(), the decorated function
    def wrapper(*args, **kwargs):
        # Call the function being decorated
        result = func(*args, **kwargs)
        print('{}() returned type {}'.format(
            func.__name__, type(result)
        ))
        return result
    # Return the decorated function
    return wrapper


@print_return_type
def foo(value):
    return value


print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))


def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        # Call the function being decorated and return the result
        return func(*args, **kwargs)
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

# Decorate foo() with the counter() decorator


@counter
def foo():
    print('calling foo()')


foo()
foo()

print('foo() was called {} times.'.format(foo.count))


'''
Decorators and metadata
- One of the problems with decorators is that they obscure the decorated function's metadata.

def sleep_n_seconds(n=10):
    "Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for.
    """
    time.sleep(n)
print(sleep_n_seconds.__doc__) -> in

Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for. -> out

print(sleep_n_seconds.__name__) -> in

sleep_n_seconds -> out

print(sleep_n_seconds.__defaults__) -> in

(10,) -> out

@timer
def sleep_n_seconds(n=10):
    "Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for.
    """
    time.sleep(n)
print(sleep_n_seconds.__doc__) -> in

-> out

print(sleep_n_seconds.__name__) -> in

wrapper -> out

from functools import wraps
def timer(func):
    """A decorator that prints how long a function took to run."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        t_start = time.time()

        result = func(*args, **kwargs)

        t_total = time.time() - t_start
        print('{} took {}s'.format(func.__name__, t_total))

        return result
    return wrapper

@timer
def sleep_n_seconds(n=10):
    "Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for.
    """
    time.sleep(n)
print(sleep_n_seconds.__doc__) -> in

Pause processing for n seconds.

    Args:
        n (int): The number of seconds to pause for. -> out

print(sleep_n_seconds.__name__) -> in

sleep_n_seconds -> out

print(sleep_n_seconds.__defaults__) -> in

(10,) -> out

Access to the original function 
sleep_n_seconds.__wrapped__ -> in

<function sleep_n_seconds at 0x7f52cab44ae8>-> out
'''


def add_hello(func):
    # Decorate wrapper() so that it keeps func()'s metadata
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Print 'hello' and then call the decorated function."""
        print('Hello')
        return func(*args, **kwargs)
    return wrapper


@add_hello
def print_sum(a, b):
    """Adds two numbers and prints the sum"""
    print(a + b)


print_sum(10, 20)
print_sum_docstring = print_sum.__doc__
print(print_sum_docstring)


def check_everything(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        check_inputs(*args, **kwargs)
        result = func(*args, **kwargs)
        check_outputs(result)
        return result
    return wrapper


@check_everything
def duplicate(my_list):
    """Return a new list that repeats the input twice"""
    return my_list + my_list


t_start = time.time()
duplicated_list = duplicate(list(range(50)))
t_end = time.time()
decorated_time = t_end - t_start

t_start = time.time()
# Call the original function instead of the decorated one
duplicated_list = duplicate.__wrapped__(list(range(50)))
t_end = time.time()
undecorated_time = t_end - t_start

print('Decorated time: {:.5f}s'.format(decorated_time))
print('Undecorated time: {:.5f}s'.format(undecorated_time))


'''
Decorators that take arguments
def run_three_times(func):
    def wrapper(*args, **kwargs):
        for i in range(3):
            func(*args, **kwargs)
    return wrapper

@run_three_times
def print_sum(a, b):
    print(a + b)
print_sum(3, 5) -> in

8
8
8 -> out

A decorator factory
def run_n_times(n):
    """Define and return a decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(n):
        return wrapper
    return decorator

@run_n_times(3)
def print_sum(a, b):
    print(a + b)
print_sum(3, 5) -> in

8
8
8 -> out

@run_n_times(5)
def print_hello():
    print('Hello!')
print_hello() -> in

Hello!
Hello!
Hello!
Hello!
Hello! -> out
'''


def run_n_times(n):
    """Define and return a decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator


# Make print_sum() run 10 times with the run_n_times() decorator
@run_n_times(10)
def print_sum(a, b):
    print(a + b)


print_sum(15, 20)


# Use run_n_times() to create the run_five_times() decorator
run_five_times = run_n_times(5)


@run_five_times
def print_sum(a, b):
    print(a + b)


print_sum(4, 100)


# Modify the print() function to always run 20 times
print = run_n_times(20)(print)

print('What is happening?!?!')


def bold(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        msg = func(*args, **kwargs)
        return '<b>{}</b>'.format(msg)
    return wrapper


def italics(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        msg = func(*args, **kwargs)
        return '<i>{}</i>'.format(msg)
    return wrapper

# OR


def html(open_tag, close_tag):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = func(*args, **kwargs)
            return '{}{}{}'.format(open_tag, msg, close_tag)
        # Return the decorated function
        return wrapper
    # Return the decorator
    return decorator

# Make hello() return bolded text


@html('<b>', '</b>')
def hello(name):
    return 'Hello {}!'.format(name)


print(hello('Alice'))

# Make goodbye() return italicized text


@html('<i>', '</i>')
def goodbye(name):
    return 'Goodbye {}.'.format(name)


print(goodbye('Alice'))

# Wrap the result of hello_goodbye() in <div> and </div>


@html('<div>', '</div>')
def hello_goodbye(name):
    return '\n{}\n{}\n'.format(hello(name), goodbye(name))


print(hello_goodbye('Alice'))


'''
Timeout(): a real world example
Timeout - background info
import signal
def raise_timeout(*args, **kwargs):
    raise TimeoutError()
# When an 'alarm' signal goes off, call raise_timeout()
signal.signal(signalnum = signal.SIGALRM, handler = raise_timeout)
# Set off an alarm in 5 seconds
signal.alarm(5)
# Cancel the alarm
signal.alarm(0)

def timeout_in_5s(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set an alarm for 5 seconds
        signal.alarm(5)
        try:
            # Call the decorated func
            return func(*args, **kwargs)
        finally:
            # Cancel alarm
            signal.alarm(0)
    return wrapper

@timeout_in_5s
def foo():
    time.sleep(10)
    print('foo!')

foo() -> in

TimeoutError -> out

def timeout(n_seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set an alarm for 5 seconds
            signal.alarm(5)
            try:
                # Call the decorated func
                return func(*args, **kwargs)
            finally:
                # Cancel alarm
                signal.alarm(0)
        return wrapper
    return decorator

@timeout(5)
def foo():
    time.sleep(10)
    print('foo!')

foo() -> in

TimeoutError -> out


@timeout(20)
def bar():
    time.sleep(10)
    print('bar!')

bar()

bar! -> out
'''


def tag(*tags):
    # Define a new decorator, named "decorator", to return
    def decorator(func):
        # Ensure the decorated function keeps its metadata
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call the function being decorated and return the result
            return func(*args, **kwargs)
            wrapper.tags = tags
        return wrapper
    # Return the new decorator
    return decorator


@tag('test', 'this is a tag')
def foo():
    pass


print(foo.tags)


def returns_dict(func):
    # Complete the returns_dict() decorator
    def wrapper(*args, **kwargs):
        result = AssertionError
        assert type(result) == dict
        return result
    return wrapper


@returns_dict
def foo(value):
    return value


try:
    print(foo([1, 2, 3]))
except AssertionError:
    print('foo() did not return a dict!')


def returns(return_type):
    # Complete the returns() decorator
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = AssertionError
            assert type(result) == return_type
            return result
        return wrapper
    return decorator


@returns(dict)
def foo(value):
    return value


try:
    print(foo([1, 2, 3]))
except AssertionError:
    print('foo() did not return a dict!')
