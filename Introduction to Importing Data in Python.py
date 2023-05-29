""" Introduction and flat files """

"""
Welcome to the course!
Import data
- Flat files, e.g .txts and .csvs
- Files from other software e.g Excel Spreadsheets, Stata, SAS and MATLAB
- Relational databases e.g SQLite & PostgreSQL

Reading a text file
e.g
filename = 'huck_finn.txt'
file = open( filename, mode='r' )   # 'r' is to read
text = file.read()                  # to open a connection to the file
file.close()                        # to close the connection to the file
print( text )                       # Printing a text file

Writing to a file
e.g
filename = 'huck_finn.txt'
file = open( filename, mode='w' )   # 'w' is to write
file.close()

Context manager with
e.g
with open('huck_finn.txt', 'r') as file:
    print(file.read())
"""

# Open a file: file
import matplotlib.pyplot as plt
file = open("moby_dick.txt", mode="r")

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)


# Read & print the first 3 lines
with open("moby_dick.txt") as file:
    print(file.readline())  # print only the first few lines
    print(file.readline())
    print(file.readline())


"""
The importance of flat files in data science
Flat files
- e.g titanic.csv
- They are basic text files containing records
- Record: row of fields or attributes, each of which contains at most one item of information.
- Column: i.e feature or attribute
- i.e they are Table data, without structured relationships.

Header
Flat files may have a header which is a row that occurs as the first row and decribes the contents of the data columns or states what the corresponding attributes or features in each column are.

File extension
- .csv - Comma Separated Values
- .txt - Text file
- commas, tabs - Delimiters

How do you import flat files?
- 2 main packages: NumPy, Pandas

import this <- in

The Zen of Python, by Tim Peters
* Beautiful is better than ugly.
* Explicit is better than implicit.
* Simple is better than complex.
* Complex is better than complicated.
* Flat is better than nested.
* Sparse is better than dense.
* Readability counts.
* Special cases aren't special enough to break the rules.
* Although practicality beats purity.
* Errors should never pass silently.
* Unless explicitly silenced.
* In the face of ambiguity, refuse the temptation to guess.
* There should be one-- and preferably only one --obvious way to do it.
* Although that way may not be obvious at first unless you're Dutch.
* Now is better than never.
* Although never is often better than *right* now.
* If the implementation is hard to explain, it's a bad idea.
* If the implementation is easy to explain, it may be a good idea.
* Namespaces are one honking great idea -- let's do more of those! <- out
"""


"""
Importing flat files using NumPy
- This uses the package NumPy to import the data as a NumPy array ( i.e if all the data are numerical).

Why NumPy?
- NumPy arrays are Python standards for storing numerical data.
- They are often essential for other packages: e.g. scikit-learn
- Numpy functions to import data as arrays
* loadtxt() - Best used for basic cases
* genfromtxt() - Best used when there are mixed datatypes
* np.recfromcsv() - Behaves like genfromtxt() but has the defaults delimiter=',' and names=True in addition to dtype=None

Importing flat files using NumPy
e.g
import numpy as np
filename = 'MNIST_header.txt'
data = np.loadtxt( filename, delimiter=',', skiprows=1, usecols=[0, 2], dtype=str )
print(data)

NB: The default delimiter is whitespace ( ' ' )
* You can use ',' for comma-delimited
* You can use '\t' for tab-delimited
- skiprows: if the data consists of numerics and  the header has strings in it
- Usecols: used to select columns of the data to use ( i.e [ 0, 2 ] = 1st and 3rd )
- dtype: used to import different datatypes into NumPy arrays
"""

# Import package

# Assign filename to variable: file
file = "digits.csv"

# Load file as array: digits
digits = np.loadtxt(file, delimiter=",")

# Print datatype of digits
print(type(digits))

# Select and reshape a row
im = digits[21, 1:]
im_sq = np.reshape(im, (28, 28))

# Plot reshaped data (matplotlib.pyplot already loaded as plt)
plt.imshow(im_sq, cmap="Greys", interpolation="nearest")
plt.show()


# Import numpy

# Assign the filename: file
file = "digits_header.txt"

# Load the data: data
data = np.loadtxt(file, delimiter="\t", skiprows=1, usecols=[0, 2])

# Print data
print(data)


# Assign filename: file
file = "seaslug.txt"

# Import file: data
data = np.loadtxt(file, delimiter="\t", dtype=str)

# Print the first element of data
print(data[0])

# Import data as floats and skip the first row: data_float
data_float = np.loadtxt(file, delimiter="\t", dtype=float, skiprows=1)

# Print the 10th element of data_float
print(data_float[9])

# Plot a scatterplot of the data
plt.scatter(data_float[:, 0], data_float[:, 1])
plt.xlabel("time (min.)")
plt.ylabel("percentage of larvae")
plt.show()


# Import file: data
data = np.genfromtxt("titanic.csv", delimiter=",", names=True, dtype=None)

# Subsetting data
data["Survived"]


# Assign the filename: file
file = "titanic.csv"

# Import file using np.recfromcsv: d
d = np.recfromcsv(file)

# Print out first three entries of d
print(d[:3])


"""
Importing flat files using pandas
What a data scientist needs
- 2-dimensional labeled data structure(s)
- Columns of potentially different types
- Manipulate, slice, reshape, groupby, join, merge
- Perform statistics
- Work with time series data

Pandas and the DataFrame
- DataFrame = pythonic analog of R's data frame
- A matrix has rows and columns. A data frame has observations and variables.

Manipulating pandas DataFrames
they can be useful in all steps of the data scientific method, i.e
- Exploratory data analysis
- Data wrangling
- Data preprocessing
- Building models
- Visualization 
NB: Its now standard and best practice to use pandas to import flat files as dataframes
- Missing values are also commonly referred to as NA or NaN
- The sep (the pandas version of delim)
- Comment takes characters that comments occur after in the file e.g '#'
- na_values takes a list of strings to recognize as NA/NaN e.g 'Nothing'

Importing using pandas
import pandas as pd
filename = 'winequality-read.csv'
data = pd.read_csv( filename )
data.head()                 # print first 5 rows of the DataFrame

data_array = data.values    # to convert the dataframe to a numpy array  
"""

# Import pandas as pd

# Assign the filename: file
file = "titanic.csv"

# Read the file into a DataFrame: df
df = pd.read_csv(file)

# View the head of the DataFrame
print(df.head())


# Assign the filename: file
file = "digits.csv"

# Read the first 5 rows of the file into a DataFrame: data
data = pd.read_csv(file, nrows=5, header=None)

# Build a numpy array from the DataFrame: data_array
data_array = data.values

# Print the datatype of data_array to the shell
print(type(data_array))


# Import matplotlib.pyplot as plt

# Assign filename: file
file = "titanic_corrupt.txt"

# Import file: data
data = pd.read_csv(file, sep="\t", comment="#", na_values="Nothing")

# Print the head of the DataFrame
print(data.head())

# Plot 'Age' variable in a histogram
pd.DataFrame.hist(data[["Age"]])
plt.xlabel("Age (years)")
plt.ylabel("count")
plt.show()


""" Importing data from other file types """

"""
Introduction to other file types
Other file types
- Excel spreadsheets
- MATLAB files
- SAS files
- Stata files
- HDF5 files

NB: The first line of the following code imports the library os, the second line stores the name of the current directory in a string called wd and the third outputs the contents of the directory in a list to the shell.
#
import os
wd = os.getcwd()
os.listdir(wd)

Pickled files
- This file type is native to Python
- Motivation: many datatypes for which it isn't obvious how to store them
- Pickled files are serialized
- Serialize = convert object to bytestream

Pickled files
import pickle
with open( 'pickled_fruit.pkl', 'rb') as file:      # 'rb' is used to specify that the file is both read only and binary
    data = pickle.load(file)
print(data)

Importing Excel spreadsheets
import pandas as pd
file = 'urbanpop.xlsx'
data =  pd.ExcelFile(file)                          # to read excel files
print( data.sheet_names ) <- in                     # to get the sheet names
['1960-1966', '1967-1974', '1975-2011'] <- out      # The sheet names

df1 = data.parse('1960-1966')       # to load a particular sheet as a dataframe using sheet name, as a string
df2 = data.parse(0)                 # to load a particular sheet as a dataframe using sheet index, as a int

NB: additional arguments to df.parse() 
* skiprows      - To skip rows
* names         - To name the columns
* usecols       - To designate which columns to parse
"""

# Import pickle package

# Open pickle file and load data: d
with open("data.pkl", "rb") as file:
    d = pickle.load(file)

# Print d
print(d)

# Print datatype of d
print(type(d))


# Import pandas

# Assign spreadsheet filename: file
file = "battledeath.xlsx"

# Load spreadsheet: xls
xls = pd.ExcelFile(file)

# Print sheet names
print(xls.sheet_names)

# Load a sheet into a DataFrame by name: df1
df1 = xls.parse("2004")

# Print the head of the DataFrame df1
print(df1.head())

# Load a sheet into a DataFrame by index: df2
df2 = xls.parse(0)

# Print the head of the DataFrame df2
print(df2.head())

# Parse the first sheet and rename the columns: df1
df1 = xls.parse(0, skiprows=[0], names=["Country", "AAM due to War (2002)"])

# Print the head of the DataFrame df1
print(df1.head())

# Parse the first column of the second sheet and rename the column: df2
df2 = xls.parse(1, usecols=[0], skiprows=[0], names=["Country"])

# Print the head of the DataFrame df2
print(df2.head())


"""
Importing SAS/Stata files using pandas
SAS and Stata files
- SAS: Statistical Analysis System
- Stata: 'Statistics' + 'data'
- SAS: Used in Business analytics and Biostatistics
- Stata: Used in Academic social sciences research such as Economincs and Epidemiology

SAS files
- Used for:
* Advanced analytics
* Multivariate analysis
* Business intelligence
* Data management
* Predictive analytics
- Its a standard for computational analysis
- The most common SAS files have the extension
* .sas7bdat - Dataset files
* .sas7bcat - Catalog files

Importing SAS files
import pandas as pd
from sas7bdat import SAS7BDAT
with SAS7BDAT( 'urbanpop.sas7bdat' )as file:
    df_sas = file.to_data_frame()

Importing Stata files
import pandas as pd
data = pd.read_stata('urbanpop.dta')
"""

# Import sas7bdat package

# Save file to a DataFrame: df_sas
with SAS7BDAT("sales.sas7bdat") as file:
    df_sas = file.to_data_frame()

# Print head of DataFrame
print(df_sas.head())

# Plot histogram of DataFrame features (pandas and pyplot already imported)
pd.DataFrame.hist(df_sas[["P"]])
plt.ylabel("count")
plt.show()


# Import pandas

# Load Stata file into a pandas DataFrame: df
df = pd.read_stata("disarea.dta")

# Print the head of the DataFrame df
print(df.head())

# Plot histogram of one column of the DataFrame
pd.DataFrame.hist(df[["disa10"]])
plt.xlabel("Extent of disease")
plt.ylabel("Number of countries")
plt.show()


"""
Importing HDF5 files
- HDF5: Hierarchical Data Format version 5
- Standard for storing large quantities of numerical data
- Datasets can be hundreds of gigabytes or terabytes
- HDF5 can scale to exabytes

Importing HDF5 files
import h5py
filename = 'H-H1_LOSC_4_V1-815411200-4096.hdf5'
data = h5py.File( filename, 'r' ) # 'r' is to read only
print(type(data))

The structure of HDF5 files
- The contents of this file can be explored just like python dictionaries.
e.g
for key in data.key(): 
    print(key)              <- in
meta
quality
strain                      <- out  

print(type(data['meta']))           <- in
<class 'h5py._hl.group.Group'>      <- out

for key in data['meta'].keys():
    print(key)      <- in
Description
DescriptionURL
Detector
Duration
GPSstart
Observatory
Type
UTCstart            <- out

print( np.array(data['meta']['Description']), np.array(data['meta']['Detector']) )      <- in
b'Strain data time series from LIGO'  b'H1'                                             <- out
"""

# Import packages

# Assign filename: file
file = "LIGO_data.hdf5"

# Load file: data
data = h5py.File(file, "r")

# Print the datatype of the loaded file
print(type(data))

# Print the keys of the file
for key in data.keys():
    print(key)


# Get the HDF5 group: group
group = data["strain"]

# Check out keys of group
for key in group.keys():
    print(key)

# Set variable equal to time series data: strain
strain = np.array(data["strain"]["Strain"])

# Set number of time points to sample: num_samples
num_samples = 10000

# Set time vector
time = np.arange(0, 1, 1 / num_samples)

# Plot data
plt.plot(time, strain[:num_samples])
plt.xlabel("GPS Time (s)")
plt.ylabel("strain")
plt.show()


"""
Importing MATLAB files
- MATLAB: 'Matrix Laboratory'
- It is a numerical computing environment that is an industry standard in engineering and science
- Data saved as .mat files

SciPy to the rescue!
- scipy.io.loadmat() - read .mat files
- scipy.io.savemat() - write .mat files

Importing a .mat file
import scipy.io
filename = 'workspace.mat'
mat = scipy.io.loadmat( filename )
print( type(mat) )      <- in
<class 'dict'>          <- out

* keys = MATLAB variable names
* values = objects assigned to variables
e.g
print( type(mat['x']) )     <- in
<class 'numpy.ndarray'>     <- out
i.e mat['x'] is a numpy corresponding to the MATLAB arry x in your MATLAB workspace
"""

# Import package

# Load MATLAB file: mat
mat = scipy.io.loadmat("albeck_gene_expression.mat")

# Print the datatype type of mat
print(type(mat))


# Print the keys of the MATLAB dictionary
print(mat.keys())

# Print the type of the value corresponding to the key 'CYratioCyt'
print(type(mat["CYratioCyt"]))

# Print the shape of the value corresponding to the key 'CYratioCyt'
print(np.shape(mat["CYratioCyt"]))

# Subset the array and plot it
data = mat["CYratioCyt"][25, 5:]
fig = plt.figure()
plt.plot(data)
plt.xlabel("time (min.)")
plt.ylabel("normalized fluorescence (measure of expression)")
plt.show()


""" Working with relational databases in Python """

"""
Introduction to relational databases
What is a relational database?
- It's a type of database that is based on the Relational mode of data
- First described by Edgar 'Ted' Codd

Database
- It consists of tables ( e.g Orders table, Customers table and Employees table in a fictional Northwind database )

Tables
- It generally represents one entity type, such as order and it looks like a data frame
- In a relational database table, each row or record represents an instance of the entity type while each column represents an attribute of each instance
- It is essential that each row contains a unique identifier, known as a primary key, that can be used to explicitly access the row in question ( e.g OrderID ).
- The tables in a Relational Database are linked

Relational model
- It was proposed by Ted Codd and has been widely adopted
- It most neatly summarized into Codd's 12 Rules / Commandments
* It Consists of 13 rules ( but its zero-indexed! )
* The rules were defined to describe what a Relational Database Management System should adhere to, to be considered relational

Relational Database Management Systems
examples include
- PostgreSQL
- MySQL
- SQLite
* SQL = Structured Query Language
* SQL describes how you communicate with a datbase in order to both access and update the infromation it contains
* Querying is just a fancy way of saying, getting data out from the database
"""


"""
Creating a database engine in Python
Creating a database engine
- We will use a SQLite database
* because its fast and simple
- Packages that can be used to access an SQLite database include
* SQLite3
* SQLAlchemy - preferred because it works with many Relational Database Management Systems

First create SQLite engine
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')

Then, Getting table names
from sqlalchemy import create_engine
engine = create_engine('sqlite:///Northwind.sqlite')

table_names = engine.table_names()
print( table_names )
"""

# Import necessary module

# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")


# Import necessary module

# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Save the table names to a list: table_names
table_names = engine.table_names()

# Print the table names to the shell
print(table_names)


"""
Querying relational databases in Python
Basic SQL query
SELECT * FROM Table_name
- Returns all columns of all rows of the table
e.g
SELECT * FROM Orders - This are SQL query

- We can query in python, SQLAlchemy and also use Pandas to store the results of the queries

Workflow of SQL querying
- Import packages and functions
- Create the database engine
- Connect to the engine
- Query the database
- Save query results to a DataFrame
- Close the conection

Your first SQL query
from sqlalchemy import create_engine # import packages and functions
import pandas as pd # import packages and functions
engine = create_engine('sqlite:///Northwind.sqlite')        # Create the engine

con =  engine.connect()                                     # Connect to the engine
rs = con.execute( 'SELECT * FROM Ordes' )                   # Query the database
df = pd.DataFrame( rs.fetchall() )                          # Save query results to a DataFrame ( .fechall() fetches all rows )
df.columns = rs.keys()                                      # To retain the column names

con.close()                                                 # close the connection
print( df.head() )

Using the context manager 
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect() as con:
    rs = con.execute( 'SELECT OrderID, OrderDate, ShipName FROM Orders' )   # import only the selected columns fromthe table
    df = pd.DataFrame( rs.fetchmany(size=5) )                               # to import 5 rows 
    df.columns = rs.keys()
"""

# Import packages

# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Open engine connection: con
con = engine.connect()

# Perform query: rs
rs = con.execute("SELECT * FROM Album")

# Save results of the query to DataFrame: df
df = pd.DataFrame(rs.fetchall())

# Close connection
con.close()

# Print head of DataFrame df
print(df.head())


# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT LastName, Title FROM Employee")
    df = pd.DataFrame(rs.fetchmany(size=3))
    df.columns = rs.keys()

# Print the length of the DataFrame df
print(len(df))

# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee WHERE EmployeeID >= 6")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print the head of the DataFrame df
print(df.head())


# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Open engine in context manager
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Employee ORDER BY BirthDate")
    df = pd.DataFrame(rs.fetchall())

    # Set the DataFrame's column names
    df.columns = rs.keys()

# Print head of DataFrame
print(df.head())


"""
Querying relational databases directly with pandas
The Pandas way to query
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

with engine.connect() as con:
    rs = con.execute( 'SELECT * FROM Orders' )
    df = pd.DataFrame( rs.fetchall() )
    df.columns = rs.keys()

# OR

df = pd.read_sql_query( 'SELECT * FROM Orders', engine )
"""

# Import packages

# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Execute query and store records in DataFrame: df
df = pd.read_sql_query("SELECT * FROM Album", engine)

# Print head of DataFrame
print(df.head())


# Open engine in context manager and store query result in df1
with engine.connect() as con:
    rs = con.execute("SELECT * FROM Album")
    df1 = pd.DataFrame(rs.fetchall())
    df1.columns = rs.keys()

# Confirm that both methods yield the same result
print(df.equals(df1))


# Import packages

# Create engine: engine
engine = create_engine("sqlite:///Chinook.sqlite")

# Execute query and store records in DataFrame: df
df = pd.read_sql_query(
    "SELECT * FROM Employee WHERE EmployeeId >= 6 ORDER BY BirthDate", engine
)

# Print head of DataFrame
print(df.head())


"""
Advanced querying: exploiting table relationships
JOINing tables
e.g
INNER JOIN in Python (Pandas)
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('sqlite:///Northwind.sqlite')

df = pd.read_sql_query( 'SELECT OrderID, CompanyName FROM Orders INNER JOIN Customers on Orders.CustomerID = Customers.CustomerID', engine )
print( df.head() )
"""

# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute(
        "SELECT Title, Name FROM Album INNER JOIN Artist on Album.ArtistID = Artist.ArtistID"
    )
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())


# Execute query and store records in DataFrame: df
df = pd.read_sql_query(
    "SELECT * FROM PlaylistTrack INNER JOIN Track on PlaylistTrack.TrackId = Track.TrackId WHERE Milliseconds < 250000",
    engine,
)

# Print head of DataFrame
print(df.head())
