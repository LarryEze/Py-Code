''' Introduction '''

'''
Machine Learning & Spark
- A Regression model learns to predict a number.
- A Classification model, predicts a discrete or categorical value.

Data in RAM
- The performance of a Machine Learning model depends on data.
* In general, more data is a good thing.
- If an algorithm is able to train on a larger set of data, then its ability to generalize to new data will inevitably improve.
- If the data can fit entirely into RAM, then the algorithm can operate efficiently.

Data exceeds RAM
- If the data no longer fit into memory, the computer will start to use *virtual memory* and data will be *paged* back and forth between RAM and disk.
* Relative to RAM access, retrieving data from disk is slow.
- As the size of the data grows, paging becomes more intense and the computer begins to spend more and more time waiting for data.
* Performance plummets.

Data distributed across a cluster
- To deal with truly large datsets!
* Distribute the problem across multiple computers in a cluster.
- Rather than trying to handle a large dataset on a single machine, it's divided up into partitions which are processed separately.
* Ideally each data partition can fit into RAM on a single computer in the cluster

What is Spark?
- It is a general purpose framework for cluster computing.
- It is popular for 2 main reasons:
* It's generally much faster than other Big Data technologies like Hadoop, because it does most processing in memory
* It has a developer friendly interface which hides much of the complexity of distributed computing.

Components of a Spark cluster.
- The cluster itself consists of one or more nodes.
- Each node is a computer with CPU, RAM and physical storage.
- A cluster manager allocates resources and coordinates activity across the cluster.
- Every application running on the spark cluster has a driver program.
- Using the spark API, the driver communicates with the cluster manager, which in turn distributes work to the nodes.
- One each node, spark launches an executor process which persists for the duration of the application.
- Work is divided up into tasks, which are simply units of computation. 
- The executors run tasks in multiple threads across the cores in a node.
'''


'''
Connecting to Spark
Interacting with Spark
- The connection with Spark is established by the driver, which can be written in either Java, Scala, Python or R.
* Java - Low-level, compiled
* Scala, Python and R - High-level with interactive REPL (Read-Evaluate-Print loop).

Importing pyspark
- From Python import the pyspark module

import pyspark

- Check version of the pyspark module

pyspark.__version__ -> in

'2.4.1' -> out

Sub-modules
- In addition to pyspark there are
* Structured Data - pyspark.sql
* Streaming Data - pyspark.streaming
* Machine Learning - pyspark.mllib (deprecated) and pyspark.ml

Spark URL
- Remote Cluster using Spark URL - spark://<IP address | DNS name>:<port>
* A Spark URL must always include a port number, so this URL is not valid (default is 7077 but must always be specified).
example:
* spark://13.59.151.161:7077

- Local Cluster
examples:
* local - only 1 core;
* local[4] - 4 cores; or
* local[*] - all available cores

Creating a SparkSession
from pyspark.sql import SparkSession

- Create a local cluster using a SparkSession builder.

spark = SparkSession.builder.master('local[*]').appName('first_spark_application').getOrCreate()

* master() method is used to specify the location of the cluster
* appName() method is used optionally to assign a name to the application
* getOrCreate() method will either create a new session object or return an existing object. 

- Although it's possible for multiple SparkSessions to co-exist, it's good practice to stop the SparkSession when you're done.

# Close connection to Spark
spark.stop() 
'''

# Import the SparkSession class

# Create SparkSession object
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.feature import Bucketizer, OneHotEncoder
from pyspark.ml.feature import StopWordsRemover, HashingTF, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
spark = SparkSession.builder.master('local[*]').appName('test').getOrCreate()

# What version of Spark?
print(spark.version)

# Terminate the cluster
spark.stop()


'''
Loading Data
- Spark represents tabular data using the DataFrame class.

- Some selected methods:
* count()
* show()
* printSchema()
- Some selected attributes:
* dtypes

- Reading data from CSV
- The .csv() method reads a CSV file and returns a DataFrame.

cars = spark.read.csv('cars.csv', header=True)

- Optional arguments:
* header - is first row a header? (default: False)
* sep - field separator (default: a comma ',')
* schema - explicit column data types
* inferSchema - deduce column data types from data?
* nullValue - placeholder for missing data

Peek at the data
- The first five records from the DataFrame

cars.show(5)

Check column types
cars.printSchema() -> in

root
|-- mfr: string (nullable = true)
|-- mod: string (nullable = true)
|-- org: string (nullable = true)
|-- type: string (nullable = true)
|-- cyl: string (nullable = true)
|-- size: string (nullable = true)
|-- weight: string (nullable = true)
|-- len: string (nullable = true)
|-- rpm: string (nullable = true)
|-- cons: string (nullable = true) -> out

- The csv() method treats all columns as strings by default.

Inferring column types from data
cars = spark.read.csv('cars.csv', header=True, inferSchema=True)
cars.dtypes -> in

[   ('mfr', 'string'),
    ('mod, 'string'),
    ('org', 'string'),
    ('type', 'string'),
    ('cyl', 'string'),
    ('size', 'double'),
    ('weight', 'int'),
    ('len', 'int'),
    ('rpm', 'int'),
    ('cons', 'double')  ] -> out

- 'string' = strings, 'double' = floats, 'int' = integers
- 'cyl' was incorrectly classified as a string because it had missing values set to 'NA'

Dealing with missing data
- Handle missing data using the nullValue argument.

cars = spark.read.csv('cars.csv', header=True, inferSchema=True, nullValue='NA')

- The nullValue argument is case sensitive so always provide it in exact same form as it appears in the data file.

Specify column types
- If inferring column type is not successful, then you have the option of specifying the type of each column in an explicit schema.

schema = StructType([
    StructField('maker', StringType()),
    StructField('model', StringType()),
    StructField('origin', StringType()),
    StructField('type', StringType()),
    StructField('cyl', IntegerType()),
    StructField('size', DoubleType()),
    StructField('weight', IntegerType()),
    StructField('length', DoubleType()),
    StructField('rpm', IntegerType()),
    StructField('consumption', DoubleType())
])

cars = spark.read.csv('cars.csv', header=True, schema=schema, nullValue='NA')
'''

# Read data from CSV file
flights = spark.read.csv('flights.csv', sep=',',
                         header=True, inferSchema=True, nullValue='NA')

# Get number of records
print("The data contain %d records." % flights.count())

# View the first five records
flights.show(5)

# Check column data types
print(flights.dtypes)


# Specify column names and types
schema = StructType([
    StructField("id", IntegerType()),
    StructField("text", StringType()),
    StructField("label", IntegerType())
])

# Load data from a delimited file
sms = spark.read.csv('sms.csv', sep=';', header=False, schema=schema)

# Print schema of DataFrame
sms.printSchema()


''' Classification '''

'''
Data Preparation
Do you need all of those columns?
maker   model       origin      type    cyl     size    weight  length  rpm     consumption
Mazda   RX-7        non-USA     Sporty  null    1.3     2895    169.0   6500    9.41
Geo     Metro       non-USA     Small   3       1.0     1695    151.0   5700    4.7
Ford    Festiva     USA         Small   4       1.3     1845    141.0   5000    7.13

- Remove the maker and model fields

Dropping columns
# Either drop the columns you don't want . . . 
cars = cars.drop('maker', 'model')

# . . . or select the columns you want to retain.
cars = cars.select('origin', 'type', 'cyl', 'size', 'weight'. 'length', 'rpm', 'consumption') -> in

origin      type    cyl     size    weight  length  rpm     consumption
non-USA     Sporty  null    1.3     2895    169.0   6500    9.41
non-USA     Small   3       1.0     1695    151.0   5700    4.7
USA         Small   4       1.3     1845    141.0   5000    7.13 -> out

Filtering out missing data
# How many missing values?
cars.filter('cyl IS NULL').count() -> in

1 -> out

- Drop records with missing values in the cylinders column.
cars = cars.filter('cyl IS NOT NULL')

- Drop records with missing values in any column.
cars = cars.dropna()

Mutating columns
from pyspark.sql.functions import round

# Create a new 'mass' column
cars =  cars.withColumn('mass', round(cars.weight / 2.205, 0))

# Convert length to metres
cars =  cars.withColumn('length', round(cars.length * 0.0254, 3)) -> in

origin      type    cyl     size    weight  length  rpm     consumption     mass
non-USA     Small   3       1.0     1695    3.835   5700    4.7             769.0
USA         Small   4       1.3     1845    3.581   5000    7.13            837.0
non-USA     Small   3       1.3     1965    4.089   6000    5.47            891.0 -> out

Indexing categorical data
from pyspark.ml.feature import StringIndexer

Indexer = StringIndexer(inputCol='type', outputCol='type_idx')

# Assign index values to strings
indexer = indexer.fit(cars)

# Create column with index values
cars = indexer.transform(cars) -> in

type        type_idx
Midsize     0.0 <- most frequent value
Small       1.0
Compact     2.0
Sporty      3.0
Large       4.0
Van         5.0 <- least frequent value

- By default, the index values are assigned according to the descending relative frquency of each of the string values.
- Use stringOrderType() to choose different strategies for assigning index values

Indexing country of origin:
* USA -> 0
* non-USA -> 1

cars = StringIndexer(inputCol='origin', outputCol='label').fit(cars).transform(cars) -> in

origin      label
USA         0.0
non-USA     1.0 -> out

Assembling columns
- Use a vector assembler to transform the data.

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inpuCols=['cyl', 'size'], outputCol='features')
assembler.transform(cars) -> in

cyl     size    features
3       1.0     [3.0, 1.0]
4       1.3     [4.0, 1.3]
3       1.3     [3.0, 1.3] -> out
'''

# Remove the 'flight' column
flights_drop_column = flights.drop('flight')

# Number of records with missing 'delay' values
flights_drop_column.filter('delay IS NULL').count()

# Remove records with missing 'delay' values
flights_valid_delay = flights_drop_column.filter('delay IS NOT NULL')

# Remove records with missing values in any column and get the number of remaining rows
flights_none_missing = flights_valid_delay.dropna()
print(flights_none_missing.count())


# Import the required function

# Convert 'mile' to 'km' and drop 'mile' column (1 mile is equivalent to 1.60934 km)
flights_km = flights.withColumn('km', round(
    flights.mile * 1.60934, 0)).drop('mile')

# Create 'label' column indicating whether flight delayed (1) or not (0)
flights_km = flights_km.withColumn(
    'label', (flights_km.delay >= 15).cast('integer'))

# Check first five records
flights_km.show(5)


# Create an indexer
indexer = StringIndexer(inputCol='carrier', outputCol='carrier_idx')

# Indexer identifies categories in the data
indexer_model = indexer.fit(flights)

# Indexer creates a new column with numeric index values
flights_indexed = indexer_model.transform(flights)

# Repeat the process for the other categorical feature
flights_indexed = StringIndexer(inputCol='org', outputCol='org_idx').fit(
    flights_indexed).transform(flights_indexed)
flights_indexed.show(5)


# Import the necessary class

# Create an assembler object
assembler = VectorAssembler(inputCols=[
                            'mon', 'dom', 'dow', 'carrier_idx', 'org_idx', 'km', 'depart', 'duration'], outputCol='features')

# Consolidate predictor columns
flights_assembled = assembler.transform(flights)

# Check the resulting column
flights_assembled.select('features', 'delay').show(5, truncate=False)


'''
Decision Tree
- It is constructed using an algorithm called 'Recursive Partitioning'.
- The depth of each branch of the tree need not be the same.
* There are a variety of stopping criteria which can cause splitting to stop along a branch
- e.g, if the nuber of records in a node falls below a threshold or the purity of a node is above a threshold, then you might stop splitting. 

                Data
            /        \
        Green         Blue
    /         \
Blue            Green
                /   \
            Green   Blue

Split train / test
- Split data into training and testing sets

# Specify a seed for reproducibility
cars_train, cars_test = cars.randomSplit([0.8, 0.2], seed=23)

- Two DataFrames: cars_train and cars_test

[cars_train.count(), cars_test.count()] -> in

[79, 13] -> out

Building a Decision Tree model
from pyspark.ml.classification import DecisionTreeClassifier

- Create a Decision Tree classifier.

tree = DecisionTreeClassifier()

- Learn from the training data.

tree_model = tree.fit(cars_train)

Evaluating
- Make predictions on the testing data and compare to known values.

prediction = tree_model.transform(cars_test)

- The transform() method adds new columns to the DataFrame.
* The predicttion column gives the class assigned by the model.
* There's also a probability column which gives the probabilities assigned to each of the outcome classes.

Confusion matrix
- A confusion matrix is a table which describes performance of a model on testing data.

prediction.groupBy('label', 'prediction').count().show() -> in

label   prediction  count
1.0     1.0         8   <- True positive (TP)
0.0     1.0         2   <- False positive (FP)
1.0     0.0         3   <- False negative (FN)
0.0     0.0         6   <- True negative (TN) -> out 

Accuracy = (TN + TP) / (TN + TP + FN + FP) - proportion of correct predictions.
'''

# Split into training and testing sets in a 80:20 ratio
flights_train, flights_test = flights.randomSplit([0.8, 0.2], seed=43)

# Check that training set has around 80% of records
training_ratio = flights_train.count() / flights.count()
print(training_ratio)


# Import the Decision Tree Classifier class

# Create a classifier object and fit to the training data
tree = DecisionTreeClassifier()
tree_model = tree.fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
prediction = tree_model.transform(flights_test)
prediction.select('label', 'prediction', 'probability').show(5, False)


# Create a confusion matrix
prediction.groupBy('label', 'prediction').count().show()

# Calculate the elements of the confusion matrix
TN = prediction.filter('prediction = 0 AND label = prediction').count()
TP = prediction.filter('prediction = 1 AND label = prediction').count()
FN = prediction.filter('prediction = 0 AND label != prediction').count()
FP = prediction.filter('prediction = 1 AND label != prediction').count()

# Accuracy measures the proportion of correct predictions
accuracy = (TN + TP) / (TN + TP + FN + FP)
print(accuracy)


'''
Logistic Regression
- It uses a logistic function to model a binary target, where the target states are usually denoted by 1 and 0 or True and False.
- Since its values is a number between 0 and 1, it's often thought of as a probability.

- In order to translate the number into 1 or other of the target states, it's compared to a threshold, which is normally set at 0.5
* If the number is above the threshold, the the predicted state is 1
* If the number is below the threshold, then the predicted state is 0.
- The model derives coefficients for each of the numerical predictors.

Build a Logistic Regression model
from pyspark.ml.classification import LogisticRegression

- Create a Logistic Regression classifier
logistic = LogisticRegression()

- Learn from the training data
logistic = logistic.fit(cars_train)

Predictions
prediction= logistic.transform(cars_test)

Precision and recall
- How well does a model work on testing data?
Consult the confusion matrix

label   prediction  count
1.0     1.0         8   <- TP (true positive)
0.0     1.0         4   <- FP (false positive)
1.0     0.0         2   <- FN (false negative)
0.0     0.0         10  <- TN (true negative)

- Precision is the proportion of positive predictions which are correct

# Precision (positive)
TP / (TP + FP)

- Recall is the proportion of positive targets which are correctly predicted

# Recall (positive)
TP / (TP + FN)

Weighted metrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator()
evaluator.evaluate(prediction, {evaluator.metricName: 'weightedPrecision'})

- Other metrics:
* weightedRecall
* accuracy
* f1 metric : it is the harmonic mean of precision and recall, which is generally more robust than the accuracy.

ROC and AUC
- A threshold is used to decide whether the number returned by the Logistic Regression model translates into either the positive or the negative class.
* By default that threshold is set at 0.5 (half)

- The ROC curve plots the true positive rate versus the false positive rate as the threshold increases from 0 to 1.
* ROC = 'Receiver Operating Characteristic'
* TP versus FP
* threshold = 0
* threshold = 1

- The AUC summarizes the ROC curve in a single number
* It indicates how well a model performs across all values of the threshold.
* AUC = 'Area Under the Curve'
* ideally AUC = 1
'''

# Import the logistic regression class

# Create a classifier object and train on training data
logistic = LogisticRegression().fit(flights_train)

# Create predictions for the testing data and show confusion matrix
prediction = logistic.transform(flights_test)
prediction.groupBy('label', 'prediction').count().show()


# Calculate precision and recall
precision = TP / (TP + FP)
recall = TP / (TP + FN)
print('precision = {:.2f}\nrecall    = {:.2f}'.format(precision, recall))

# Find weighted precision
multi_evaluator = MulticlassClassificationEvaluator()
weighted_precision = multi_evaluator.evaluate(
    prediction, {multi_evaluator.metricName: "weightedPrecision"})

# Find AUC
binary_evaluator = BinaryClassificationEvaluator()
auc = binary_evaluator.evaluate(
    prediction, {binary_evaluator.metricName: "areaUnderROC"})


'''
Turning Text into Tables
A selection of children's books
books.show(truncate=False)

id  text
0   Forever, or a Long, Long Time -> 'Long' is only present in this title
1   Winnie-the-Pooh
2   Ten Little Fingers and Ten Little Toes
3   Five Get into Trouble -> 'Five' is present in other titles
4   Five Have a Wonderful Time
5   Five Get into a Fix
6   Five Have Plenty of Fun

Removing punctuation
from pyspark.sql.functions import regexp_replace

# Regular expression (REGEX) to match commas and hyphens
REGEX = '[.\\-]'

books = books.withColumn('text', regexp_replace(books.text, REGEX, ' '))

Before
id  text
0   Forever, or a Long, Long Time
1   Winnie-the-Pooh

After
id  text
0   Forever  or a Long  Long Time
1   Winnie the Pooh

Text to tokens
from pyspark.ml.feature import Tokenizer

books = Tokenizer(inputCol='text', outputCol='tokens').transform(books)

text                                        tokens
Forever or a Long, Long Time                [forever, or, a, long, long, time]
Winnie the Pooh                             [winnie, the, pooh]
Ten Little Fingers and Ten Little Toes      [ten, little, fingers, and, ten, little, toes]
Five Get into Trouble                       [five, get, into, trouble ]
Five Have a Wonderful Time                  [five, have, a, wonderful, time]

What are stop words?
from pyspark.ml.feature import StopWordsRemover

stopwords = StopWordsRemover()

# Take a look at the list of stop words
stopwords.getStopWords()

Removing stop words
# Specify the input and output column names
stopwords = stopwords.setInputCol('tokens').setOutputCol('words)

books = stopwords.transform(books)

tokens                                              words
[forever, or, a, long, long, time]                  [forever, long, long, time]
[winnie, the, pooh]                                 [winnie, pooh]
[ten, little, fingers, and, ten, little, toes]      [ten, little, fingers, ten, little, toes]
[five, get, into, trouble ]                         [five, get, trouble ]
[five, have, a, wonderful, time]                    [five, wonderful, time]

Feature hashing
- It converts words into numbers
- numFeatures: it is effectively the largest number that will be produced by the hashing trick.
* First list contains the hashed values
* Second list indicates how many times each of those values occurs

from pyspark.ml.feature import HashingTF

hasher = HashingTF(inputCol='words', outputCol='hash', numFeatures=32)
books = hasher.transform(books)

id  words                                           hash
0   [forever, long, long, time]                     (32, [8,13, 14], [2.0, 1.0, 1.0]) 
1   [winnie, pooh]                                  (32, [1, 31], [1.0, 1.0])
2   [ten, little, fingers, ten, little, toes]       (32, [1, 15, 25, 30], [2.0, 2.0, 1.0, 1.0])
3   [five, get, trouble]                            (32, [6, 7, 23], [1.0, 1.0, 1.0])
4   [five, wonderful, time]                         (32, [6, 13, 25], [1.0, 1.0, 1.0])

Dealing with common words
from pyspark.ml.feature import IDF

books = IDF(inputCol='hash', outputCol='features').fit(books).transform(books)

words                                           features
[forever, long, long, time]                     (32, [8,13, 14], [2.598, 1.299, 1.704]) 
[winnie, pooh]                                  (32, [1, 31], [1.299, 1.704])
[ten, little, fingers, ten, little, toes]       (32, [1, 15, 25, 30], [2.598, 3.409, 1.011, 1.704])
[five, get, trouble]                            (32, [6, 7, 23], [0.788, 1.704, 1.299])
[five, wonderful, time]                         (32, [6, 13, 25], [0.788, 1.299, 1.011])
'''

# Import the necessary functions

# Remove punctuation (REGEX provided) and numbers
wrangled = sms.withColumn('text', regexp_replace(
    sms.text, '[_():;,.!?\\-]', ' '))
wrangled = wrangled.withColumn(
    'text', regexp_replace(wrangled.text, '[0-9]', ' '))

# Merge multiple spaces
wrangled = wrangled.withColumn(
    'text', regexp_replace(wrangled.text, ' +', ' '))

# Split the text into words
wrangled = Tokenizer(inputCol='text', outputCol='words').transform(wrangled)

wrangled.show(4, truncate=False)


# Remove stop words.
wrangled = StopWordsRemover(inputCol='words', outputCol='terms').transform(sms)

# Apply the hashing trick
wrangled = HashingTF(inputCol='terms', outputCol='hash',
                     numFeatures=1024).transform(wrangled)

# Convert hashed symbols to TF-IDF
tf_idf = IDF(inputCol='hash', outputCol='features').fit(
    wrangled).transform(wrangled)

tf_idf.select('terms', 'features').show(4, truncate=False)


# Split the data into training and testing sets
sms_train, sms_test = sms.randomSplit([.8, .2], seed=13)

# Fit a Logistic Regression model to the training data
logistic = LogisticRegression(regParam=0.2).fit(sms_train)

# Make predictions on the testing data
prediction = logistic.transform(sms_test)

# Create a confusion matrix, comparing predictions to known labels
prediction.groupBy('label', 'prediction').count().show()


''' Regression '''

'''
One-Hot Encoding
The problem with indexed values
# Counts for 'type' category
type        count
Midsize     22
Small       21
Compact     16
Sporty      14
Large       11
Van         9

# Numerical indices for 'type' category
type        type_idx
Midsize     0.0
Small       1.0
Compact     2.0
Sporty      3.0
Large       4.0
Van         5.0

Dummy variables: binary encoding
type        Midsize     Small   Compact     Sporty  Large   Van
Midsize     1           0       0           0       0       0
Small       0           1       0           0       0       0
Compact ->  0           0       1           0       0       0 
Sporty      0           0       0           1       0       0
Large       0           0       0           0       1       0
Van         0           0       0           0       0       1

* Binary values indicates the presence (1) or absence (0) of the corresponding level.

Dummy variables: sparse representation
type        Midsize     Small   Compact     Sporty  Large   Van     Column  Value
Midsize     1           0       0           0       0       0       0       1
Small       0           1       0           0       0       0       1       1
Compact ->  0           0       1           0       0       0  ->   2       1
Sporty      0           0       0           1       0       0       3       1
Large       0           0       0           0       1       0       4       1
Van         0           0       0           0       0       1       5       1

* Sparse representation: store column index and value.

Dummy variables: redundant column
type        Midsize     Small   Compact     Sporty  Large   Column  Value
Midsize     1           0       0           0       0       0       1
Small       0           1       0           0       0       1       1
Compact ->  0           0       1           0       0  ->   2       1
Sporty      0           0       0           1       0       3       1
Large       0           0       0           0       1       4       1
Van         0           0       0           0       0   

* Levels are mutually exclusive, so drop one.

One-hot encoding
- One-Hot encoding is the process of creating dummy variables because only one of the columns created is ever active or 'hot'.

from pyspark.ml.feature import OneHotEncoder

onehot = OneHotEncoder(inputCols=['type_idx'], outputCols=['type_dummy'])

- Fit the encoder to the data
onehot = onehot.fit(cars)

# How many category levels?
onehot.categorySizes -> in

[6] -> out

One-hot encoding
cars = onehot.transform(cars)
cars.select('type', 'type_idx', 'type_dummy').distinct().sort('type_idx').show() -> in

type        type_idx    type_dummy
Midsize     0.0         (5, [0], [1.0])
Small       1.0         (5, [1], [1.0])
Compact     2.0         (5, [2], [1.0])
Sporty      3.0         (5, [3], [1.0])
Large       4.0         (5, [4], [1.0])
Van         5.0         (5, [], []) -> out

Dense versus sparse
from pyspark.mllib.linalg import DesneVector, SparseVector

- Store this vector: [1, 0, 0, 0, 0, 7, 0, 0]

DenseVector([1, 0, 0, 0, 0, 7, 0, 0]) -> in

DenseVector([1,0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0]) -> out

SparseVector(8, [0, 5], [1, 7]) -> in

SparseVector(8, {0: 1.0, 5: 7.0}) -> out
'''

# Import the one hot encoder class

# Create an instance of the one hot encoder
onehot = OneHotEncoder(inputCols=['org_idx'], outputCols=['org_dummy'])

# Apply the one hot encoder to the flights data
onehot = onehot.fit(flights)
flights_onehot = onehot.transform(flights)

# Check the results
flights_onehot.select(
    'org', 'org_idx', 'org_dummy').distinct().sort('org_idx').show()


'''
Regression
Residuals
- It is the difference between the observed value and the corresponding modeled value.

Loss function
- The best model is found by minimizing a loss function, which is an equation that describes how well the model fits the data.

Assemble predictors
- Predict 'consumption' using 'mass', 'cyl' and 'type_dummy'

Build regression model
from pyspark.ml.regression import LinearRegression

regression = LinearRegression(labelCol='consumption')

- Fit to cars_train (training data)
regression = regression.fit(cars_train)

- Predict on cars_test (testing data)
predictions = regression.transform(cars_test) -> in

Examine predictions
consumption     predictions
7.84            8.92699470743404
9.41            9.379295891451353
8.11            7.23487264538364
9.05            9.409860194333735
7.84            7.059190923328711
7.84            7.785909738591766
7.59            8.129959405158547
5.11            6.836843743852942
8.11            7.17173702652015 -> out

Calculate RMSE
- A smaller RMSE, however, always indicates better predictions.

from pyspark.ml.evaluation import RegressionEvaluator

# Find RMSE (Root Mean Squared Error)
RegressionEvaluator(labelCol='consumption').evaluate(predictions) -> in

0.708699086182001 -> out

- A RegressionEvaluator can also calculate the following metrics:
* mae (Mean Absolute Error)
* r^2 (R^2)
* mse (Mean Squared Error)

Examine intercept
regression.intercept -> in

4.9450616833727095 -> out

Examine Coefficients
regression.coefficients -> in

DenseVector([0.0027, 0.1897, -1.309, -1.7933, -1.3594, -1.2917, -1.9693]) 

mass    0.0027 
cyl     0.1897

Midsize     -1.309
Small       -1.7933
Compact     -1.3594
Sporty      -1.2917
Large       -1.9693 -> out
'''


# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data and take a look at the predictions
predictions = regression.transform(flights_test)
predictions.select('duration', 'prediction').show(5, False)

# Calculate the RMSE
RegressionEvaluator(labelCol='duration').evaluate(predictions)


# Intercept (average minutes on ground)
inter = regression.intercept
print(inter)

# Coefficients
coefs = regression.coefficients
print(coefs)

# Average minutes per km
minutes_per_km = regression.coefficients[0]
print(minutes_per_km)

# Average speed in km per hour
avg_speed = 60 / minutes_per_km
print(avg_speed)


# Create a regression object and train on training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Create predictions for the testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
RegressionEvaluator(labelCol='duration').evaluate(predictions)


# Average speed in km per hour
avg_speed_hour = 60 / regression.coefficients[0]
print(avg_speed_hour)

# Average minutes on ground at OGG
inter = regression.intercept
print(inter)

# Average minutes on ground at JFK
avg_ground_jfk = inter + regression.coefficients[3]
print(avg_ground_jfk)

# Average minutes on ground at LGA
avg_ground_lga = inter + regression.coefficients[4]
print(avg_ground_lga)


'''
Bucketing & Engineering
Bucketing
- It is the conversion of a continuous variable into discrete values.
- It can be done by assigning values to buckets or bins with well defined boundaries.
* The buckets might have uniform or variable width.

Bucketing heights
height  height_bin
1.42    short
1.45    short
1.47    short
1.50    short
1.52    average
1.57    average
1.60    average
1.75    average
1.85    tall
1.88    tall

RPM histogram
- Car RPM has 'n atural' breaks:
* RPM < 4500 - low
* RPM > 6000 - high
* Otherwise - medium

RPM buckets
from pyspark.ml.feature import Bucketizer

bucketizer = Bucketizer(splits=[3500, 4500, 6000, 6500], inputCol='rpm', outputCol='rpm_bin')

- Apply buckets to rpm column

bucketed = bucketizer.transform(cars)
bucketed.select('rpm', 'rpm_bin').show(5) -> in

rpm     rpm_bin
3800    0.0
4500    1.0
5750    1.0
5300    1.0
6200    2.0 -> out

bucketed.groupBy('rpm_bin').count().show() -> in

rpm_bin     count
0.0         8   <- low
1.0         67  <- medium
2.0         17  <- high -> out

One-hot encoded RPM buckets
- The RPM buckets are one-hot encoded to dummy variables

rpm_bin     rpm_dummy
0.0         (2, [0], [1.0]) <- low
1.0         (2, [1], [1.0]) <- medium
2.0         (2, [], [])     <- high

* The 'high' RPM bucket is the reference level and doesn't get a dummy variable

Model with bucketed RPM
regression.coefficients -> in

DenseVector([1.3814, 0.1433]) 

rpm_bin     rpm_dummy
0.0         (2, [0], [1.0]) <- low
1.0         (2, [1], [1.0]) <- medium
2.0         (2, [], []) <- high -> out

regression.intercept -> in

8.1835 -> out

- Consumption for 'low' RPM:
8.1835 + 1.3814 = 9.5649

- Consumption for 'medium' RPM:
8.1835 + 0.1433 = 8.3268

More feature engineering
- Operations on a single column:
* log()
* sqrt()
* pow()
- Operations on two columns:
* product
* ratio

Mass & Height to BMI
height  mass    bmi
1.52    77.1    33.2
1.60    58.1    22.7
1.57    122.0   49.4
1.75    95.3    31.0
1.80    99.8    30.7
1.65    90.7    33.3
1.60    70.3    27.5
1.78    81.6    25.8
1.65    77.1    28.3
1.78    128.0   40.5 

bmi = mass / height ^ 2

Engineering density
cars = cars.withColumn('density_line', cars.mass / cars.length) # Linear density
cars = cars.withColumn('density_quad', cars.mass / cars.length**2) # Area density
cars = cars.withColumn('density_cube', cars.mass / cars.length**3) # Volume density -> in

mass    length  density_line    density_quad    density_cube
1451.0  4.775   303.87434554    63.638606397    13.327456837
1129.0  4.623   244.21371403    52.825808790    11.426737787
1399.0  4.547   307.67539036    67.665579583    14.881367843 -> out
'''


# Create buckets at 3 hour intervals through the day
buckets = Bucketizer(splits=[0, 3, 6, 9, 12, 15, 18, 21, 24],
                     inputCol='depart', outputCol='depart_bucket')

# Bucket the departure times
bucketed = buckets.transform(flights)
bucketed.select('depart', 'depart_bucket').show(5)

# Create a one-hot encoder
onehot = OneHotEncoder(
    inputCols=['depart_bucket'], outputCols=['depart_dummy'])

# One-hot encode the bucketed departure times
flights_onehot = onehot.fit(bucketed).transform(bucketed)
flights_onehot.select('depart', 'depart_bucket', 'depart_dummy').show(5)


# Find the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Average minutes on ground at OGG for flights departing between 21:00 and 24:00
avg_eve_ogg = regression.intercept
print(avg_eve_ogg)

# Average minutes on ground at OGG for flights departing between 03:00 and 06:00
avg_night_ogg = regression.intercept + regression.coefficients[8]
print(avg_night_ogg)

# Average minutes on ground at JFK for flights departing between 03:00 and 06:00
avg_night_jfk = regression.intercept + \
    regression.coefficients[3] + regression.coefficients[9]
print(avg_night_jfk)


'''
Regularization
- A parsimonious model is one that has just the minimum required number of predictors. as simple as possible, yet still able to make robust predictions.

Loss function with regularization
- Linear regression aims to minimize the MSE.
* Add a regularization term which depends on coefficients.

Regularized term
- An extra regularization term is added to the loss function
- The regularization term can be either
* Lasso - Absolute value of the coefficients
* Ridge - Square of the coefficients

- It's also possible to have a blend of Lasso and Ridge regression.
- The strength of the regularization is determined by a parameter which is generally denoted by the Greek symbol Lambda.
* Lambda = 0 -> no regularization (standard regression)
* Lambda = infinity - complete regularization (all coefficients zero)

Cars again
assembler = VectorAssembler(inputCols=['mass', 'cyl', 'type_dummy', 'density_line', 'density_quad', density_cube']. outputCol='features')
cars =  assembler.transform(cars) -> in

Cars: Linear regression
- Fit a (standard) Linear Regression model to the training data.

regression = LinearRegression(labelCol='consumption').fit(cars_train) -> in

# RMSE on testing data
0.708699086182001 -> out

- Examine the coefficients:
regression.coefficients -> in

DenseVector([-0.012, 0.174, -0.897, -1.445, -0.985, -1.071, -1.335, 0.189, -0.780, 1.160]) -> out

* This means that every predictor is contributing to the model.

Cars: Ridge regression
# alpha = 0 | lambda = 0.1 -> Ridge
ridge = LinearRegression(labelCol='consumption', elasticNetParaam=0, regParam=0.1)
ridge.fit(cars_train) -> in

# RMSE on testing data
0.724535609745491

# Ridge coefficients
DenseVector([0.001, 0.137, -0.395, -0.822, -0.450, -0.582, -0.806, 0.008, 0.029, 0.001])
# Linear coefficients
DenseVector([-0.012, 0.174, -0.897, -1.445, -0.985, -1.071, -1.335, 0.189, -0.780, 1.160]) -> out

Cars: Lasso regression
# alpha = 1 | lambda = 0.1 -> Lasso
lasso = LinearRegression(labelCol='consumption', elasticNetParaam=1, regParam=0.1)
lasso.fit(cars_train) -> in

# RMSE on testing data
0.771988667026998

# Lasso coefficients
DenseVector([ 0.0, 0.0, 0.0, -0.056, 0.0, 0.0, 0.0, 0.026, 0.0, 0.0])
# Ridge coefficients
DenseVector([0.001, 0.137, -0.395, -0.822, -0.450, -0.582, -0.806, 0.008, 0.029, 0.001])
# Linear coefficients
DenseVector([-0.012, 0.174, -0.897, -1.445, -0.985, -1.071, -1.335, 0.189, -0.780, 1.160]) -> out
'''


# Fit linear regression model to training data
regression = LinearRegression(labelCol='duration').fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)


# Fit Lasso model (λ = 1, α = 1) to training data
regression = LinearRegression(
    labelCol='duration', regParam=1, elasticNetParam=1).fit(flights_train)

# Calculate the RMSE on testing data
predictions = regression.transform(flights_test)
rmse = RegressionEvaluator(labelCol='duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)


''' Ensembles & Pipelines '''

'''
Pipeline
- It consists of a series of operations, that are grouped together and applied as a single unit.

Leakage?
- .fit() - Only for training data
- .transform() - For testing and training data
- leakage occurs whenever a fit() method is applied to the testing data.

Cars model: steps
indexer = StringIndexer(inputCol='type', outputCol='type_idx')
onehot = OneHotEncoder(inputCol=['type_idx'], outputCol=['type_dummy'])
assemble = VectorAssembler(inputCols=['mass', 'cyl', 'type_dummy'], outputCol='features')
regression = LinearRegression(labelCol='consumption')

Training data
indexer = indexer.fit(cars_train)
cars_train = indexer.transform(cars_train)

Testing data
cars_test = indexer.transform(car_test)

Training data
onehot = onehot.fit(cars_train)
cars_train = onehot.transform(cars_train)

Testing data
cars_test = onehot.transform(car_test)

Training data
cars_train = assemble.transform(cars_train)

Testing data
cars_test = assemble.transform(car_test)

Training data
# Fit model to training data
regression = regression.fit(cars_train)

Testing data
# Make predictions on testing data
predictions = regression.transform(car_test)

Cars model: Pipeline
- Combine steps into a pipeline

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[indexer, onehot, assemble, regression])

Training data
pipeline = pipeline.fit(cars_train)

# Testing data
predictions = pipeline.transform(cars_test)

Cars model: Stages
- Access individual stages using the .stages attribute.

# The LinearRegression object (fourth stage -> index 3)
pipeline.stages[3]

print(pipeline.stages[3].intercept) -> in

4.19433571782916 -> out

print(pipeline.stages[3].coefficients) -> in

DenseVector([0.0028, 0.2705, -1.1813, -1.3696, -1.1751, -1.1553, -1.8894]) -> out
'''

# Convert categorical strings to index values
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# One-hot encode index values
onehot = OneHotEncoder(inputCols=['org_idx', 'dow'], outputCols=[
                       'org_dummy', 'dow_dummy'])

# Assemble predictors into a single column
assembler = VectorAssembler(
    inputCols=['km', 'org_dummy', 'dow_dummy'], outputCol='features')

# A linear regression object
regression = LinearRegression(labelCol='duration')


# Import class for creating a pipeline

# Construct a pipeline
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])

# Train the pipeline on the training data
pipeline = pipeline.fit(flights_train)

# Make predictions on the testing data
predictions = pipeline.transform(flights_test)


# Break text into tokens at non-word characters
tokenizer = Tokenizer(inputCol='text', outputCol='words')

# Remove stop words
remover = StopWordsRemover(
    inputCol=tokenizer.getOutputCol(), outputCol='terms')

# Apply the hashing trick and transform to TF-IDF
hasher = HashingTF(inputCol=remover.getOutputCol(), outputCol="hash")
idf = IDF(inputCol=hasher.getOutputCol(), outputCol="features")

# Create a logistic regression object and add everything to a pipeline
logistic = LogisticRegression()
pipeline = Pipeline(stages=[tokenizer, remover, hasher, idf, logistic])


'''
Cross-Validation
Cars revisited
cars.select('mass', 'cyl', consumption').show(5) -> in

mass    cyl     consumption
1451.0  6       9.05
1129.0  4       6.53
1399.0  4       7.84
1147.0  4       7.84
1111.0  4       9.05    -> out

Estimator and evaluator
- An object to build the model. This can be a pipeline
regression = LinearRegression(labelCol='consumption')

- An object to evaluate model performance
evaluator = RegressionEvaluator(labelCol='consumption')

Grid and cross-validator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

- A grid of parameter values ( empty for the moment).
params = ParamGridBuilder().build()

- The cross-validation object
cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator, numFolds=10, seed=13)

Cross-validators need training too
- Apply cross-validation to the training data
cv = cv.fit(cars_train)

- What's the average RMSE across the folds?
cv.avgMetrics -> in

[0.800663722151572] -> out

Cross-validators act like models
- Make predictions on the original testing data
evaluator.evaluate(cv.transform(cars_test)) -> in

# RMSE on testing data
0.745974203928479

- Much smaller than the cross-validated RMSE

# RMSE from cross-validation
[0.800663722151572] -> out

* A simple train-test split would have given an overly optimistic view on model performance
'''

# Create an empty parameter grid
params = ParamGridBuilder().build()

# Create objects for building and evaluating a regression model
regression = LinearRegression(labelCol='duration')
evaluator = RegressionEvaluator(labelCol='duration')

# Create a cross validator
cv = CrossValidator(estimator=regression,
                    estimatorParamMaps=params, evaluator=evaluator, numFolds=5)

# Train and test model on multiple folds of the training data
cv = cv.fit(flights_train)

# NOTE: Since cross-valdiation builds multiple models, the fit() method can take a little while to complete.


# Create an indexer for the org field
indexer = StringIndexer(inputCol='org', outputCol='org_idx')

# Create an one-hot encoder for the indexed org field
onehot = OneHotEncoder(inputCols=['org_idx'], outputCols=['org_dummy'])

# Assemble the km and one-hot encoded fields
assembler = VectorAssembler(
    inputCols=['km', 'org_dummy'], outputCol='features')

# Create a pipeline and cross-validator.
pipeline = Pipeline(stages=[indexer, onehot, assembler, regression])
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params, evaluator=evaluator)


'''
Grid Search
Cars revisited
cars.select('mass', 'cyl', consumption').show(5) -> in

mass    cyl     consumption
1451.0  6       9.05
1129.0  4       6.53
1399.0  4       7.84
1147.0  4       7.84
1111.0  4       9.05    -> out

Fuel consumption with intercept
- Linear regression with an intercept. Fit to training data.

regression = LinearRegression(labelCol='consumption', fitIntercept=True)
regression = regression.fit(cars_train)

- Calculate the RMSE on the testing data
evaluator.evaluate(regression.transform(cars_test)) -> in

# RMSE for model with an intercept
0.745974203928479 -> out

Fuel consumption without intercept
- Linear regression without an intercept. Fit to training data.

regression = LinearRegression(labelCol='consumption', fitIntercept=False)
regression = regression.fit(cars_train)

- Calculate the RMSE on the testing data
evaluator.evaluate(regression.transform(cars_test)) -> in

# RMSE for model without an intercept (second model)
0.852819012439 -> out

# RMSE for model with an intercept (first model)
0.745974203928479 -> out

Parameter grid
from pyspark.ml.tuning import ParamGridBuilder

# Create a parameter grid builder
params = ParamGridBuilder()

# Add grid points
params = params.addGrid(regression.fitIntercept, [True, False])

# Construct the grid
params = params.build()

# How many models?
print('Number of models to be tested: ', len(params)) -> in

Number of models to be tested: 2

Grid search with cross-validation
- Create a cross-validator and fit to the traing data

cv = CrossValidator(estimator=regression, estimatorParamMaps=params, evaluator=evaluator)
cv = cv.setNumFolds(10).setSeed(13).fit(cars_train)

- What's the cross-validated RMSE for each model?
cv.avgMetrics -> in

[0.800663722151, 0.907977823182] -> out

The best model & parameters
# Access the best model
cv.bestModel

- Or just use the cross-validator object
predictions = cv.transform(cars_test)

- Retrieve the best parameter
cv.bestModel.explainParam('fitIntercept') -> in

'fitIntercept: whether to fit an intercept term (default: True, current: True)' -> out

# A more complicated grid
params = ParamGridBuilder().addGrid(regression.fitIntercept, [True, False]).addGrid(regression.regParam, [0.001, 0.01, 0.1, 1, 10]).addGrid(regression.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]).build()

- How many models now?
print('Number of models to be tested: ', len(params)) -> in

Number of models to be tested: 50 -> out
'''

# Create parameter grid
params = ParamGridBuilder()

# Add grids for two parameters
params = params.addGrid(regression.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(
    regression.elasticNetParam, [0.0, 0.5, 1.0])

# Build the parameter grid
params = params.build()
print('Number of models to be tested: ', len(params))

# Create cross-validator
cv = CrossValidator(estimator=pipeline,
                    estimatorParamMaps=params, evaluator=evaluator, numFolds=5)


# Get the best model from cross validation
best_model = cv.bestModel

# Look at the stages in the best model
print(best_model.stages)

# Get the parameters for the LinearRegression object in the best model
best_model.stages[3].extractParamMap()

# Generate predictions on testing data using the best model then calculate RMSE
predictions = best_model.transform(flights_test)
print("RMSE =", evaluator.evaluate(predictions))


# Create parameter grid
params = ParamGridBuilder()

# Add grid for hashing trick parameters
params = params.addGrid(hasher.numFeatures, [1024, 4096, 16384]).addGrid(
    hasher.binary, [True, False])

# Add grid for logistic regression parameters
params = params.addGrid(logistic.regParam, [0.01, 0.1, 1.0, 10.0]).addGrid(
    logistic.elasticNetParam, [0.0, 0.5, 1.0])

# Build parameter grid
params = params.build()


'''
Ensemble
- It is a collection of models
- It combines the results from multiple models to produce better predictions than any one of those models acting alone.
- Its concept is that of 'Wisdom of the Crowd'
* Collective opinion of a group better than that of a single expert.

Ensemble diversity
- Diversity and independence are import because the best collective decisions are the product of disagreement and contest, not consensus or compromise.
* James Surowiecki, The Wisdom of Crowds.

Random Forest
- Random Forest - an ensemble of Decision Trees
- Creating model diversity:
* Each tree trained on random subset of data
* random subset of features used for splitting at each node
- No 2 trees in the forest should be the same.

Create a forest of trees
- Returning to cars data: manufactured in USA (0.0) or not (1.0)
- Create Random Forest classifier.

from pyspark.ml.classification import RandomForestClassifier

forest = RandomForestClassifier(numTrees=5)

- Fit to the training data

forest = forest.fit(cars_train)

Seeing the trees
- How to access trees within forest?
forest.trees -> in

[   DecisionTreeClassificationModel (uid=dtc_aa66702a4ce9) of depth 5 with 17 nodes,
    DecisionTreeClassificationModel (uid=dtc_99f7efedafe9) of depth 5 with 31 nodes,
    DecisionTreeClassificationModel (uid=dtc_9306e4a5fa1d) of depth 5 with 21 nodes,
    DecisionTreeClassificationModel (uid=dtc_d643bd48a8dd) of depth 5 with 23 nodes,
    DecisionTreeClassificationModel (uid=dtc_a2d5abd67969) of depth 5 with 27 nodes ] -> out

* These can each be used to make individual predictions.

Predictions from individual trees
- What predictions are generated by each tree?

tree 0  tree 1  tree 2  tree 3  tree 4  label
0.0     0.0     0.0     0.0     0.0     0.0     <- perfect agreement
1.0     1.0     0.0     1.0     0.0     0.0
0.0     0.0     0.0     1.0     1.0     1.0
0.0     0.0     0.0     1.0     0.0     0.0
0.0     1.0     1.0     1.0     0.0     1.0
1.0     1.0     0.0     1.0     1.0     1.0
1.0     1.0     1.0     1.0     1.0     1.0     <- perfect agreement

Consensus predictions
- Use the .transform() method to generate consensus predictions

label   probability                                 prediction
0.0     [0.8, 0.2]                                  0.0
0.0     [0.4, 0.6]                                  1.0
1.0     [0.5333333333333333, 0.4666666666666666]    0.0
0.0     [0.7177777777777778, 0.2822222222222226]    0.0
1.0     [0.39396825396825397, 0.606031746031746]    1.0
1.0     [0.17660818713450294, 0.823391812865497]    1.0
1.0     [0.053968253968253964, 0.946031746031746]   1.0

Feature importances
- The model uses these features: cyl, size, mass, length, rpm and comsumption
- Which of these is most or least important?

forest.featureImportances -> in

SparseVector(6, {0: 0.0205, 1: 0.2701, 2: 0.108, 3: 0.1895, 4: 0.2939, 5: 0.1181}) -> out

- Looks like:
* rpm is most important
* cyl is least important

Gradient-Boosted Trees
- Iterative boosting algorithm:
* Build a Decision Tree and add to ensemble
* Predict label for each training instance using ensemble
* Compare predictions with known labels
* Emphasize training instances with incorrect predictions
* Return to 1
- Model improves on each iteration because each new tree focuses on correcting the shortcomings of the preceding trees.

Boosting trees
- Create a Gradient-Boosted Tree classifier

from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(maxIter=10)

- Fit to the training data
gbt = gbt.fit(cars_train)

Comparing trees
- Let's compare the 3 typres of tree models on testing data

# AUC for DecisionTree
0.5875

# AUC for Random Forest
0.65

# AUC for Gradient-Boosted Tree
0.65

* Both of the ensemble methods perform better than a plain Decision Tree.
'''

# Import the classes required

# Create model objects and train on training data
tree = DecisionTreeClassifier().fit(flights_train)
gbt = GBTClassifier().fit(flights_train)

# Compare AUC on testing data
evaluator = BinaryClassificationEvaluator()
print(evaluator.evaluate(tree.transform(flights_test)))
print(evaluator.evaluate(gbt.transform(flights_test)))

# Find the number of trees and the relative importance of features
print(gbt.getNumTrees)
print(gbt.featureImportances)


# Create a random forest classifier
forest = RandomForestClassifier()

# Create a parameter grid
params = ParamGridBuilder().addGrid(forest.featureSubsetStrategy, [
    'all', 'onethird', 'sqrt', 'log2']).addGrid(forest.maxDepth, [2, 5, 10]).build()

# Create a binary classification evaluator
evaluator = BinaryClassificationEvaluator()

# Create a cross-validator
cv = CrossValidator(estimator=forest, estimatorParamMaps=params,
                    evaluator=evaluator, numFolds=5)


# Average AUC for each parameter combination in grid
print(cv.avgMetrics)

# Average AUC for the best model
print(max(cv.avgMetrics))

# What's the optimal parameter value for maxDepth?
print(cv.bestModel.explainParam('maxDepth'))
# What's the optimal parameter value for featureSubsetStrategy?
print(cv.bestModel.explainParam('featureSubsetStrategy'))

# AUC for best model on testing data
print(evaluator.evaluate(cv.transform(flights_test)))
