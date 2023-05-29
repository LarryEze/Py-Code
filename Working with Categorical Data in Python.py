''' Introduction to Categorical Data '''

'''
Course introduction
What does it mean to be 'categorical'?
Categorical
- Finite number of groups (or categories)
- These categories are usually fixed or known (eye color, hair color, etc.)
- Known as qualitative data

Numerical 
- Known as quantitative data
- Expressed using a numerical value
- Is usually a measurement (height, weight, IQ, etc)

Ordinal vs nominal variables
Ordinal
- Categorical variables that have a natural order

Strongly Disagree   Disagree    Neutral     Agree   Strongly Agree
        1               2           3         4           5

Nominal
- Categorical variables that cannot be placed into a natural order

Blue  Green  Red  Yellow  Purple

Our first dataset
adult.info() -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 32561 entries, 0 to 32560
Data columns (total 15 columns):
#  Column           Non-Null Count      Dtype
0  Age              32561 non-null       int64
1  Workclass        32561 non-null      object
2  fnlgwt           32561 non-null       int64
3  Education        32561 non-null      object
4  Education Num    32561 non-null       int64
5  Marital Status   32561 non-null      object
...  ->  out

Using describe
adult['Marital Status'].describe() -> in

count       32561
unique      7
top         Married-civ-spouse
freq        14976
Name: Marital Status, dtype: object -> out

Using value counts
adult['Marital Status'].value_counts() -> in

Married-civ-spouse        14976
Never-married             10683
Divorced                   4443
Separated                  1025
Widowed                     993
Married-spouse-absent       418
Married-AF-spouse            23
Name: Marital Status, dtype: int64 -> out

Using value counts
adult['Marital Status'].value_counts(normalize=True) -> in

Married-civ-spouse          0.459937
Never-married               0.328092
Divorced                    0.136452
Separated                   0.031479
Widowed                     0.030497
Married-spouse-absent       0.012837
Married-AF-spouse           0.000706
Name: Marital Status, dtype: float64 -> out
'''

# Explore the Above/Below 50k variable
print(adult["Above/Below 50k"].describe())

# Print a frequency table of "Above/Below 50k"
print(adult["Above/Below 50k"].value_counts())

# Print relative frequency values
print(adult["Above/Below 50k"].value_counts(normalize=True))


'''
Categorical data in pandas
dtypes: object
adult = pd.read_csv('data/adult.csv')
adult.dtypes    -> in

Age  int64
Workclass           object
fnlgwt               int64
Education           object
Education Num        int64
Marital Status      object
Occupation          object
Relationship        object
...     -> out

dtypes: categorical
Default dtype:

adult['Marital Status'].dtype   -> in

dtype('O')  -> out

Set as categorical:

adult['Marital Status'] = adult['Marital Status'].astype('category')
adult['Marital Status'].dtype   -> in

CategoricalDtype(categories=[' Divorced', ' Married-AF-spouse',
' Married-civ-spouse', ' Married-spouse-absent', ' Never-married',
' Separaated', ' Widowed'], ordered=False)  -> out

Creating a categorical Series
my_data = ['A', 'A', 'C', 'B', 'C', 'A']

my_series1 = pd.Series(my_data, dtype='category')
print(my_series1) -> in

0   A
1   A
2   C
...
dtype: category
Categories (3, object): [A, B, C] -> out

my_series2 = pd.Categorical(my_data, categories=['C', 'B', 'A'], ordered=True)
my_series2 -> in

[A, A, C, B, C, A]
Categories (3, object): [C < B < A] -> out

Why do we use categorical: memory
Memory saver:

adult = pd.read_csv('data/adult.csv')
adult['Marital Status'].nbytes  -> in

260488  -> out

adult['Marital Status'] = adult['Marital Status'].astype('category')
adult['Marital Status'].nbytes  -> in

32617  -> out

Specify dtypes when reading data
1. Create a dictionary:
adult_dtypes = {'Marital Status': 'category'}

2. Set the 'dtype' parameter:
adult = pd.read_csv('data/adult.csv', dtype=adult_dtypes)

3. Check the dtype:
adult['Marital Status'].dtype   -> in

CategoricalDtype(categories=[' Divorced', ' Married-AF-spouse',
' Married-civ-spouse', ' Married-spouse-absent', ' Never-married',
' Separaated', ' Widowed'], ordered=False)  -> out
'''

# Create a Series, default dtype
series1 = pd.Series(list_of_occupations)

# Print out the data type and number of bytes for series1
print("series1 data type:", series1.dtype)
print("series1 number of bytes:", series1.nbytes)

# Create a Series, "category" dtype
series2 = pd.Series(list_of_occupations, dtype="category")

# Print out the data type and number of bytes for series2
print("series2 data type:", series2.dtype)
print("series2 number of bytes:", series2.nbytes)


# Create a categorical Series and specify the categories (let pandas know the order matters!)
medals = pd.Categorical(medals_won, categories=[
                        "Bronze", "Silver", "Gold"], ordered=True)
print(medals)


# Check the dtypes
print(adult.dtypes)

# Create a dictionary with column names as keys and "category" as values
adult_dtypes = {
    "Workclass": "category",
    "Education": "category",
    "Relationship": "category",
    "Above/Below 50k": "category"
}

# Read in the CSV using the dtypes parameter
adult2 = pd.read_csv(
    "adult.csv",
    dtype=adult_dtypes
)
print(adult2.dtypes)


'''
Grouping data by category in pandas
The basics of .groupby(): splitting data
adult = pd.read_csv('data/adult.csv')
adult1 = adult[adult['Above/Below 50k'] == ' <=50K']
adult2 = adult[adult['Above/Below 50k'] == ' >50K']

is replaced by
groupby_object = adult.groupby(by=['Above/Below 50K'])

The basics of .groupby(): apply a function
groupby_object.mean() -> in

                    Age         fnlgwt          Education Num   Capital Gain    ...
Above/Below 50K
    <=50K           36.783738   190340.86517     9.595065        148.752468     ...
    >50K            44.249841   188005.00000    11.611657       4006.142456     ... -> out

One liner:
adult.groupby(by=['Above/Below 50K']).mean()

Specifying columns
Option 1: only runs .sum() on two columns.

adult.groupby(by=['Above/Below 50k'])['Age', 'Education Num'].sum() -> in

                    Age         Education Num
Above/Below 50K
    <=50K           909294      237190
    >50K            346963       91047      -> out

Option 2: runs .sum() on all numeric columns and then subsets.

adult.groupby(by=['Above/Below 50k']).sum()[['Age', 'Education Num']]

Option 1 is preferred - especially when using large datasets.

Groupby multiple columns
adult.groupby(by=['Above/Below 50k', 'Marital Status']).size() -> in

Above/ Below 50k    Marital Staus
    <=50K           Divorced                 3980
                    Married-AF-spouse          13
                    Married-civ-spouse       8284
                    Married-spouse-absent     384
                    Never-married           10192
                    Separated                 959
                    Widowed                   908
    >50K            Divorced                  463
                    Married-AF-spouse          10   <--- Only 10 records
                    Married-civ-spouse       6692
    ...     -> out
'''

# Group the adult dataset by "Sex" and "Above/Below 50k"
gb = adult.groupby(by=["Sex", "Above/Below 50k"])

# Print out how many rows are in each created group
print(gb.size())

# Print out the mean of each group for all columns
print(gb.mean())


# Create a list of user-selected variables
user_list = ["Education", "Above/Below 50k"]

# Create a GroupBy object using this list
gb = adult.groupby(by=user_list)

# Find the mean for the variable "Hours/Week" for each group - Be efficient!
print(gb["Hours/Week"].mean())


''' Categorical pandas Series '''

'''
Setting category variables
New dataset: adoptable dogs
dog.info() -> in

RangeIndex: 2937 entries, 0 to 2936, Data columns (total 19 columns):
#   Column          Non-Null Count      Dtype
0   ID              2937 non-null       int64
    ...
8   color           2937 non-null      object
9   coat            2937 non-null      object
    ...
17  get_along_cats   431 non-null      object
18  keep_in         1916 non-null      object
dtypes: float64(1), int64(1), object(17)
memory usage: 436.1+ KB  -> out

A dog's coat
dogs['coat'] = dogs['coat'].astype('category')
dogs['coat'].value_counts(dropna=False) -> in

short          1972
medium          565
wirehaired      220
long            180
Name: coat, dtype: int64 -> out

The .cat accessor object
Series.cat.method_name

Common parameters:
* new_categories: a list of categories
* inplace: Boolean - whether or not the update should overwrite the Series
* ordered: Boolean - whether or not the categorical is treated as an ordered categorical

Setting Series categories
Set categories:
dogs['coat'] = dogs['coat'].cat.set_categories(new_categories=['short', 'medium', 'long'])

Check value counts:
dogs['coat'].value_counts(dropna=False) -> in

short      1972
medium      565
NaN         220
long        180  -> out

Setting order
dogs['coat'] = dogs['coat'].cat.set_categories(new_categories=['short', 'medium', 'long'], ordered=True)

dogs['coat'].head(3) -> in

0   short
1   short
2   short
Name: coat, dtype: category
Categories (3, object): ['short' < 'medium' < 'long'] -> out

Missing categories
dogs['likes_people'].value_counts(dropna=False) -> in

yes     1991
NaN      938
no         8  -> out

A NaN could mean:
1. truly unknown (we didn't check)
2. Not sure (dog likes 'some' people)

Adding categories
Add categories
dogs['likes_people'] = dogs['likes_people'].astype('category')
dogs['likes_people'] = dogs['likes_people'].cat.add_categories(new_categories=['did not check', 'could not tell'])

Check categories:
dogs['likes_people'].cat.categories  -> in

Index(['no', 'yes', 'did not check', 'could not tell'], dtype='object') -> out

New categories
dogs['likes_people'].value_counts(dropna=False) -> in

yes              1991
NaN               938
no                  8  
could not tell      0
did not check       0  -> out

Removing categories
dogs['coat'] = dogs['coat'].astype('category')
dogs['coat'] = dogs['coat'].cat.remove_categories(removals=['wirehaired'])

Check the caategories:
dogs['coat'].cat.categories -> in

Index(['long', 'medium', 'short'], dtype='object') -> out

Methods recap
- Setting: cat.set_categories()
* Can be used to set the order of categories
* All values not specified in this method are dropped
- Adding: cat.add_categories()
* Does not change the value of any data in the DataFrame
* Categories not listed in this method are left alone
- Removing: cat.remove_categories()
* Values matching categories listed are set to NaN
'''

# Check frequency counts while also printing the NaN count
print(dogs["keep_in"].value_counts(dropna=False))

# Switch to a categorical variable
dogs["keep_in"] = dogs["keep_in"].astype("category")

# Add new categories
new_categories = ["Unknown History", "Open Yard (Countryside)"]
dogs["keep_in"] = dogs["keep_in"].cat.add_categories(new_categories)

# Check frequency counts one more time
print(dogs["keep_in"].value_counts(dropna=False))


# Set "maybe" to be "no"
dogs.loc[dogs["likes_children"] == "maybe", "likes_children"] = "no"

# Print out categories
print(dogs["likes_children"].cat.categories)

# Print the frequency table
print(dogs["likes_children"].value_counts())

# Remove the `"maybe" category
dogs["likes_children"] = dogs["likes_children"].cat.remove_categories([
                                                                      "maybe"])
print(dogs["likes_children"].value_counts())

# Print the categories one more time
print(dogs["likes_children"].cat.categories)


'''
Updating categories
The breed variable
Breed value counts:
dogs['breed'] = dogs['breed'].astype('category')
dogs['breed'].value_counts() -> in

Unknown Mix                1524
German Shepherd Dog Mix     190
Dachshund Mix               147
Labrador Retriever Mix       83
Staffordshire Terrier Mix    62
... -> out

Renaming categories
The rename_categories method:
Series.cat.rename_categories(new_categories=dict)
Make a dictionary:
my_changes = {'Unknown Mix': 'Unknown'}
Rename the category:
dogs['breed'] = dogs['breed'].cat.rename_categories(my_changes)

The updated breed variable
Breed value counts:
dogs['breed'].value_counts() -> in

Unknown                    1524
German Shepherd Dog Mix     190
Dachshund Mix               147
Labrador Retriever Mix       83
Staffordshire Terrier Mix    62
... -> out

Renamingcategories with a function
Update multiple categories:
dogs['sex'] = dogs['sex'].cat. rename_categories(lambda c: c.title())
dogs['sex'].cat.categories -> in

Index(['Female', 'Male'], dtype='object') -> out

Common replacement issues
- Must use new category names
# Does not work! 'Unknown' already exists
use_new_categories = {'Unknown Mix': 'Unknown'}
- Cannot collapse two categories into one
# Does not work! New names must be unique
cannot_repeat_categories = {'Unknown Mix': 'Unknown', 'Mixed Breed': 'Unknown'}

Collapsing categories setup
A dogs color:
dogs['color'] = dogs['color'].astype('category')
print(dogs['color'].cat.categories) -> in

Index(['apricot', 'black', 'black and brown', 'black and tan', 
    'black and white', 'brown', 'brown and white', 'dotted', 'golden',
    'gray', 'gray and black', 'gray and white', 'red', 'red and white',
    'sable', 'saddle back', 'spotty', 'striped', 'tricolor', 'white',
    'wild boar', 'yellow', 'yellow-brown'], 
    dtype='object')
... -> out

Create a dictionary and use .replace():
update_colors = {'black and brown': 'black', 'black and tan': 'black', 'black and white': 'black'}

dogs['main_color'] = dogs['color'].replace(update_colors)

Check the Series data type:
dogs['main_color'].dtype -> in

dtype('O') -> out

Convert back to categorical
dogs['main_color'] = dogs['main_color'].astype('category')
dogs['main_color'].cat.categories  -> in

Index(['apricot', 'black', 'brown', 'brown and white', 'dotted', 'golden',
    'gray', 'gray and black', 'gray and white', 'red', 'red and white',
    'sable', 'saddle back', 'spotty', 'striped', 'tricolor', 'white',
    'wild boar', 'yellow', 'yellow-brown'], 
    dtype='object')  -> out
'''

# Create the my_changes dictionary
my_changes = {'Maybe?': 'Maybe'}

# Rename the categories listed in the my_changes dictionary
dogs["likes_children"] = dogs["likes_children"].cat.rename_categories(
    my_changes)

# Use a lambda function to convert all categories to uppercase using upper()
dogs["likes_children"] = dogs["likes_children"].cat.rename_categories(
    lambda c: c.upper())

# Print the list of categories
print(dogs["likes_children"].cat.categories)


# Create the update_coats dictionary
update_coats = {'wirehaired': 'medium', 'medium-long': 'medium'}

# Create a new column, coat_collapsed
dogs["coat_collapsed"] = dogs['coat'].replace(update_coats)

# Convert the column to categorical
dogs["coat_collapsed"] = dogs["coat_collapsed"].astype('category')

# Print the frequency table
print(dogs["coat_collapsed"].value_counts())


'''
Reordering categories
Why would you reorder?
1. Creating a ordinal variable
2. To set the order that variables are displayed in analysis
3. Memory savings

Reordering example
dogs['coat'] = dogs['coat'].cat.reorder_categories(new_categories = ['short', 'medium', 'wirehaired', 'long'], ordered=True)

Using inplace:
dogs['coat'].cat.reorder_categories(new_categories = ['short', 'medium', 'wirehaired', 'long'], ordered=True, inplace=True)

Grouping when ordered = True
dogs['coat'] = dogs['coat'].cat.reorder_categories(new_categories = ['short', 'medium', 'wirehaired', 'long'], ordered=True)

dogs.groupby(by=['coat'])['age'].mean() -> in

coat
short           8.364746
medium          9.027982
wirehaired      8.424136
long            9.552056 -> out

Grouping when ordered = False
dogs['coat'] = dogs['coat'].cat.reorder_categories(new_categories = ['short', 'medium', 'long', 'wirehaired'], ordered=False)

dogs.groupby(by=['coat'])['age'].mean() -> in

coat
short           8.364746
medium          9.027982
long            9.552056 
wirehaired      8.424136  -> out
'''

# Print out the current categories of the size variable
print(dogs["size"].cat.categories)

# Reorder the categories, specifying the Series is ordinal, and overwriting the original series
dogs["size"].cat.reorder_categories(
    new_categories=["small", "medium", "large"],
    ordered=True,
    inplace=True
)

# How many Male/Female dogs are available of each size?
print(dogs.groupby(by=['size'])["sex"].value_counts())

# Do larger dogs need more room to roam?
print(dogs.groupby(by=['size'])['keep_in'].value_counts())


'''
Cleaning and accessing data
Possible issues with categorical data
1. Inconsistent values: 'Ham', 'ham', ' Ham'
2. Misspelled values: 'Ham', 'Hma'
3. Wrong dtype: 
df['Our Column'].dtype - > in

dtype('O') -> out

Identifying issues
Use either:
- Series.cat.categories
- Series.value_counts()

dogs['get_along_cats'].value_counts() -> in

No     2503
yes     275
no      156
Noo       2
 NO       1 -> out

Fixing issues: whitespace
Removing whitespace: .strip()
dogs['get_along_cats'] = dogs['get_along_cats'].str.strip()

Check the frequency counts:
dogs['get_along_cats'].value_counts() -> in

No     2503
yes     275
no      156
Noo       2
NO        1 # < --- no more whitespace      -> out

Fixing issues: capitalization
Capitalization: .title(), .upper(), .lower()
dogs['get_along_cats'] = dogs['get_along_cats'].str.title()

Check the frequency counts:
dogs['get_along_cats'].value_counts() -> in

No     2660
Yes     275
Noo       2  -> out

Fixing issues: misspelled words
Fixing a typo with .replace()
replace_map = {'Noo': 'No'}
dogs['get_along_cats'].replace(replace_map, inplace=True)

Check the frequency counts:
dogs['get_along_cats'].value_counts() -> in

No     2662
Yes     275  -> out

Checking the data type
Checking the dtype
dogs['get_along_cats'].dtype -> in

dtype('O') -> out

Converting back to a category
dogs['get_along_cats'] = dogs['get_along_cats'].astype('category')

Using the str accessor object
Searching for a string
dogs['breed'].str.contains('Shepherd', regex=False) -> in

0       False
1       False
2       False
    ...
2935    False
2936     True  -> out

Accessing data with loc
Access Series values based on category
dogs.loc[dogs['get_along_cats'] == 'Yes', 'size']

Series value counts:
dogs.loc[dogs['get_along_cats'] == 'Yes', 'size'].value_counts(sort=False) -> in

small        69
medium      169
large        37  -> out
'''

# Fix the misspelled word
replace_map = {"Malez": "male"}

# Update the sex column using the created map
dogs["sex"] = dogs["sex"].replace(replace_map)

# Strip away leading whitespace
dogs["sex"] = dogs["sex"].str.strip()

# Make all responses lowercase
dogs["sex"] = dogs["sex"].str.lower()

# Convert to a categorical Series
dogs["sex"] = dogs["sex"].astype('category')

print(dogs["sex"].value_counts())


# Print the category of the coat for ID 23807
print(dogs.loc[dogs.index == 23807, 'coat'])

# Find the count of male and female dogs who have a "long" coat
print(dogs.loc[dogs['coat'] == 'long', 'sex'].value_counts())

# Print the mean age of dogs with a breed of "English Cocker Spaniel"
print(dogs.loc[dogs['breed'] == 'English Cocker Spaniel', 'age'].mean())

# Count the number of dogs that have "English" in their breed name
print(dogs[dogs["breed"].str.contains('English', regex=False)].shape[0])


''' Visualizing Categorical Data '''

'''
Introduction to categorical plots using Seaborn
Our third dataset
- Name: Las Vegas TripAdvisor Reviews - reviews
- Rows: 504
- Columns: 20

Las Vegas reviews
reviews.info() -> in

RangeIndex: 504 entries, 0 to 503
Data columns (total 20 columns):
#   Column          Non-Null Count  Dtype
0   User country    504 non-null    object
    ...
6   Traveler type   504 non-null    object
7   Pool            504 non-null    object
8   Gym             504 non-null    object
9   Tennis court    504 non-null    object
    ...

dtypes: inte64(7), object(13)
memory usage: 78.9+ KB -> out

seaborn
Categorical plots:
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(...)
plt.show()

The catplot function
Paraameters:
- x: name of variable in data
- y: name of variable in data
- data: a DataFrame
- kind: type of plot to create - one of: 'strip', 'swarm', 'box', 'violin', 'boxen', 'point', 'bar', or 'count'

Box plot
It shows information on the quartiles of numerical data.

Review score
reviews['Score'].value_counts() -> in

5   227
4   164
3    72
2    30
1    11 -> out

# Setting font size and plot background
sns.set(font_scale=1.4)
sns.set_style('whitegrid')

sns.catplot(x='Pool', y='Score', data=reviews, kind='box')
plt.show()
'''

# Set the font size to 1.25
sns.set(font_scale=1.25)

# Set the background to "darkgrid"
sns.set_style('darkgrid')

# Create a boxplot
sns.catplot(x='Traveler type', y='Helpful votes', data=reviews, kind='box')

plt.show()


'''
Seaborn bar plots
Traditional bar chart
# Code provided for clarity
reviews['Traveler type'].value_counts().plot.bar()

The syntax
sns.set(font_scale=1.3)
sns.set_style('darkgrid')
sns.catplot(x='Traveler type', y='Score', data=reviews, kind='bar')
plt.show()

Ordering your categories
reviews['Traveler type'] = reviews['Traveler type'].astype('category')
reviews['Traveler type'].cat.categories -> in

Index(['Business', 'Couples', 'Families', 'Friends', 'Solo'], dtype='object') -> out

* Note: catplot() has an order parameter, but not all visulaization methods have this.

The hue parameter
- hue:
* name of a variable in data
* used to split the data by a second category
* also used to color the graphic

sns.set(font_scale=1.2)
sns.set_style('darkgrid')
sns.catplot(x='Traveler type', y='Score', data=reviews, kind='bar', hue='Tennis court')  # <--- new parameter
plt.show()
'''

# Print the frequency counts of "Period of stay"
print(reviews['Period of stay'].value_counts())

sns.set(font_scale=1.4)
sns.set_style("whitegrid")

# Create a bar plot of "Helpful votes" by "Period of stay"
sns.catplot(x='Period of stay', y='Helpful votes', data=reviews, kind='bar')
plt.show()


# Set style
sns.set(font_scale=.9)
sns.set_style("whitegrid")

# Print the frequency counts for "User continent"
print(reviews["User continent"].value_counts())

# Convert "User continent" to a categorical variable
reviews["User continent"] = reviews["User continent"].astype("category")

# Reorder "User continent" using continent_categories and rerun the graphic
continent_categories = list(reviews["User continent"].value_counts().index)
reviews["User continent"] = reviews["User continent"].cat.reorder_categories(
    new_categories=continent_categories)
sns.catplot(x="User continent", y="Score", data=reviews, kind="bar")
plt.show()


# Add a second category to split the data on: "Free internet"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x='Casino', y="Score", data=reviews,
            kind="bar", hue='Free internet')
plt.show()

# Switch the x and hue categories
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x='Free internet', y="Score",
            data=reviews, kind="bar", hue='Casino')
plt.show()

# Update x to be "User continent"
sns.set(font_scale=2)
sns.set_style("darkgrid")
sns.catplot(x='User continent', y="Score",
            data=reviews, kind="bar", hue="Casino")
plt.show()

# Lower the font size so that all text fits on the screen.
sns.set(font_scale=1.0)
sns.set_style("darkgrid")
sns.catplot(x="User continent", y="Score",
            data=reviews, kind="bar", hue="Casino")
plt.show()


'''
Point and count plots
Point plot example
sns.catplot(x='Pool', y='Score', data=reviews, kind='point')  # <--- updated

- The point plot shows the mean of the reviewer score just as a bar plot does.
- The point plot may help users focus on the different values across the categories by adding a connecting line across the points, while the y-axis is changed to better focus on the points.
- the bar plot, however, may have a more familiar look and does provide color differences even if only one categorical variable is used.

point plot with hue
sns.catplot(x='Spa', y='Score', data=reviews, kind='point', hue='Tennis court', dodge=True)  # <--- New Parameter!

The dodge = True offsets the lines so that they don't overlap and makes it easier for users to see where the mean and confidence intervals fall.

Using the join parameter 
sns.catplot(x='Score', y='Review weekday', data=reviews, kind='point', join=False)  # <--- New !

The join = False makes the lines no longer connected

One last catplot type
sns.catplot(x='Tennis court', data=reviews, kind='count', hue='Spa')

The catplot method used the count plot to display frequencies.
The count plot simply counts the number of occurrences of the categorical variables specified in the x or y and hue parameters.
'''

# Create a point plot with catplot using "Hotel stars" and "Nr. reviews"
sns.catplot(
    # Split the data across Hotel stars and summarize Nr. reviews
    x='Hotel stars',
    y='Nr. reviews',
    data=reviews,
    # Specify a point plot
    kind='point',
    hue="Pool",
    # Make sure the lines and points don't overlap
    dodge=True
)
plt.show()


sns.set(font_scale=1.4)
sns.set_style("darkgrid")

# Create a catplot that will count the frequency of "Score" across "Traveler type"
sns.catplot(
    x='Score', data=reviews, kind='count', hue='Traveler type'
)
plt.show()


'''
Additional catplot() options
Using different arguments
sns.catplot(x='Traveler type', kind='count', col='User continent', col_wrap=3, palette= sns.color_palette('Set1'), data=reviews)

* x: 'Traveler type'
* kind: 'count'
* col: 'User continent'
* col_wrap: 3
* palette: sns.color_palette('Set1')
- Common colors: 'Set'. 'Set2', 'Tab10', 'Paired'

Updating plots
- Setup: save your graphic as an object: ax
- Plot title: ax.fig.suptitle('My title')
- Axis labels: ax.set_axis_labels('x-axis-label', 'y-axis-label')
- Title height: plt.subplots_adjust(top=.9)

ax = sns.catplot(x='Traveler type', col='User continent', col_wrap=3, kind='count', palette= sns.color_palette('Set1'), data=reviews)
ax.fig.suptitle('Hotel Score by Traveler Type & User Continent')
ax.set_axis_labels('Traveler Type', 'Number of Reviews')
plt.subplots_adjust(top=.9)
plt.show()
'''

# Create a catplot for each "Period of stay" broken down by "Review weekday"
ax = sns.catplot(
    # Make sure Review weekday is along the x-axis
    x='Review weekday',
    # Specify Period of stay as the column to create individual graphics for
    col='Period of stay',
    # Specify that a count plot should be created
    kind='count',
    # Wrap the plots after every 2nd graphic.
    col_wrap=2,
    data=reviews
)
plt.show()


# Adjust the color
ax = sns.catplot(
    x="Free internet", y="Score",
    hue="Traveler type", kind="bar",
    data=reviews,
    palette=sns.color_palette("Set2")
)

# Add a title
ax.fig.suptitle("Hotel Score by Traveler Type and Free Internet Access")
# Update the axis labels
ax.set_axis_labels("Free Internet", "Average Review Rating")

# Adjust the starting height of the graphic
plt.subplots_adjust(top=.93)
plt.show()


''' Pitfalls and Encoding '''

'''
Categorical pitfalls
Used cars: the final dataset
import pandas as pd

used_cars = pd.read_csv('used_cars.csv')
used_cars.info() -> in

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 38531 entries, 0 to 38530
Data columns (total 30 columns):
#  Column               Non-Null Count      Dtype
0  manufacturer_name    38531 non-null     object
1  model_name           38531 non-null     object
2  transmission         38531 non-null     object  -> out
...  -> out

Huge memory savings 
used_cars['manufacturer_name'].describe() -> in

count            38531
unique              55
top         Volkswagen
freq              4243
Name: manufacturer_name, dtype: object -> out

print('As object: ', used_cars['manufacturer_name'].nbytes)
print('As category: ', used_cars['manufacturer_name'].astype('category').nbytes)  -> in

As object:      308248
As category:    38971  -> out

Little memory savings
used_cars['odometer_value'].describe() -> in

count      38531
unique      6063
top       300000
freq        1794
Name: odometer_value, dtype: int64 -> out

print(f'As float: {used_cars['odometer_value'].nbytes}')
print(f'As category: {used_cars['odometer_value'].astype('category').nbytes}')  -> in

As float:       308248
As category:    125566  -> out

Using categories can be frustrating
- Using the .str accessor object to manipulate data converts the Series to an object.
- The .apply() method outputs a new Series as an object
- The common methods of adding, removing, replacing, or setting categories do not all handle missing categories the same way.
- NumPy functions generally do not work with categorical Series.

Check and convert
Check
used_cars['color'] = used_cars['color'],astype(;category')
used_cars['color'] = used_cars['color'].str.upper()
print(used_cars['color'].dtype) -> in

object  -> out

Convert
used_cars['color'] = used_cars['color'].astype('category')
print(used_cars['color'].dtype) -> in

category  -> out 

Look for missing values
Set categories
used_cars['color'] = used_cars['color'].astype('category')
used_cars['color'].cat.set_categories(['black', 'silver', 'blue'], inplace=True)
used_cars['color'].value_counts(dropna=False) -> in

NaN        18172
black       7705
silver      6852
blue        5802
Name: color, dtype: int64  -> out

Using NumPy arrays
used_cars['number_of_photos'] = used_cars['number_of_photos'].astype('category')
used_cars['number_of_photos'].sum()  # <--- Gives an Error -> in

TypeError: Categorical cannot perform the operation sum  -> out

used_cars['number_of_photos'].astype(int).sum()

Note:
# .str converts te column to an array
used_cars['color'].str.contains('red')  -> in

0   False
1   False
...  -> out
'''

# Print the frequency table of body_type and include NaN values
print(used_cars["body_type"].value_counts(dropna=False))

# Update NaN values
used_cars.loc[used_cars["body_type"].isna(), "body_type"] = "other"

# Convert body_type to title case
used_cars["body_type"] = used_cars["body_type"].str.title()

# Check the dtype
print(used_cars["body_type"].dtype)


# Print the frequency table of Sale Rating
print(used_cars["Sale Rating"].value_counts())

# Find the average score
average_score = used_cars["Sale Rating"].astype(int).mean()

# Print the average
print(average_score)


'''
Label encoding
What is label encoding?
The basics:
- Codes each category as an integer from 0 through n -1, where n is the number of categories
- A -1 code is reserved for any missing values
- Can save on memory
- Often used in surveys
The drawback:
- It is not the best encoding method for machine learning (see next lesson)

Creating codes
Convert to categorical and sort by manufacturer name
used_cars['manufacturer_name'] = used_cars['manufacturer_name'].astype('category')

Use .cat.codes
used_cars['manufacturer_code'] = used_cars['manufacturer_name'].cat.codes

Check output
print(used_cars[['manufacturer_name', 'manufacturer_code']]) -> in

        manufacturer_name   manufacturer_code
0       Subaru              45
1       Subaru              45
2       Subaru              45
...     ...                 ...
38526   Chrysler            8
38527   Chrysler            8  -> out

Code books / data dictionaries
Survey Year(s):         2013
Topic                   Admin
Description             New construction in last 4 years
Table Name              NEWHOUSE
Type                    Character
Edit Flag Variable      NA
Imputation
Strategy
Response Codes          1: Yes
                        2: No

Creating a code book
codes = used_cars['manufacturer_name'].cat.codes
categories = used_cars['manufacturer_name']
name_map = dict(zip(codes, categories))
print(name_map) -> in

{45: 'Subaru',
 24: 'LADA',
 12: 'Dodge',
 ...
} -> out

Using a code book
creating the codes:
used_cars['manufacturer_code'] = used_cars['manufacturer_name'].cat.codes

Reverting to previous values:
.map() is similar to .replace(), and it will replace the series values based on the keys of the name-map and their corresponding values.

used_cars['manufacturer_code'].map(name_map) -> in

0   Acura
1   Acura
2   Acura
...     -> out

Boolean coding
Find all body types that have 'van' in them:

used_cars['body_type'].str.contains('van', regex=False)

Create a boolean coding:
used_cars['van_code'] = np.where(
used_cars['body_type'].str.contains('van', regex=False), 1, 0)
used_cars['van_code'].value_counts() -> in

0   34115
1    4416
Name: van_code, dtype: int64 -> out
'''

# Convert to categorical and print the frequency table
used_cars["color"] = used_cars["color"].astype("category")
print(used_cars["color"].value_counts())

# Create a label encoding
used_cars["color_code"] = used_cars["color"].cat.codes

# Create codes and categories objects
codes = used_cars["color"].cat.codes
categories = used_cars["color"]
color_map = dict(zip(codes, categories))

# Print the map
print(color_map)


# Update the color column using the color_map
used_cars_updated["color"] = used_cars_updated["color"].map(color_map)
# Update the engine fuel column using the fuel_map
used_cars_updated["engine_fuel"] = used_cars_updated["engine_fuel"].map(
    fuel_map)
# Update the transmission column using the transmission_map
used_cars_updated["transmission"] = used_cars_updated["transmission"].map(
    transmission_map)

# Print the info statement
print(used_cars_updated.info())


# Print the "manufacturer_name" frequency table.
print(used_cars["manufacturer_name"].value_counts())

# Create a Boolean column for the most common manufacturer name
used_cars["is_volkswagen"] = np.where(
    used_cars["manufacturer_name"].str.contains("Volkswagen", regex=False), 1, 0)

# Check the final frequency table
print(used_cars["is_volkswagen"].value_counts())


'''
One-hot encoding
One-hot encoding is the process of creating dummy variables.

One-hot encodingwith pandas
pd.get_dummies()
- data: a pandas DataFrame
- columns: a list-like object of column names
- prefix: a string to add to the beginning of each category

One-hot encoding on a DataFrame
used_cars[['odometer_value', 'color']].head() -> in

    odometer_value      color
0   190000              silver
1   290000              blue
2   402000              red
3    10000              blue
4   280000              black
... -> out

used_cars_onehot = pd.get_dummies(used_cars[['odometer_value', 'color']])
used_cars_onehot.head() -> in

    odometer_value  color_black     color_brown     color_green     ...
0   190000          0               0               0               ...
1   290000          0               0               0               ...
2   402000          0               0               0               ...
3    10000          0               0               0               ...
4   280000          1               0               0               ... -> out

print(used_cars_onehot.shape) -> in

(38531, 13) -> out

Specifying columns to use
used_cars_onehot = pd.get_dummies(used_cars, columns=['color'], prefix='')
used_cars_onehot.head() -> in

    manufacturer_name   ...     _black  _blue   _brown  
0   Subaru              ...     0       0       0
1   Subaru              ...     0       1       0
2   Subaru              ...     0       0       0
3   Subaru              ...     0       1       0
4   Subaru              ...     1       0       0       -> out

print(used_cars_onehot.shape) -> in

(38531, 41) -> out

A few quick notes
- Might create too many features
used_cars_onehot = pd.get_dummies(used_cars)
print(used_cars_onehot.shape) -> in

(38531, 1240) -> out

- NaN values do not get their own column
'''

# Create one-hot encoding for just two columns
used_cars_simple = pd.get_dummies(
    used_cars,
    # Specify the columns from the instructions
    columns=["manufacturer_name", "transmission"],
    # Set the prefix
    prefix="dummy"
)

# Print the shape of the new dataset
print(used_cars_simple.shape)
