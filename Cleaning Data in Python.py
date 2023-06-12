# import necessary packages
import pandas as pd
import numpy as np
import datetime as dt
import missingno as msno
import matplotlib.pyplot as plt
from thefuzz import process
import recordlinkage


''' Common data problems '''

'''
Data type constraints
- text data:    string
- Integers:     Integers
- Decimals:     Floats
- Binary:       Boolean
- Dates:        datetime
- Categories:   Category

Strings to integers
e.g
# Import CSV file and output header
sales = pd.read_csv('sales.csv')
sales.head(2) -> in

    SalesOrderID   Revenue Quantity
0          43659    23153$       12
1          43660     1457$        2 -> out

# Get data types of columns
sales.dtypes -> in

SalesOrderID    int64
Revenue         object
Quantity        int64
dtype: object   -> out

# Get DataFrame information
sales.info() -> in

<class 'pandas.core.frame.DateFrame'>
RangeIndex: 31465 entries, 0 to 31464
Data columns (total 3 columns):
SalesOrderID    31465   non-null    int64
Revenue         31465   non-null    object
Quantity        31465   non-null    int64
dtypes: int64(2), object(1)
memory usage: 737.5+ KB -> out

# Print sum of all Revenue column
sales['Revenue'].sum() -> in

'23153$1457$36865$32474$472$27510$16158$5694$6876$40487$807$... -> out

# Remove $ from Revenue column
sales['Revenue'] = sales['Revenue'].str.strip('$')
sales['Revenue'] = sales['Revenue'].astype('int')

# Verify that Revenue is now an integer
assert sales['Revenue'].dtype == 'int'

The assert statement
# This will pass
assert 1 + 1 == 2

# This will not pass
assert 1 + 1 == 3 -> in

AssertionError                                  Traceback (most recent call last)
        assert 1 + 1 == 3
AssertionError: -> out

Numeric or categorical?
e.g
... marriage_status ...
...               3 ...
...               1 ...
...               2 ...

0 = Never married   1 = Married     2 = Separated   3 = Divorced

df['marriage_status'].describe() -> in

    marriage_status
...
mean    1.4
std     0.20
min     0.00
50%     1.8 ... -> out

# Convert to categorical
df['marriage_status'] = df['marriage_status'].astype('category')
df.describe -> in

        marriage_status
count               241
unique                4
top                   1
freq                120 -> out
'''

ride_sharing = pd.read_csv('Cleaning Data in Python/ride_sharing_new.csv')

# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')

# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'

# Print new summary statistics
print(ride_sharing['user_type_cat'].describe())


# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration
print(ride_sharing[['duration', 'duration_trim', 'duration_time']])
print(ride_sharing['duration_time'].mean())


'''
Data range constraints
How to deal with out of range data?
- Dropping data ( rule of thumb: Only drop data when a small proportion of the dataset is affected by out of range values)
- Setting custom minimums and maximums 
- Treat as missing and impute
- Setting custom value depending on business assumptions


Motivation
e.g
movies.head()
        movie_name  avg_rating
0    The Godfather           5
1         Frozen 2           3
2            Shrek           4
...

import matplotlib.pyplot as plt
plt.hist(movies['avg_rating'])
plt.title('Average rating of movies (1 - 5)')

import pandas as pd
# Output Movies with rating > 5
movies[ movies['avg_rating'] > 5 ] -> in

            movie_name avg_rating
23    A Beautiful Mind          6
65     La Vita e Bella          6
77              Amelie          6 -> out

# Drop values using filtering
movies = movies[ movies['avg_rating'] <= 5 ]

# Drop values using .drop()
movies.drop( movies[movies['avg_rating'] > 5 ].index, inplace = True )

# Covert avg_rating > 5 to 5
movies.loc[ movies['avg_rating'] > 5, 'avg_rating' ] = 5

# Assert results
assert movies['avg_rating'].max() <= 5

# OR

# Import data time
import datetime as dt
today_date = dt.date.today()
user_signups[ user_signups['subscription_date'] > dt.date.today() ] -> in

    subscription_date   user_name  ...           Country
0          01/05/2023       Marah  ...             Nauru
1          09/08/2022      Joshua  ...           Austria
2          04/01/2022       Heidi  ...            Guinea
3          11/10/2022        Rina  ...      Turkmenistan
4          11/07/2022   Christine  ...  Marshall Islands
5          07/07/2022      Ayanna  ...             Gabon -> out

import pandas as pd
# Output data types
user_signups.dtypes -> in

subscription_date   object
user_name           object
Country             object
dtype:              object -> out

# Convert to date
user_signups['subscription_data'] = pd.to_datetime( user_signups['subscription_date'] ).dt.date

today_date = dt.date.today()

Drop the data
# Drop values using filtering
user_signups = user_signups[ user_signups['subscription_date'] < today_date ]

# Drop values using .drop()
user_signups.drop( user_signups[ user_signups['subscription_date'] > today_date ].index, inplace = True)

Hardcode dates with upper limit
# Drop values using filtering
user_signups.loc[  user_signups['subscription_date'] > today_date, 'subscription_date' ] = today_date

# Assert is true
assert user_signups.subscription_date.max().date() <= today_date 
'''

# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())


# Convert ride_date to date
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date']).dt.date

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())


'''
Uniqueness constraints
How to find duplicate values?
# Print the header
height_weight.head() -> in

    first_name last_name                        address  height weight
0         Lane     Reese               534-1559 Nam St.     181     64
1         Ivor    Pierce              102-3364 Non Road     168     66
2        Roary    Gibson    P.O. Box 344, 7785 Nisi Ave     191     99
3      Shannon    Little   691-2550 Consectetur Street     185     65
4        Abdul       Fry                 4565 Risus St.     169     65 -> out

# Get duplicates across all columns
duplicates = height_weight.duplicated()
print(duplicates) -> in

1   False
... ...
22  True
23  False
... ... -> out

# Get duplicate rows
duplicates =  height_weight.duplicated()
height_weight[ duplicates ] -> in

    first_name last_name                               address height weight
100       Mary     Colon                           4674 Ut Rd.    179     75
101       Ivor    Pierce                     102-3364 Non Road    168     66
102       Cole    Palmer                       8366 At, Street    178     91
103    Desirae   Shannon  P.O. Box 643, 5251 Consectetur, Rd.    196     83 -> out

The .duplicated() method
subset: List of column names to check for duplication
keep: Whether to keep first ('first), last ('last'), or all (False) duplicate values

# Column names to check for duplication
column_names = [ 'first_name', 'last_name', 'address' ]
duplicates = height_weight.duplicated( subset =  column_names, keep = False )

The .drop_duplicates() method
subset: List of column names to check for duplication
keep: Whether to keep first ('first), last ('last'), or all (False) duplicate values
inplace: Drop duplicated rows directly inside DataFrame without creating new object ( True )

# Drop duplicates
height_weight.drop_duplicates( inplace =  True )

# OR

The .groupby() and .agg() methods
# Group by column names and produce statistical summaries
column_names = [ 'first_name', 'last_name', 'address' ]
summaries = { 'column_name': 'max', 'column_name1': 'mean' }
height_weight =  height_weight.groupby( by = column_names ).agg( summaries ).reset_index()

# Make sure aggregation is done
duplicates = height_weight.duplicated( subset =  column_names, keep = False )
height_weight[ duplicates ].sort_values( by = 'first_name' )
'''

# Find duplicates
duplicates = ride_sharing.duplicated(subset='ride_id', keep=False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id', 'duration', 'user_birth_year']])


# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset='ride_id', keep=False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0


''' Text and categorical data problems '''

'''
Membership constraints
Predefined finite set of categories
Type of data                         Example values      Numeric representation
Marriage status                  unmarried, married                        0, 1
Household income Category        0-20k, 20-40k, ...                   0, 1, ...
Loan Status                 default, payed, no_loan                     0, 1, 2

Why could we have these problems?
- Data entry issues with free text vs dropdown fields
- Data parsing errors
- Other types of errors

How do we treat these problems?
- Dropping data ( i.e rows with incorrect categories )
- Remapping Categories ( i.e to correct ones )
- Inferring Categories

e.g
# Read study data and print it
study_data = pd.read_csv('study.csv')
study_data -> in

        name    birthday blood_type
1       Beth  2019-10-20         B-
2   Ignatius  2020-07-08         A-
3       Paul  2019-08-12         O+
4      Helen  2019-03-17         O-
5   Jennifer  2019-12-17         Z+
6    Kennedy  2020-04-27         A+
7      Keith  2019-04-19        AB+ -> out

# Correct possible blood types
categories -> in

    blood_type
1           O-
2           O+
3           A-
4           A+
5           B+
6           B-
7          AB+
8          AB-  -> out

Finding inconsistent categories
inconsistent_categories = set(study_date['blood_type']).difference(categories['blood_type'])
print(inconsistent_categories) -> in

{'Z+'} -> out

# Get and print rows with inconsistent categories
inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
inconsistent_data = study_data[inconsistent_rows] -> in

        name    birthday blood_type
5   Jennifer  2019-12-17         Z+ -> out

# Drop inconsistent categories and get consistent data only
consistent_data = study_data[~inconsistent rows] -> in

        name    birthday blood_type
1       Beth  2019-10-20         B-
2   Ignatius  2020-07-08         A-
3       Paul  2019-08-12         O+
4      Helen  2019-03-17         O-
6    Kennedy  2020-04-27         A+
7      Keith  2019-04-19        AB+ -> out

A note on joins
- Anti Joins: What is in A and not in B
- Inner Joins: What is in both A and B
'''

airlines = pd.read_csv('Cleaning Data in Python/airlines_final.csv')

categories = pd.DataFrame({'cleanliness': ['Clean', 'Average', 'Somewhat clean', 'Somewhat dirty', 'Dirty'], 'safety': ['Neutral', 'Very safe', 'Somewhat safe',
                          'Very unsafe',  'Somewhat unsafe'], 'satisfaction': ['Very satisfied', 'Neutral', 'Somewhat satisfied', 'Somewhat unsatisfied', 'Very unsatisfied']})

# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")

# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])


'''
Categorical variables
What type of errors could we have?
- Value inconsistency
*   Inconsistent fields: 'married', 'Married', 'UNMARRIED', 'not married' ...
*   Trailing white spaces: 'married ', ' married ' ...
- Collapsing too many categories to few
*   Creating new groups: 0-20k, 20-40k categories ... from continuous household income data
*   Mapping groups to new ones: Mapping household income categories to 2 'rich', 'poor'
- Making sure data is of type category

Value consistency
Capitalization: 'married', 'Married', 'UNMARRIED', 'unmarried' ...
e.g
# Get marriage status column ( for Series data )
marriage_status = demographics['marriage_status']
marriage_status.value_counts() -> in

unmarried   352
married     268
MARRIED     204
UNMARRIED   176
dtype: int64 -> out

# Get value counts on DataFrame
marriage_status.groupby('marriage_status').count() -> in

                    household_income gender
marriage_status
MARRIED                          204    204
UNMARRIED                        176    176
married                          268    268
unmarried                        352    352  -> out

# Capitalize
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
marriage_status['marriage_status'].value_counts() -> in

UNMARRIED   528
MARRIED     472 -> out

# Lowercase
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
marriage_status['marriage_status'].value_counts() -> in

unmarried   528
married     472 -> out

Trailing spaces: 'married ', 'married', 'unmarried', ' unmarried' ...
e.g
# Get marriage status column ( for Series data )
marriage_status = demographics['marriage_status']
marriage_status.value_counts() -> in

unmarried   352
unmarried   268
married     204
married     176
dtype: int64 -> out

# Strip all spaces
demographics = demographics['marriage_status'].str.strip()
demographics['marriage_status'].value_counts() -> in

unmarried   528
married     472 -> out

Collapsing data into categories
Create categories out of data: income_group column from income column
e.g
# Using qcut()
import pandas as pd
group_names = [ '0-200k', '200k-500k', '500k+' ]
demographics['income_group'] = pd.qcut( demographics['household_income'], q = 3, labels = group_names )
# Print income_group column
demographics[ ['income_group', 'household_income'] ] -> in

    category    household_income
0  200k-500k    189243
1      500k+    778533
...     -> out

# OR

# Using cut() - create category ranges and names
import pandas as pd
ranges = [ 0, 200000, 500000, np.inf ]
group_names = [ '0-200k', '200k-500k', '500k+' ]

# Create income group column
demographics['income_group'] = pd.cut( demographics['household_income'], bins = ranges, labels = group_names )
demographics[ ['income_group', 'household_income'] ] -> in

    category    household_income
0     0-200k    189243
1      500k+    778533 -> out

Map categories to fewer ones: reducing categories in categorical column
e.g
operating_system column is: 'Microsoft', 'MacOS', 'IOS', 'Android', 'Linux'
operating_system column should become: 'DesktopOS', 'MobileOS'

# Create mapping dictionary and replace
mapping = { 'Microsoft': 'DesktopOS', 'MacOS': 'DesktopOS', 'Linux': 'DesktopOS', 'IOS': 'MobileOS', 'Android': 'MobileOS' }
devices['operating_system'] = devices['operating_system'].replace(mapping)
devices['operating_system'].unique() -> in

array( ['DesktopOS', 'MobileOS'], dtype=object ) -> out
'''

# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower()
airlines['dest_region'] = airlines['dest_region'].replace({'eur': 'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())


# Create ranges for categories
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'medium', 'long']

# Create wait_type column
airlines['wait_type'] = pd.cut(
    airlines['wait_min'], bins=label_ranges, labels=label_names)

# Create mappings and replace
mappings = {'Monday': 'weekday', 'Tuesday': 'weekday', 'Wednesday': 'weekday',
            'Thursday': 'weekday', 'Friday': 'weekday', 'Saturday': 'weekend', 'Sunday': 'weekend'}

airlines['day_week'] = airlines['day'].replace(mappings)


'''
Cleaning text data
What is text data?
Type of data             example values
Names                    Alex, Sara ...
Phone numbers          +96171679912 ...
Emails          'adel@datacamp.com' ...
Passwords                           ...

Common text data problems
- Data inconsistency: +96171679912 or 0096171679912 or ... ?
- Fixed length violations: Passwords needs to be at least 8 characters
- Typos: +961.71.679912

e.g
                Full name       Phone number
0         Noelani A. Gray   001-702-397-5143
1          Myles Z. Gomez   001-329-485-0540
2            Gil B. Silva   001-195-492-2338
3      Prescott D. Hardin    +1-297-996-4904 <-- Inconsistent data format
4      Benedict G. Valdez   001-969-820-3536
5        Reece M. Andrews               4138 <-- Length violation
6          Hayfa E. Keith   001-536-175-8444
7         Hedley I. Logan   001-681-552-1823
8        Jack W. Carrillo   001-910-323-5265
9         Lionel M. Davis   001-143-119-9210 

Fixing the phone number column
# Replace '+' with '00'
phones['Phone number'] = phones['Phone number'].str.replace( '+', '00' )

# Replace '-' with nothing
phones['Phone number'] = phones['Phone number'].str.replace( '-', '' )

# Replace phone numbers with lower than 10 digits to NaN
digits = phones['Phone number'].str.len()
phones.loc[ digits < 10, 'Phone number' ] = np.nan

phones -> in

                Full name    Phone number
0         Noelani A. Gray   0017023975143
1          Myles Z. Gomez   0013294850540
2            Gil B. Silva   0011954922338
3      Prescott D. Hardin   0012979964904 
4      Benedict G. Valdez   0019698203536
5        Reece M. Andrews             Nan 
6          Hayfa E. Keith   0015361758444
7         Hedley I. Logan   0016815521823
8        Jack W. Carrillo   0019103235265
9         Lionel M. Davis   0011431199210 -> out

# Find length of each row in Phone number column
sanity_check = phone['Phone number'].str.len()

# Assert minimum phone number length is 10
assert sanity_check.min() >= 10

# Assert all numbers do not have '+' or '-'
assert phone['Phone number'].str.contains('+ | -').any() == False

More complicated examples
e.g
phones.head() -> in

            Full name       Phone number
0       Olga Robinson     +(01706)-25891
1         Justina Kim       +0500-571437
2      Tamekah Henson         +0800-1111
3       Miranda Solis      +07058-879063
4    Caldwell Gilliam     +(016977)-8424 -> out

Regular expressions in action
# Replace letters with nothing
phones['Phone number'] = phones['Phone number'].str.replace( r'\D+', '' )
phones.head() -> in

            Full name  Phone number
0       Olga Robinson    0170625891
1         Justina Kim    0500571437
2      Tamekah Henson      08001111
3       Miranda Solis   07058879063
4    Caldwell Gilliam    0169778424 -> out
'''

# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.", "")

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Mr.", "")

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Miss", "")

# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Ms.", "")

# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False


# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()

# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]

# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40

# Print new survey_response column
print(airlines_survey['survey_response'])


''' Advanced data problems '''

'''
Uniformity
e.g
temperatures = pd.read_csv('temperature.csv')
temperatures.head() -> in

        Date Temperature
0   03.03.19        14.0
1   04.03.19        15.0
2   05.03.19        18.0
3   06.03.19        16.0
4   07.03.19        62.6 -> out

Treating temperature data
C = (F - 32) x 5/9

temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels

# Assert conversion is correct
assert temperatures['Temperature'].max() < 40

Treating data data
e.g
birthdays.head()

            Birthday First name Last name
0           27/27/19      Rowan     Nunez  <- Error
1           03-29-19      Brynn      Yang
2    March 3rd, 2019     Sophia    Reilly
3           24-03-19     Deacon    Prince
4           06-03-19   Griffith      Neal

Datetime formatting
datetime is useful for representing dates
Date                    datetime format
25-12-2019              %d-%m-%Y
December 25th 2019      %c
12-25-2019              %m-%d-%Y
...                     ...

pandas.to_datetime()
- can recognize most formats automatically
- Sometimes fails with erroneous or unrecognizable formats

# Converts to datetime - but won't work!
birthdays['Birthday'] = pd.to_datetime( birthdays['Birthday']) -> in

-> ValueError: month must be 1..12 -> out

# Will work!
birthdays['Birthday'] = pd.to_datetime( birthdays['Birthday'], infer_datetime_format=True, errors = 'coerce' ) -> in

# Attempt to infer format of eac date: infer_datetime_format=True
# Return NA for rows where conversion failed: errors = 'coerce'

    Birthday First name Last name
0        NaT      Rowan     Nunez  
1 2019-03-29      Brynn      Yang
2 2019-03-03     Sophia    Reilly
3 2019-03-24     Deacon    Prince
4 2019-06-03   Griffith      Neal -> out

birthdays['Birthday'] = pd.to_datetime( birthdays['Birthday'].dt.strftime( '%d-%m-%Y')
birthdays.head() -> in

    Birthday First name Last name
0        NaT      Rowan     Nunez  
1 29-03-2019      Brynn      Yang
2 03-03-2019     Sophia    Reilly
3 24-03-2019     Deacon    Prince
4 03-06-2019   Griffith      Neal -> out

Treating ambiguous date data
is 2019-03-03 in August or March?

* Convert to NA and treat accordingly
* Infer format by understanding data source
* Infer format by understanding previous and subsequent data in DataFrame
'''

banking = pd.read_csv(
    'Cleaning Data in Python/banking_dirty.csv', parse_dates=['birth_date'])

# Find values of acct_cur that are equal to 'euro'
acct_eu = banking['acct_cur'] == 'euro'

# Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1

# Unify acct_cur column by changing 'euro' values to 'dollar'
banking.loc[acct_eu, 'acct_cur'] = 'dollar'

# Assert that only dollar currency remains
assert banking['acct_cur'].unique() == 'dollar'


# Print the header of account_opened
print(banking['account_opened'].head())

# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                           # Infer datetime format
                                           infer_datetime_format=True,
                                           # Return missing value for error
                                           errors='coerce')

# Get year of account opened
banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')

# Print acct_year
print(banking['acct_year'])


'''
Cross field validation
This is the use of multiple fields in a dataset to sanity check data integrity

e.g
import pandas as pd

flights = pd.read_csv('flights.csv')
flights.head() -> in

    flight_number economy_class business_class first_class total_passengers
0           DL140           100             60          40              200
1           BA248           130            100          70              300
2          MEA124           100             50          50              200
3          AFR939           140             70          90              300
4          TKA101           130            100          20              250 -> out

sum_classes = flights[ ['economy_class', 'business_class', 'first_class']].sum(axis = 1)
passenger_equ = sum_classes == flights['total_passengers']

# Find and filter out rows with inconsistent passenger totals
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]


e.g
import pandas as pd
import datetime as dt
user.head() -> in

    user_id Age     Birthday
0     32985  22   1998-03-02
1     94387  27   1993-12-05
2     34236  42   1978-11-24
3     12551  31   1989-01-03 
4     55212  18   2002-07-02 -> out

# Convert to datetime and get today's date
users['Birthday'] = pd.to_datetime(users['Birthday'])
today = dt.date.today()

# For each row in the Birthday column, calculate year difference
age_manual = today.year - users['Birthday'].dt.year

# Find instances where ages match
age_equ = age_manual == users['Age']

# Find and filter out rows with inconsistent age
inconsistent_age = users[~age_equ]
consistent_age = users[age_equ]

What to do when we catch inconsistencies?
- Drop inconsistent data
- Set it to missing and impute
- Apply rules from domain knowledge
'''

# Store fund columns to sum against
fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']

# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis=1) == banking['inv_amount']

# Store consistent and inconsistent data
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]

# Store consistent and inconsistent data
print("Number of inconsistent investments: ", inconsistent_inv.shape[0])


# Store today's date and find ages
today = dt.date.today()
ages_manual = today.year - banking['birth_date'].dt.year

# Find rows where age column == ages_manual
age_equ = banking['Age'] == ages_manual

# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]

# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])


'''
Completeness
What is missing data?
- They occur when no data value is stored for a variable in an observation
- Can be represented as NA, NaN, 0, ., ...
- They can be caused by Technical and Human errors

e.g
import pandas as pd
airquality = pd.read_csv('airquality.csv')
print(airquality) -> in

            Date Temperature CO2
987   20/04/2004        16.8 0.0
2119  07/06/2004        18.7 0.8
2451  20/06/2004       -40.0 NaN <- missing value
1984  01/06/2004        19.6 1.8
8299  19/12/2005        11.2 1.2
...          ...        ... ...

# Return missing values
airquality.isna() -> in

        Date Temperature   CO2
987    False       False False
2119   False       False False
2451   False       False  True
1984   False       False False
8299   False       False False -> out

# Get summary of missingness
airquality.isna().sum() -> in

Date 0
Temperature 0
CO2 366
dtype: int64 -> out

Useful package for visualizing and understanding missing data
import missingno as msno
import matplotlib.pyplot as plt

# Visualize missingness
msno.matrix(airquality)
plt.show()

# Isolate missing and complete values aside
missing = airquality[ airquality['co2'].isna()]
complete = airquality[ ~airquality['co2'].isna()]

# Describe complete DataFrame
complete.describe()

        Temperature         CO2
count   8991.000000 8991.000000
mean      18.317829    1.739584
std        8.832116    1.537580
min       -1.900000    0.000000
...             ...         ...
max       44.600000   11.900000

# Describe missing DataFrame
missing.describe()

        Temperature CO2
count    366.000000 0.0
mean     -39.655738 NaN
std        5.988716 NaN
min      -49.000000 NaN
...             ... ...
max      -30.000000 NaN

sorted_airquality = airquality.sort_value(by = 'Temperature')
msno.matrix(sorted_airquality)
plt.show()

# Drop missing values
airquality_dropped = airquality.dropna(subset = ['CO2'])
airquality_dropped.head()

# OR

# Replacing with statistical measures
co2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna( {'CO2': co2_mean} )
airquality_imputed.head()

Missingness types
- MCAR : Missing Completely at Random ( i.e No systematic relationship between missing data and other values e.g Data entry errors when inputting data)
- MAR : Missing at Random ( i.e Systematic relationship between missing data and other observed values e.g Missing ozone data for high temperatures)
- MNAR : Missing Not at Random ( i.e Systematic relationship between missing data and unobserved values e.g Missing temperature values for high temperatures)

How to deal with missing data?
Simple approaches:
- Drop missing data
- Impute with statistical measures (mean, median, mode ...)

More complex approaches:
- Imputing using an algorithmic approach
- Impute with machine learning models
'''

# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]

# Sort banking by age and visualize
banking_sorted = banking.sort_values(by='Age')
msno.matrix(banking_sorted)
plt.show()


# Drop missing values of cust_id
banking_fullid = banking.dropna(subset=['cust_id'])

# Compute estimated acct_amount
acct_imp = banking_fullid['inv_amount'] * 5

# Impute missing acct_amount with corresponding acct_imp
banking_imputed = banking_fullid.fillna({'acct_amount': acct_imp})

# Print number of missing values
print(banking_imputed.isna().sum())


''' Record linkage '''

'''
Comparing strings
Minimum edit distance
- This is a systematic way to identify how close 2 strings are.
- Its the least possible amount of steps needed to transition from one string to another
i.e
* + : Insertion
* - : Deletion
*  Substitution
* <-> : Transposition 

Minimum edit distance algorithms
Algorithm                           Operations
Damerau-Levenshtein     insertion, substitution, deletion, transposition
Levenshtein             insertion, substitution, deletion
Hamming                 substitution only
Jaro distance           transposition only
...                     ...

Possible packages: nltk, thefuzz, textdistance ...

Simple string comparison
# Lets us compare between two strings
from thefuzz import fuzz

# Compare reeding vs reading
fuzz.WRation('reeding', 'reading') -> in

86 -> out (similarity scale from 0 - 100)

Partial strings and different orderings
# Partial string comparison 
fuzz.WRatio('Houston Rockets', 'Rockets') -> in

90 -> out

# Partial string comparison with different order
fuzz.WRatio('Houston Rockets vs Los Angeles Lakers', 'Lakers vs Rockets')

86 -> out

Comparison with arrays
# Import process
from thefuzz import process

# Define string and array of possible matches
string = 'Houston Rockets vs Los Angeles Lakers'
choices = pd.Series( ['Rockets vs Lakers', 'Lakers vs Rockets', 'Houston vs Los Angeles', 'Heat vs Bulls'] )
process.extract(string, choices, limit = 2) -> in

[( 'Rockets vs Lakers', 86, 0), ('Lakers vs Rockets' 86, 1)] -> out

Collapsing categories with string similarity
- Use .replace() to collapse 'eur' into 'Europe'.

e.g
print(survey['state'].unique()) -> in

id          state
0      California
1            Cali
2      Calefornia
3      Calefornie
4      Californie
5       Calfornia
6      Calefernia
7        New York
8   New York City -> out

print(categories) -> in

        state
0   California
1     New York -> out

# For each correct category
for state in categories['state']:
    # Find potential matches in states with typo's
    matches = process.extract(state, survey['state'], limit = survey.shape[0])
        # For each potential match match
        for potential_match in matches:
            # If high similarity score
            if potential_match[1] >= 80:
                # Replace typo with correct category
                survey.loc[ survey['state'] == potential_match[0], 'state'] = state 
'''

restaurants = pd.read_csv('Cleaning Data in Python/restaurants_L2.csv')

# Import process from thefuzz

# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['type'].unique()

# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit=len(unique_types)))

# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit=len(unique_types)))

# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit=len(unique_types)))


# Inspect the unique values of the cuisine_type column
print(restaurants['type'].unique())

# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract(
    'italian', restaurants['type'], limit=restaurants.shape[0])

# Inspect the first 5 matches
print(matches[0:5])

# Iterate through categories
for cuisine in categories:
    # Create a list of matches, comparing cuisine with the cuisine_type column
    matches = process.extract(
        cuisine, restaurants['type'], limit=len(restaurants.type))

    # Iterate through the list of matches
    for match in matches:
        # Check whether the similarity score is greater than or equal to 80
        if match[1] >= 80:
            # If it is, select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
            restaurants.loc[restaurants['type'] == match[0]] = cuisine

# Inspect the final result
print(restaurants['type'].unique())


'''
Generating pairs
e.g
census_A

                given_name  surname     date_of_birth          suburb   state           address_1
rec_id
rec-1070-org      michaela  neumann          19151111   winston hills     cal      stanley street
rec-1016-org      courtney  painter          19161211       richlands     txs   pinkerton circuit
...

census_B

                given_name  surname     date_of_birth          suburb   state           address_1
rec_id
rec-561-dup-0       elton       NaN          19651013      windermere      ny        light street
rec-2642-dup-0   mitchell     maxon          19390212      north ryde     cal       edkins street

Generating pairs 
# Import recordlinkage
import recordlinkage

# Create indexing object
indexer = recordlinkage.Index()

# Generate pairs blocked on state
indexer.block('state')
pairs = indexer.index(census_A, census_B)

# Create a Compare object
compare_cl = recordlinkage.Compare()

# Find exact matches for pairs of date_of_birth and state
compare_cl.exact('date_of_birth', 'date_of_birth', label= 'date_of_birth')
compare_cl.exact('state', 'state', label= 'state')

# Find similar matches for pairs of surname and address_1 using string similarity
compare_cl.string('surname', 'surname', threshold=0.85, label= 'surname')
compare_cl.string('address_1', 'address_1', threshold=0.85, label= 'address_1')

# Find matches
potential_matches = compare_cl.compute(pairs, census_A, census_B)

Finding the only pairs we want
potential_matches[ potential_matches.sum(axis = 1) => 2]
'''

restaurants_new = pd.read_csv(
    'Cleaning Data in Python/restaurants_L2_dirty.csv')

# Create an indexer and object and find possible pairs
indexer = recordlinkage.Index()

# Block pairing on cuisine_type
indexer.block('type')

# Generate pairs
pairs = indexer.index(restaurants, restaurants_new)

# Create a comparison object
comp_cl = recordlinkage.Compare()

# Find exact matches on city, cuisine_types -
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('type', 'type', label='cuisine_type')

# Find similar matches of rest_name
comp_cl.string('name', 'name', label='name', threshold=0.8)

# Get potential matches and print
potential_matches = comp_cl.compute(pairs, restaurants, restaurants_new)
print(potential_matches)

potential_matches[potential_matches.sum(axis=1) >= 3]


'''
Linking DataFrames
Probable matches
matches = potential_matches[ potential_matches.sum(axis = 1) >= 3]
print(matches)

Get the indices
matches.index

# Get indices from census_B only
duplicate_rows = matches.index.get_level_values(1)
print( census_B_index )

Linking DataFrames
# Finding duplicates in census_B
census_B_duplicates = census_B[ census_B.index.isin(duplicate_rows)]

# Finding new rows in census_B
census_B_new = census_B[ ~census_B.index.isin(duplicate_rows)]

# Link the DataFrames!
full_census = census_A.append(census_B_new)
'''

# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis=1) >= 3]

# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)

# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]

# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)
