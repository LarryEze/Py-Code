# Import pandas
import matplotlib.pyplot as plt
import pandas as pd

# Import some of the course datasets
taxi_owners = pd.read_pickle("Joining Data with pandas/taxi_owners.p")
taxi_veh = pd.read_pickle("Joining Data with pandas/taxi_vehicles.p")


'''DATA MERGING BASICS'''

'''
INNER JOIN
#   Inner join will only return rows that have matching values in both tables

table_name = table_name1.merge(
    table_name2, on='column_name', suffixes=('_suffix1', '_suffix2'))
'''

# Merge the taxi_owners and taxi_veh tables
taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid')

# Print the column names of the taxi_own_veh
print(taxi_own_veh.columns)

# Merge the taxi_owners and taxi_veh tables setting a suffix
taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid', suffixes=('_own', '_veh'))

# Print the column names of taxi_own_veh
print(taxi_own_veh.columns)

# Merge the taxi_owners and taxi_veh tables setting a suffix
taxi_own_veh = taxi_owners.merge(taxi_veh, on='vid', suffixes=('_own', '_veh'))

# Print the value_counts to find the most popular fuel_type
print(taxi_own_veh['fuel_type'].value_counts())


# Import some of the course datasets
wards = pd.read_pickle("Joining Data with pandas\ward.p")
census = pd.read_pickle("Joining Data with pandas\census.p")

# Merge the wards and census tables on the ward column
wards_census = wards.merge(census, on='ward')

# Print the shape of wards_census
print('wards_census table shape:', wards_census.shape)


# Print the first few rows of the wards_altered table to view the change
print(wards_altered[['ward']].head())

# Merge the wards_altered and census tables on the ward column
wards_altered_census = wards_altered.merge(census, on='ward')

# Print the shape of wards_altered_census
print('wards_altered_census table shape:', wards_altered_census.shape)


# Print the first few rows of the census_altered table to view the change
print(census_altered[['ward']].head())

# Merge the wards and census_altered tables on the ward column
wards_census_altered = wards.merge(census_altered, on='ward')

# Print the shape of wards_census_altered
print('wards_census_altered table shape:', wards_census_altered.shape)


'''
ONE-TO-MANY RELATIONSHIPS
#   This is when every row in the left table is related to one or more rows in the right table

table_name = table_name1.merge(table_name2, on = 'column_name', suffixes = ('_suffix1', '_suffix2') )
'''

licenses = pd.read_pickle("Joining Data with pandas\licenses.p")
biz_owners = pd.read_pickle("Joining Data with pandas/business_owners.p")

# Merge the licenses and biz_owners table on account
licenses_owners = licenses.merge(biz_owners, on='account')

# Group the results by title then count the number of accounts
counted_df = licenses_owners.groupby('title').agg({'account':  'count'})

# Sort the counted_df in descending order
sorted_df = counted_df.sort_values('account', ascending=False)

# Use .head() method to print the first few rows of sorted_df
print(sorted_df.head())


'''
MERGING MULTIPLE DATAFRAMES
table_name = table_name1.merge( table_name2, on = ['column_name', 'column_name']) \ .merge( table_name3, on =  'column_name', suffixes = ('_suffix1', '_suffix2') )

Merge 3 tables
df1.merge(df2, on='col') \
.merge(df3, on='col')

Merge 4 tables
df1.merge(df2, on='col') \
.merge(df3, on='col') \
.merge(df4, on='col')
'''

ridership = pd.read_pickle("Joining Data with pandas\cta_ridership.p")
cal = pd.read_pickle("Joining Data with pandas\cta_calendar.p")
stations = pd.read_pickle("Joining Data with pandas\stations.p")

# Merge the ridership and cal tables
ridership_cal = ridership.merge(cal, on=['year', 'month', 'day'])

# Merge the ridership, cal, and stations tables
ridership_cal_stations = ridership.merge(cal, on=['year', 'month', 'day']) \
    .merge(stations, on='station_id')

# Create a filter to filter ridership_cal_stations
filter_criteria = ((ridership_cal_stations['month'] == 7)
                   & (ridership_cal_stations['day_type'] == 'Weekday')
                   & (ridership_cal_stations['station_name'] == 'Wilson'))

# Use .loc and the filter to select for rides
print(ridership_cal_stations.loc[filter_criteria, 'rides'].sum())


zip_demo = pd.read_pickle("Joining Data with pandas\zip_demo.p")

# Merge licenses and zip_demo, on zip; and merge the wards on ward
licenses_zip_ward = licenses.merge(zip_demo, on='zip') \
    .merge(wards, on='ward')

# Print the results by alderman and show median income
print(licenses_zip_ward.groupby('alderman').agg({'income': 'median'}))


land_use = pd.read_pickle("Joining Data with pandas\land_use.p")

# Merge land_use and census and merge result with licenses including suffixes
land_cen_lic = land_use.merge(census, on='ward') \
    .merge(licenses, on='ward', suffixes=('_cen', '_lic'))

# Group by ward, pop_2010, and vacant, then count the # of accounts
pop_vac_lic = land_cen_lic.groupby(['ward', 'pop_2010', 'vacant'],
                                   as_index=False).agg({'account': 'count'})

# Sort pop_vac_lic and print the results
sorted_pop_vac_lic = pop_vac_lic.sort_values(['vacant', 'account', 'pop_2010'],
                                             ascending=(False, True, True))

# Print the top few rows of sorted_pop_vac_lic
print(sorted_pop_vac_lic.head())


'''MERGING TABLES WITH DIFFERENT JOIN TYPES'''

'''
LEFT JOIN
#   Left join will only return rows from the left table and only those rows from the right table where key columns match

table_name = table_name1.merge(table_name2, on = 'column_name', suffixes = ('_suffix1', '_suffix2'), how='left')
'''

movies = pd.read_pickle("Joining Data with pandas\movies.p")
financials = pd.read_pickle("Joining Data with pandas/financials.p")

# Merge the movies table with the financials table with a left join
movies_financials = movies.merge(financials, on='id', how='left')

# Count the number of rows in the budget column that are missing
number_of_missing_fin = movies_financials['budget'].isnull().sum()

# Print the number of movies missing financials
print(number_of_missing_fin)


toy_story = movies[(movies['title'] == 'Toy Story') | (
    movies['title'] == 'Toy Story 2') | (movies['title'] == 'Toy Story 3')]
taglines = pd.read_pickle("Joining Data with pandas/taglines.p")

# Merge the toy_story and taglines tables with a inner join
toystory_tag = toy_story.merge(taglines, on='id')

# Print the rows and shape of toystory_tag
print(toystory_tag)
print(toystory_tag.shape)


'''
RIGHT JOINS
#   Right join will only return rows from the right table and only those rows from the left table where key columns match

table_name = table_name1.merge(table_name2, how='right', left_on = 'column_name', right_on = 'column_name', suffixes = ('_suffix1', '_suffix2'))

OUTER JOINS
#   Outer join will return all rows from both tables regardless if there is a match between the tables

table_name = table_name1.merge(table_name2, how='outer', left_on = 'column_name', right_on = 'column_name', suffixes = ('_suffix1', '_suffix2')) 
'''

movie_to_genres = pd.read_pickle("Joining Data with pandas\movie_to_genres.p")

scifi_movies = movie_to_genres[movie_to_genres['genre'] == 'Science Fiction']
action_movies = movie_to_genres[movie_to_genres['genre'] == 'Action']

# Merge action_movies to the scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, on='movie_id', how='right',
                                   suffixes=('_act', '_sci'))

# Print the first few rows of action_scifi to see the structure
print(action_scifi.head())

# From action_scifi, select only the rows where the genre_act column is null
scifi_only = action_scifi[action_scifi['genre_act'].isnull()]

# Merge the movies and scifi_only tables with an inner join
movies_and_scifi_only = movies.merge(
    scifi_only, left_on='id', right_on='movie_id')

# Print the first few rows and shape of movies_and_scifi_only
print(movies_and_scifi_only.head())
print(movies_and_scifi_only.shape)


pop_movies = movies.nlargest(10, 'popularity')

# Use right join to merge the movie_to_genres and pop_movies tables
genres_movies = movie_to_genres.merge(
    pop_movies, how='right', left_on='movie_id', right_on='id')

# Count the number of genres
genre_count = genres_movies.groupby('genre').agg({'id': 'count'})

# Plot a bar chart of the genre_count
genre_count.plot(kind='bar')
plt.show()


cast = pd.read_pickle('Joining Data with pandas/casts.p')

mov = movies.merge(
    cast, how='right', left_on='id', right_on='movie_id')

iron_1_actors = mov[mov['title'] == 'Iron Man'][['character', 'id_y', 'name']]
iron_2_actors = mov[mov['title'] ==
                    'Iron Man 2'][['character', 'id_y', 'name']]


# Merge iron_1_actors to iron_2_actors on id with outer join using suffixes
iron_1_and_2 = iron_1_actors.merge(iron_2_actors,
                                   how='outer',
                                   on='id_y',
                                   suffixes=('_1', '_2'))

# Create an index that returns true if name_1 or name_2 are null
m = ((iron_1_and_2['name_1'].isnull() == True) |
     (iron_1_and_2['name_2'].isnull() == True))

# Print the first few rows of iron_1_and_2
print(iron_1_and_2[m].head())


'''
SELF JOIN
table_name = table_name1.merge(table_name1, left_on = 'column_name', right_on = 'column_name', suffixes = ('_suffix1', '_suffix2'))
'''

crews = pd.read_pickle('Joining Data with pandas\crews.p')

# Merge the crews table to itself
crews_self_merged = crews.merge(crews, on='id', how='inner',
                                suffixes=('_dir', '_crew'))

# Create a boolean index to select the appropriate rows
boolean_filter = ((crews_self_merged['job_dir'] == 'Director') &
                  (crews_self_merged['job_crew'] != 'Director'))
direct_crews = crews_self_merged[boolean_filter]

# Print the first few rows of direct_crews
print(direct_crews.head())


'''
MERGING ON INDEXES
table_name = table_name1.merge(table_name2, on = 'column_name', how='left', suffixes = ('_suffix1', '_suffix2'))

table_name = table_name1.merge(table_name2, on = ['column_name', 'column_name'], suffixes = ('_suffix1', '_suffix2'))

table_name = table_name1.merge(table_name1, left_on = 'column_name', left_index=True, right_on = 'column_name', right_index=True, suffixes = ('_suffix1', '_suffix2'))

setting an index
table_name = pd.read_csv('dataframe.csv', index_col=['column_name'])

table_name = pd.read_csv('dataframe.csv', index_col=['column_name', 'column_name']) - Multi-Index
'''

ratings = pd.read_pickle('Joining Data with pandas/ratings.p')

# Merge to the movies table the ratings table on the index
movies_ratings = movies.merge(ratings, on='id', how='left')

# Print the first few rows of movies_ratings
print(movies_ratings.head())


sequels = pd.read_pickle('Joining Data with pandas\sequels.p')

# Merge sequels and financials on index id
sequels_fin = sequels.merge(financials, on='id', how='left')

# Self merge with suffixes as inner join with left on sequel and right on id
orig_seq = sequels_fin.merge(sequels_fin, how='inner', left_on='sequel',
                             right_on='id', right_index=False,
                             suffixes=('_org', '_seq'))

# Add calculation to subtract revenue_org from revenue_seq
orig_seq['diff'] = orig_seq['revenue_seq'] - orig_seq['revenue_org']

# Select the title_org, title_seq, and diff
titles_diff = orig_seq[['title_org', 'title_seq', 'diff']]

# Print the first rows of the sorted titles_diff
print(titles_diff.sort_values('diff', ascending=False).head())


''' ADVANCED MERGING AND CONCATENATING '''

'''
FILTERING JOINS

SEMI JOIN
they filter the left table down to those observations that have a match in the right table
#   Returns the intersection, similar to an inner join
#   Returns only columns from the left table and not the right
#   No duplicates

step 1 ( Inner join )
table_name = table_name1.merge(table_name2, on = 'column_name' )
step 2 ( semi join )
table_name1['column_name'].isin(table_name['column_name'])
step 3 ( semi join )
table_name3 = table_name1[table_name1['column_name'].isin(table_name['column_name'])]

ANTI JOIN
they return the observations in the left table that do not have a matching observation in the right table
#   Returns the left table, excluding the intersection
#   Returns only columns from the left table and not the right

step 1 ( left join )
table_name = table_name1.merge(table_name2, on = 'column_name', how='left', indicator=True )
step 2 ( anti join )
table_name3 = table_name.loc[table_name['_merge'] == 'left_only', 'column_name']
step 3 ( anti join )
table_name4 = table_name1[table_name1['column_name'].isin(table_name3)]
'''

# Merge employees and top_cust
empl_cust = employees.merge(top_cust, on='srid',
                            how='left', indicator=True)

# Select the srid column where _merge is left_only
srid_list = empl_cust.loc[empl_cust['_merge'] == 'left_only', 'srid']

# Get employees not working with top customers
print(employees[employees['srid'].isin(srid_list)])


# Merge the non_mus_tck and top_invoices tables on tid
tracks_invoices = non_mus_tcks.merge(top_invoices, on='tid')

# Use .isin() to subset non_mus_tcks to rows with tid in tracks_invoices
top_tracks = non_mus_tcks[non_mus_tcks['tid'].isin(tracks_invoices['tid'])]

# Group the top_tracks by gid and count the tid rows
cnt_by_gid = top_tracks.groupby(['gid'], as_index=False).agg({'tid': 'count'})

# Merge the genres table to cnt_by_gid on gid and print
print(cnt_by_gid.merge(genres, on='gid'))


'''
CONCATENATE DATAFRAMES TOGETHER VERTICALLY
pandas .concat() method can concatenate both vertical and horizontal
axis = 0, vertical

Basic concatenation (same column names)
pd.concat( [table_name1, table_name2, table_name3], ignore_index=True )

Setting labels to original tables
pd.concat( [table_name1, table_name2, table_name3], ignore_index=False, keys=['key1', 'key2', 'key3'] )
#   You cannot add a key and ignore the index at the same time

Concatenate tables with different column names
pd.concat( [table_name1, table_name2], sort=True ) - Alphabetically sort the different column names in the result

pd.concat( [table_name1, table_name2], join='inner' ) - If you only want the matching columns between tables
#   sort has no effect when join = 'inner'. the default join is 'outer'

Using append method
.append()
#   Simplified version of the .concat() method
#   Supports: ignore_index, and sort arguments
#   Does not support: keys and join (i.e always join = 'outer')

Append the tables
table_name1.append([table_name2, table_name3], ignore_index=True, sort=True)
'''

# Concatenate the tracks
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               sort=True)
print(tracks_from_albums)

# Concatenate the tracks so the index goes from 0 to n-1
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               ignore_index=True,
                               sort=True)
print(tracks_from_albums)

# Concatenate the tracks, show only columns names that are in all tables
tracks_from_albums = pd.concat([tracks_master, tracks_ride, tracks_st],
                               join='inner',
                               sort=True)
print(tracks_from_albums)


# Concatenate the tables and add keys
inv_jul_thr_sep = pd.concat([inv_jul, inv_aug, inv_sep],
                            keys=['7Jul', '8Aug', '9Sep'])

# Group the invoices by the index keys and find avg of the total column
avg_inv_by_month = inv_jul_thr_sep.groupby(level=0).agg({'total': 'mean'})

# Bar plot of avg_inv_by_month
avg_inv_by_month.plot(kind='bar')
plt.show()


# Use the .append() method to combine the tracks tables
metallica_tracks = tracks_ride.append([tracks_master, tracks_st], sort=False)

# Merge metallica_tracks and invoice_items
tracks_invoices = metallica_tracks.merge(invoice_items, on='tid')

# For each tid and name sum the quantity sold
tracks_sold = tracks_invoices.groupby(['tid', 'name']).agg({'quantity': 'sum'})

# Sort in descending order by quantity and print the results
print(tracks_sold.sort_values('quantity', ascending=False))


'''
Verifying Integrity
#   Possible merging issue:
#   Unintentional one-to-many relationship
#   Unintentional many_to_many relationship

Validating merges
.merge( validate=None ):

Checks if merge is of specified type
#   'one_to_one'
#   'one_to_many'
#   'many_to_one'
#   'many_to_many'

Merge validate: one_to_one
table_name1.merge(table_name2, on='column_name', validate='one_to_one')

Merge validate: one_to_many
table_name1.merge(table_name2, on='column_name', validate='one_to_many')

Possible concatenating issue:
#   Duplicate records possibly unintentionally introduced

Verifying concatenations
.concat(verify_integrity=False):

Check whatever the new concatenated index contains duplicates
#   Default value is False

pd.concat( [table_name1, table_name2], verify_integrity=True )
'''

# Concatenate the classic tables vertically
classic_18_19 = pd.concat([classic_18, classic_19], ignore_index=True)

# Concatenate the pop tables vertically
pop_18_19 = pd.concat([pop_18, pop_19], ignore_index=True)

# Merge classic_18_19 with pop_18_19
classic_pop = classic_18_19.merge(pop_18_19, on='tid')

# Using .isin(), filter classic_18_19 rows where tid is in classic_pop
popular_classic = classic_18_19[classic_18_19['tid'].isin(classic_pop['tid'])]

# Print popular chart
print(popular_classic)


'''
MERGING ORDERED AND TIME-SERIES DATA
'''

'''
Using merge_ordered()
.merge() method: default is inner
calling the method
table_name1.merge(table_name2, on='column_name')

.merge_ordered() method: default is outer
import pandas as pd
pd.merge_ordered(table_name1, table_name2, on='column_name')

When to use the merge_ordered() method
#   Ordered data / time series
#   Filling in missing values

Forward Fill
This will interpolate missing data by filling the missing values with the previous value
pd.merge_ordered(table_name1, table_name2, on='column_name', suffixes=('_suf1', '_suf2'), fill_method='ffill')

* ffill = Forward fill
'''

WorldBank_GDP = pd.read_csv('Joining Data with pandas\WorldBank_GDP.csv')
Usa_gdp = WorldBank_GDP[WorldBank_GDP['Country Code'] == 'USA']
gdp = Usa_gdp[['Country Code', 'Year', 'GDP']]

WorldBank_POP = pd.read_csv('Joining Data with pandas\WorldBank_POP.csv')
Usa_pop = WorldBank_POP[WorldBank_POP['Country Code'] == 'USA']
pop = Usa_pop[['Country Code', 'Year', 'Pop']]

sp500 = pd.read_csv('Joining Data with pandas\S&P500.csv')

# Use merge_ordered() to merge gdp and sp500 on year and date
gdp_sp500 = pd.merge_ordered(gdp, sp500, left_on='Year', right_on='Date',
                             how='left')

# Print gdp_sp500
print(gdp_sp500)

# Use merge_ordered() to merge gdp and sp500, interpolate missing value
gdp_sp500 = pd.merge_ordered(
    gdp, sp500, left_on='Year', right_on='Date', how='left', fill_method='ffill')

# Print gdp_sp500
print(gdp_sp500)

# Use merge_ordered() to merge gdp and sp500, interpolate missing value
gdp_sp500 = pd.merge_ordered(gdp, sp500, left_on='Year', right_on='Date',
                             how='left',  fill_method='ffill')

# Subset the gdp and returns columns
gdp_returns = gdp_sp500[['GDP', 'Returns']]

# Print gdp_returns correlation
print(gdp_returns.corr())


# Use merge_ordered() to merge inflation, unemployment with inner join
inflation_unemploy = pd.merge_ordered(
    inflation, unemployment, on='date', how='inner')

# Print inflation_unemploy
print(inflation_unemploy)

# Plot a scatter plot of unemployment_rate vs cpi of inflation_unemploy
inflation_unemploy.plot(x='unemployment_rate', y='cpi', kind='scatter')
plt.show()


# Merge gdp and pop on date and country with fill and notice rows 2 and 3
ctry_date = pd.merge_ordered(gdp, pop, on=['date', 'country'],
                             fill_method='ffill')

# Print ctry_date
print(ctry_date)

# Merge gdp and pop on country and date with fill
date_ctry = pd.merge_ordered(gdp, pop, on=['country', 'date'],
                             fill_method='ffill')

# Print date_ctry
print(date_ctry)


'''
Using merge_asof()
#   Similar to a merge_ordered() left join
Similar features as merge_ordered()
#   Match on the nearest key column and not exact matches
Merged 'on' columns must be sorted

pd.merge_asof(table_name1, table_name2, on='column_name', suffixes=('_suf1', '_suf2'))
#   This will select the first row in the right table whose 'on' key column is less than or equal to the left's key column

pd.merge_asof(table_name1, table_name2, on='column_name', suffixes=('_suf1', '_suf2'), direction = 'forward')
#   This will select the first row in the right table whose 'on' key column is greater than or equal to the left's key column

#   direction default is 'backward'
#   direction can be set to 'nearest' which will return the nearest row in the right table regardless if it is forward or backwards 

When to use merge_asof()
#   Data sampled from a process
#   Developing a training set (no data leakage)
'''

# Use merge_asof() to merge jpm and wells
jpm_wells = pd.merge_asof(jpm, wells, on='date_time',
                          suffixes=('', '_wells'), direction='nearest')

# Use merge_asof() to merge jpm_wells and bac
jpm_wells_bac = pd.merge_asof(jpm_wells, bac, on='date_time', suffixes=(
    '_jpm', '_bac'), direction='nearest')

# Compute price diff
price_diffs = jpm_wells_bac.diff()

# Plot the price diff of the close of jpm, wells and bac only
price_diffs.plot(y=['close_jpm', 'close_wells', 'close_bac'], kind='line')
plt.show()


# Merge gdp and recession on date using merge_asof()
gdp_recession = pd.merge_asof(gdp, recession, on='date')

# Create a list based on the row value of gdp_recession['econ_status']
is_recession = ['r' if s ==
                'recession' else 'g' for s in gdp_recession['econ_status']]

# Plot a bar chart of gdp_recession
gdp_recession.plot(kind='bar', y='gdp', x='date', color=is_recession, rot=90)
plt.show()


'''
SELECTING DATA WITH .query()
table_name.query(' SOME SELECTION STATEMENT ')
'''

print(social_fin.head())

social_fin.query('value > 50000000')

social_fin.query('financial == "gross_profit" and value > 100000')

social_fin.query('financial == "net_income" and value < 0')

social_fin.query('financial == "total_revenue" and company == "facebook"')


# Merge gdp and pop on date and country with fill
gdp_pop = pd.merge_ordered(
    gdp, pop, on=['country', 'date'], fill_method='ffill')

# Add a column named gdp_per_capita to gdp_pop that divides the gdp by pop
gdp_pop['gdp_per_capita'] = gdp_pop['gdp'] / gdp_pop['pop']

# Pivot data so gdp_per_capita, where index is date and columns is country
gdp_pivot = gdp_pop.pivot_table('gdp_per_capita', 'date', 'country')

# Select dates equal to or greater than 1991-01-01
recent_gdp_pop = gdp_pivot.query('date >= "1991-01-01"')

# Plot recent_gdp_pop
recent_gdp_pop.plot(rot=90)
plt.show()


'''
RESHAPING DATA WITH .melt()
table_name = table_name.melt(id_vars=['column_name', 'column_name'])

#   id_vars are the columns in the original dataset that shouldn't change

table_name = table_name.melt(id_vars=['column_name', 'column_name'], value_vars=['column_name', 'column_name'], var_name=['column_name'], value_name=['column_name'])

#   value_vars are the columns in the original dataset that will be unpivoted
#   var_name will allow to set the name of the variable column in the output
#   value_name will allow to set the name of the value column in the output
'''

# unpivot everything besides the year column
ur_tall = ur_wide.melt(id_vars='year', var_name='month',
                       value_name='unempl_rate')

# Create a date column using the month and year columns of ur_tall
ur_tall['date'] = pd.to_datetime(ur_tall['year'] + '-' + ur_tall['month'])

# Sort ur_tall by date in ascending order
ur_sorted = ur_tall.sort_values('date')

# Plot the unempl_rate by date
ur_sorted.plot(x='date', y='unempl_rate')
plt.show()


# Use melt on ten_yr, unpivot everything besides the metric column
bond_perc = ten_yr.melt(id_vars='metric', var_name='date', value_name='close')

# Use query on bond_perc to select only the rows where metric=close
bond_perc_close = bond_perc.query('metric == "close"')

# Merge (ordered) dji and bond_perc_close on date with an inner join
dow_bond = pd.merge_ordered(
    dji, bond_perc_close, on='date', how='inner', suffixes=('_dow', '_bond'))

# Plot only the close_dow and close_bond columns
dow_bond.plot(y=['close_dow', 'close_bond'], x='date', rot=90)
plt.show()
