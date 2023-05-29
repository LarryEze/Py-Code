''' Dates and Calendars '''

'''
Dates in Python
Creating date objects
e.g
# Import date
from datetime import date
# Create dates
two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]

Attributes of a date
# Import date
from datetime import date

# Create dates
two_hurricanes_dates = [date(2016, 10, 7), date(2017, 6, 21)]
print(two_hurricanes_dates[0].year)
print(two_hurricanes_dates[0].month)
print(two_hurricanes_dates[0].day) -> in

2016
10
7 -> out

Finding the weekday of a date
print(two_hurricanes_dates[0].weekday()) -> in

4 -> out

* Weekdays in Python
- 0 = Monday
- 1 = Tuesday
- 2 = Wednesday
- ...
- 6 = Sunday

What day does the week begin for you? 
It depends where you are from! In the United States, Canada, and Japan, Sunday is often considered the first day of the week. Everywhere else, it usually begins on Monday.
'''

# Import date from datetime
from datetime import date

# Create a date object
hurricane_andrew = date(1992, 8, 24)

# Which day of the week is the date?
print(hurricane_andrew.weekday())


# Counter for how many before June 1
early_hurricanes = 0

# We loop over the dates
for hurricane in florida_hurricane_dates:
    # Check if the month is before June (month number 6)
    if hurricane.month < 6:
        early_hurricanes = early_hurricanes + 1
    
print(early_hurricanes)


'''
Math with dates
e.g
# Import date
from datetime import date
# Create our dates
d1 = date(2017, 11, 5)
d2 = date(2017, 12, 4)
l = [d1, d2]

print(min(l)) -> in
2017-11-05 -> out

# Subtract two dates
delta = d2 - d1

print(delta.days) -> in
29 -> out

# Import timedelta
from datetime import timedelta
# Create a 29 dat timedelta
td = timedelta(days = 29)

print(d1 + td) -> in
2017-12-04 -> out
'''

# Import date
from datetime import date

# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)


# A dictionary to count hurricanes per calendar month
hurricanes_each_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6:0, 7: 0, 8:0, 9:0, 10:0, 11:0, 12:0}

# Loop over all hurricanes
for hurricane in florida_hurricane_dates:
    # Pull out the month
    month = hurricane.month
    # Increment the count in your dictionary by one
    hurricanes_each_month[month] += 1

print(hurricanes_each_month)


# Print the first and last scrambled dates
print(dates_scrambled[0])
print(dates_scrambled[-1])

# Put the dates in order
dates_ordered = sorted(dates_scrambled)

# Print the first and last ordered dates
print(dates_ordered[0])
print(dates_ordered[-1])


'''
Turning dates into strings
ISO 8601 format
from datetime import date

# Example date
d = date(2017, 11, 5)
# ISO format: YYYY-MM-DD
print(d) -> in

2017-11-05 -> out

# Express the date in ISO 8601 format and put it in a list
print( [d.isoformat()] ) -> in

['2017-11-05'] -> out

# A few dates that computers once had trouble with
some_dates = ['2000-01-01', '1999-12-31']
# Print them in order
print(sorted(some_dates)) -> in

['1999-12-31', '2000-01-01'] -> out

Every other format
d.strftime() 
e.g
d = date(2017, 1, 5)

print(d.strftime('%Y')) -> in
2017 -> out

# Format string with more text in it
print(d.strftime('Year is %Y')) -> in
Year is 2017 -> out

# Format: YYYY/MM/DD
print(d.strftime('%Y/%m/%d')) -> in

2017/01/05 -> out
'''

# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)


# Import date
from datetime import date

# Create a date object
andrew = date(1992, 8, 26)

# Print the date in the format 'YYYY-MM'
print(andrew.strftime('%Y-%m'))

# Print the date in the format 'MONTH (YYYY)'
print(andrew.strftime('%B (%Y)'))

# Print the date in the format 'YYYY-DDD' (where DDD is the day of the year)
print(andrew.strftime('%Y-%j'))


''' Combining Dates and Times '''

'''
Dates and times
e.g
# Import datetime
from datetime import datetime

dt = datetime(year=2017, month=10, day=1, hour=15, minute=23, second=25, microsecond=500000)

print(dt) -> in
2017-10-01 15:23:25.500000 -> out

Replacing parts of a datetime
dt_hr = dt.replace(minute=0, second=0, microsecond=0)
print(dt_hr) -> in


2017-10-01 15:00:00 -> out
'''

# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 10, 1, 15, 26, 26)

# Print the results in ISO 8601 format
print(dt.isoformat())


# Create a datetime object
dt1 = datetime(2017, 12, 31, 15, 19, 13)

# Print the results in ISO 8601 format
print(dt1.isoformat())


# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt_old)


# Create dictionary to hold results
trip_counts = {'AM': 0, 'PM': 0}

# Loop over all trips
for trip in onebike_datetimes:
    # Check to see if the trip starts before noon
    if trip['start'].hour < 12:
        # Increment the counter for before noon
        trip_counts['AM'] += 1
    else:
        # Increment the counter for after noon
        trip_counts['PM'] += 1

print(trip_counts)


'''
Printing and parsing datetimes
Printing datetimes
e.g
# Create datetime
dt = datetime(2017, 12, 30, 15, 19, 13)
print(dt.strftime('%Y-%m-%d')) -> in

2017-12-30 -> out

print(dt.strftime('%Y-%m-%d %H:%M:%S')) -> in
2017-12-30 15:19:13 -> out

print(dt.strftime('%H:%M:%S on %Y/%m/%d/')) -> in
15:19:13 on 2017/12/30  -> out

ISO 8601 Format
# ISO 8601 format
print(dt.isoformat()) -> in

2017-12-30T15:19:13 -> out

Parsing datetimes with strptime
e.g
# Import datetime
from datetime import datetime

dt = datetime.strptime('12/30/2017 15:19:13', '%m/%d/%Y %H:%M:%S')

print(type(dt)) -> in
<class 'datetime.datetime'> -> out

print(dt) -> in
2017-12-30 15:19:13

# A timestamp
ts = 1514665153.0

# Convert to datetime and print
print(datetime.fromtimestamp(ts)) -> in

2017-12-30 15:19:13 -> out

# timestamp counts number of seconds since january 1st, 1970 since the date is regarded as the birth of modern-style computers
'''

# Import the datetime class
from datetime import datetime

# Starting string, in YYYY-MM-DD HH:MM:SS format
s = '2017-02-03 00:00:01'

# Write a format string to parse s
fmt = '%Y-%m-%d %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# Starting string, in YYYY-MM-DD format
s = '2030-10-15'

# Write a format string to parse s
fmt = '%Y-%m-%d'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# Starting string, in MM/DD/YYYY HH:MM:SS format
s = '12/15/1986 08:00:00'

# Write a format string to parse s
fmt = '%m/%d/%Y %H:%M:%S'

# Create a datetime object d
d = datetime.strptime(s, fmt)

# Print d
print(d)


# Write down the format string
fmt = "%Y-%m-%d %H:%M:%S"

# Initialize a list for holding the pairs of datetime objects
onebike_datetimes = []

# Loop over all trips
for (start, end) in onebike_datetime_strings:
    trip = {'start': datetime.strptime(start, fmt), 'end': datetime.strptime(end, fmt)}

# Append the trip
onebike_datetimes.append(trip)


# Import datetime
from datetime import datetime

# Pull out the start of the first trip
first_start = onebike_datetimes[0]['start']

# Format to feed to strftime()
fmt = "%Y-%m-%dT%H:%M:%S"

# Print out date with .isoformat(), then with .strftime() to compare
print(first_start.isoformat())
print(first_start.strftime(fmt))


# Import datetime
from datetime import datetime

# Starting timestamps
timestamps = [1514665153, 1514664543]

# Datetime objects
dts = []

# Loop
for ts in timestamps:
    dts.append(datetime.fromtimestamp(ts))

# Print results
print(dts)


'''
Working with durations
e.g
# Create example datetimes
start = datetime(2017, 10, 8, 23, 46, 47)
end = datetime(2017, 10, 9, 0, 10, 57)

# Subtract datetimes to create a timedelta
duration = end - start

print(duration.total_seconds()) -> in
1450.0 -> out

Creating timedeltas
# Import timedelta
from datetime import timedelta

# Create a timedelta
delta1 = timedelta(seconds=1) 

print(start) -> in
2017-10-08 23:46:47 -> out

# One seond later
print( start + delta1) -> in

2017-10-08 23:46:48 -> out

# Create a one day and one second timedelta
delta2 = timedelta(days=1, seconds=1)

print(start) -> in
2017-10-08 23:46:47 -> out

# One day and one second later
print(start + delta2) -> in

2017-10-09 23:46: 48 -> out

Negative timedeltas
# Create a negative timedelta of one week
delta3 = timedelta(weeks = -1)

print(start) -> in
2017-10-08 23:46:47 -> out

# One week earlier
print( start + delta3) -> in

2017-10-01 23:46:47 -> out

# Same, but we'll subtract this time
delta4 = timedelta(weeks = 1)

print(start) -> in
2017-10-08 23:46:47 -> out

# One week earlier
print( start - delta4) -> in

2017-10-01 23:46:47 -> out
'''

# Initialize a list for all the trip durations
onebike_durations = []

for trip in onebike_datetimes:
    # Create a timedelta object corresponding to the length of the trip
    trip_duration = trip['end'] - trip['start']

    # Get the total elapsed seconds in trip_duration
    trip_length_seconds = trip_duration.total_seconds()

    # Append the results to our list
    onebike_durations.append(trip_length_seconds)

# What was the total duration of all trips?
total_elapsed_time = sum(onebike_durations)

# What was the total number of trips?
number_of_trips = len(onebike_durations)

# Divide the total duration by the number of trips
print(total_elapsed_time / number_of_trips)

# Calculate shortest and longest trips
shortest_trip = min(onebike_durations)
longest_trip = max(onebike_durations)

# Print out the results
print("The shortest trip was " + str(shortest_trip) + " seconds")
print("The longest trip was " + str(longest_trip) + " seconds")


''' Time Zones and Daylight Saving '''

'''
UTC offsets
UTC
# Import relevant classes
from datetime import datetime, timedelta, timezone

# US Eastern Standard time zone
ET = timezone(timedelta(hours = -5))
# Timezone-aware datetime
dt = datetime(2017, 12, 30, 15, 9, 3, tzinfo = ET)

print(dt) -> in
2017-12-30 15:09:03-05:00 -> out

# India Standard time zone
IST = timezone(timedelta(hours= 5, minutes=30))
# Convert to IST
print(dt.astimezone(IST)) -> in

2017-12-31 01:39:03+05:30 -> out

Adjusting timezone vs changing tzinfo
print(dt) -> in

2017-12-30 15:09:03-05:00 -> out

print(dt.replace(tzinfo=timezone.utc)) -> in
2017-12-30 15:09:03-00:00 -> out

# Change original to match UTC
print(dt.astimezone(timezone.utc)) -> in

2017-12-30 20:09:03+00:00 -> out
'''

# Import datetime, timezone
from datetime import datetime, timezone

# October 1, 2017 at 15:26:26, UTC
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=timezone.utc)

# Print results
print(dt.isoformat())


# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Pacific Standard Time, or UTC-8
pst = timezone(timedelta(hours= -8))

# October 1, 2017 at 15:26:26, UTC-8
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=pst)

# Print results
print(dt.isoformat())


# Import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone

# Create a timezone for Australian Eastern Daylight Time, or UTC+11
aedt = timezone(timedelta(hours= 11))

# October 1, 2017 at 15:26:26, UTC+11
dt = datetime(2017, 10, 1, 15, 26, 26, tzinfo=aedt)

# Print results
print(dt.isoformat())


# Create a timezone object corresponding to UTC-4
edt = timezone(timedelta(hours= -4))

# Loop over trips, updating the start and end datetimes to be in UTC-4
for trip in onebike_datetimes[:10]:
    # Update trip['start'] and trip['end']
    trip['start'] = trip['start'].replace(tzinfo=edt)
    trip['end'] = trip['end'].replace(tzinfo=edt)


# Loop over the trips
for trip in onebike_datetimes[:10]:
    # Pull out the start
    dt = trip['start']
    # Move dt to be in UTC
    dt = dt.astimezone(timezone.utc)

# Print the start time in UTC
print('Original:', trip['start'], '| UTC:', dt.isoformat())


'''
Time zone database
- Format: 'Continent/City'
- Examples:
* 'America/New_York'
* 'America/Mexico_City'
* 'Europe/London'
* 'Africa/Accra'
e.g
# Imports
from datetime import datetime
from dateutil import tz

# Eastern time
et = tz.gettz('America/New_York')

# Last ride
last = datetime(2017, 12, 30, 15, 9, 3, tzinfo=et)

print(last) -> in
2017-12-30 15:09:03-05:00 -> out

# First ride
first = datetime(2017, 10, 1, 15, 23, 25, tzinfo=et)

print(first) -> in
2017-10-01 15:23:25-04:00 -> out
'''

# Import tz
from dateutil import tz

# Create a timezone object for Eastern Time
et = tz.gettz('America/New_York')

# Loop over trips, updating the datetimes to be in Eastern Time
for trip in onebike_datetimes[:10]:
    # Update trip['start'] and trip['end']
    trip['start'] = trip['start'].replace(tzinfo = et)
    trip['end'] = trip['end'].replace(tzinfo = et)


# Create the timezone object
uk = tz.gettz('Europe/London')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in the UK?
notlocal = local.astimezone(uk)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


# Create the timezone object
ist = tz.gettz('Asia/Kolkata')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in India?
notlocal = local.astimezone(ist)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


# Create the timezone object
sm = tz.gettz('Pacific/Apia')

# Pull out the start of the first trip
local = onebike_datetimes[0]['start']

# What time was it in Samoa?
notlocal = local.astimezone(sm)

# Print them out and see the difference
print(local.isoformat())
print(notlocal.isoformat())


'''
Starting daylight saving time
Start of Daylight Saving Time
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59)
spring_ahead_159am.isoformat() -> in

'2017-03-12T01:59:59' -> out

spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0)
spring_ahead_3am.isoformat() -> in

'2017-03-12T03:00:00' -> out

(spring_ahead_3am - spring_ahead_159am).total_seconds() ->in

3601.0 -> out

EST : Eastern Standard Time
EDT: Eastern Daylight Time

from datetime import timezone, timedelta

EST = timezone(timedelta(hours = -5))
EDT = timezone(timedelta(hours = -4))

spring_ahead_159am = spring_ahead_159am.replace(tzinfo = EST)
spring_ahead_159am.isoformat() -> in

'2017-03-12T01:59:59-05:00' -> out

spring_ahead_3am = spring_ahead_3am.replace(tzinfo = EDT)
spring_ahead_3am.isoformat() -> in

'2017-03-12T03:00:00-04:00' -> out

(spring_ahead_3am - spring_ahead_159am).seconds ->in

1 -> out

Using dateutil
# Import tz
from dateutil import tz

# Create eastern timezone
eastern = tz.gettz('America/New_York')

# 2017-03-12 01:59:59 in Eastern Time (EST)
spring_ahead_159am = datetime(2017, 3, 12, 1, 59, 59, tzinfo = eastern)

# 2017-03-12 03:00:00 in Eastern Time (EDT)
spring_ahead_3am = datetime(2017, 3, 12, 3, 0, 0, tzinfo = eastern)
'''

# Import datetime, timedelta, tz, timezone
from datetime import datetime, timedelta, timezone
from dateutil import tz

# Start on March 12, 2017, midnight, then add 6 hours
start = datetime(2017, 3, 12, tzinfo = tz.gettz('America/New_York'))
end = start + timedelta(hours=6)
print(start.isoformat() + " to " + end.isoformat())

# How many hours have elapsed?
print((end - start).total_seconds()/(60*60))

# What if we move to UTC?
print((end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds()/(60*60))


# Import datetime and tz
from datetime import datetime
from dateutil import tz

# Create starting date
dt = datetime(2000, 3, 29, tzinfo = tz.gettz('Europe/London'))

# Loop over the dates, replacing the year, and print the ISO timestamp
for y in range(2000, 2011):
    print(dt.replace(year=y).isoformat())


'''
Ending daylight saving time
eastern = tz.gettz('US/Eastern')

# 2017-11-05 01:00:00
first_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo = eastern)
tz.datetime_ambiguous(first_1am) -> in

True -> out

# 2017-11-05 01:00:00 again
second_1am = datetime(2017, 11, 5, 1, 0, 0, tzinfo = eastern)
second_1am = tz.enfold(second_1am)

(first_1am - second_1am).total_seconds() -> in

0.0 -> out

first_1am =  first_1am.astimezone(tz.UTC)
second_1am = second_1am.astimezone(tz.UTC)
(second_1am - first_1am).total_seconds() -> in

3600.0 -> out
'''

# Loop over trips
for trip in onebike_datetimes:
    # Rides with ambiguous start
    if tz.datetime_ambiguous(trip['start']):
        print("Ambiguous start at " + str(trip['start']))
    # Rides with ambiguous end
    if tz.datetime_ambiguous(trip['end']):
        print("Ambiguous end at " + str(trip['end']))


trip_durations = []
for trip in onebike_datetimes:
    # When the start is later than the end, set the fold to be 1
    if trip['start'] > trip['end']:
        trip['end'] = tz.enfold(trip['end'])
    # Convert to UTC
    start = trip['start'].astimezone(tz.UTC)
    end = trip['end'].astimezone(tz.UTC)

    # Subtract the difference
    trip_length_seconds = (end-start).total_seconds()
    trip_durations.append(trip_length_seconds)

# Take the shortest trip duration
print("Shortest trip: " + str(min(trip_durations)))


''' Easy and Powerful: Dates and Times in Pandas '''

'''
Reading date and time data in Pandas
A simple Pandas example
# Load Pandas
import pandas as pd
# Import W20529's rides in Q4 2017
rides = pd.read_csv('capital-onebike.csv')

# See our data
print(rides.head(3)) -> in

            Start date            End date                 Start station                          End station Bike number Member type
0  2017-10-01 15:23:25 2017-10-01 15:26:26          Glebe Rd & 11th St N        George Mason Dr & Wilson Blvd      W20529      Member
1  2017-10-01 15:42:57 2017-10-01 17:49:59 George Mason Dr & Wilson Blvd        George Mason Dr & Wilson Blvd      W20529      Casual
2  2017-10-02 06:37:10 2017-10-02 06:42:53 George Mason Dr & Wilson Blvd Ballston Metro / N Stuart & (th St N      W20529      Member  -> out

rides['Start date'] -> in

0 2017-10-01 15:23:25
1 2017-10-01 15:42:57
...
Name: Start date, Length:290, dtype: object -> out

rides.iloc[2]
Start date  2017-10-02 06:37:10  
End date    2017-10-02 06:42:53
...
Name: 1, dtype: object -> out

Loading datetimes with parse_dates
# Import W20529's rides in Q4 2017
rides = pd.read_csv('capital-onebike.csv', parse_dates = ['Start date', 'End date'])
# Or:
rides['Start date'] = pd.to_datetime(rides['Start date'], format = '%Y-%m-%d %H:%M:%S')

# Select Start date for row 2
rides['Start date'].iloc[2] -> in

Timestamp('2017-10-02 06:37:10') -> out

timezone-aware arithmetic
# Create a duration column
rides['Duration'] = rides['End date'] - rides['Start date']
# Print the first 5 rows
print(rides['Duration'].head(5)) -> in

0 0 days 00:03:01
1 0 days 02:07:02
2 0 days 00:05:43
3 0 days 00:21:18
4 0 days 00:21:17
Name: Duration, dtype: timedelta64[ns] -> out

rides['duration']\.dt.total_seconds()\.head(5) -> in

0 181.0
1 7622.0
2 343.0
3 1278.0
4 1277.0
Name: Duration, dtype: float64 -> out
'''

# Import pandas
import pandas as pd

# Load CSV into the rides variable
rides = pd.read_csv('capital-onebike.csv', parse_dates = ['Start date', 'End date'])

# Print the initial (0th) row
print(rides.iloc[0])

# Subtract the start date from the end date
ride_durations = rides['End date'] - rides['Start date']

# Convert the results to seconds
rides['Duration'] = ride_durations.dt.total_seconds()

print(rides['Duration'].head())


'''
Summarizing datetime data in Pandas
# Average time out of the dock
rides['Duration'].mean() -> in

Timedelta('0 days 00:19:38.931034482') -> out

# Total time out of the dock
rides['Duration'].sum() -> in

Timedelta('3 days 22:58:10') -> out

# Percent of time out of the dock
rides['Duration'].sum() / timedelta(days=91) -> in

0.04348417785917786 -> out

# Count how many time the bike started at each station
rides['Member type'].value_counts() -> in

Member 236
Casual 54
Name: Member type, dtype: int64 -> out

# Percent of rides by member
rides['Member type'].value_counts() / len(rides)-> in

Member 0.814
Casual 0.186
Name: Member type, dtype: float64 -> out

# Add duration (in seconds) column
rides['Duration seconds'] =  rides['Duration'].dt.yotal_seconds()
# Average duration per member type
rides.groupby('Member type')['Duration seconds'].mean() -> in

Member type
Casual 1994.667
Member 992.280
Name: Duration seconds, dtype: float64 -> out

# Average duration by month
rides.resample('M', on = 'Start date')['Duration seconds'].meqan() -> in

Start date
2017-10-31 1886.454
2017-11-30 854.175
2017-12-31 635.101
Freq: M, Name: Duration seconds, dtype: float64 -> out

# Size per group
rides.groupby('Member type').size() -> in

Member type
Casual 54
Member 236
dtype: int64 -> out

# First ride per group
rides.groupby('Member type').first() -> in

                Duration ...
Member type              ...
Casual          02:07:02 ...
Member          00:03:01 ...

Plotting
rides\
    .resamle('M', on = 'Start date')\
        ['Duration seconds']\
            .mean()\
                .plot()

# Change the plot resampling from month to days
rides\
    .resamle('D', on = 'Start date')\
        ['Duration seconds']\
            .mean()\
                .plot()
'''

# Create joyrides
joyrides = (rides['Start station'] == rides['End station'])

# Total number of joyrides
print("{} rides were joyrides".format(joyrides.sum()))

# Median of all rides
print("The median duration overall was {:.2f} seconds"\
    .format(rides['Duration'].median()))

# Median of joyrides
print("The median duration for joyrides was {:.2f} seconds"\
    .format(rides[joyrides]['Duration'].median()))


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to daily, take the size, plot the results
rides.resample('D', on = 'Start date')\
    .size()\
    .plot(ylim = [0, 15])

# Show the results
plt.show()


# Import matplotlib
import matplotlib.pyplot as plt

# Resample rides to monthly, take the size, plot the results
rides.resample('M', on = 'Start date')\
    .size()\
    .plot(ylim = [0, 150])

# Show the results
plt.show()


# Resample rides to be monthly on the basis of Start date
monthly_rides = rides.resample('M', on = 'Start date')['Member type']

# Take the ratio of the .value_counts() over the total number of rides
print(monthly_rides.value_counts() / monthly_rides.size())


# Group rides by member type, and resample to the month
grouped = rides.groupby('Member type')\
    .resample('M', on = 'Start date')

# Print the median duration for each group
print(grouped['Duration'].median())


'''
Additional datetime methods in Pandas
Timezones in Pandas
rides['Duration'].dt.total_seconds().min() -> in

-3346.0 -> out

rides['Start date'].head(3) -> in

0 2017-10-01 15:23:25
1 2017-10-01 15:42:57
2 2017-10-02 06:37:10
Name: Start date, dtype: datetime64[ns] -> out

rides['Start date'].head(3).dt.tz_localize('America/New_York') -> in

0 2017-10-01 15:23:25-04:00
1 2017-10-01 15:42:57-04:00
2 2017-10-02 06:37:10-04:00
Name: Start date, dtype: datetime64[ns, America/New_York] -> out

# try to set a timezone...
rides['Start date'] = rides['Start date']\.dt.tz_localize('America/New_York') -> in

pytz.exceptions.AmbiguousTimeError: Cannot infer dst time from '2017-11-05 01:56:50', try using the 'ambiguous' argument

# Handle ambiguous datetimes
rides['Start date'] = rides['Start date']\.dt.tz_localize('America/New_York', ambiguous='NaT')

rides['End date'] = rides['End date']\.dt.tz_localize('America/New_York', ambiguous='NaT')

# Re-calculate duration, ignoring bad row
rides['Duration'] = rides['End date'] -  rides['Start date']
# Find the minimum again
rides['Duration'].dt.total_seconds().min() -> in

116.0 -> out

# look at problematic row
rides.iloc[129] -> in

Duration                  NaT
Start date                NaT
End date                  NaT
Start station   6th & H St NE
End station     3rd & M St NE
Bike number            W20529
Member type            Member
Name: 129, dtype: object -> out

Othe datetime operations in Pandas
# Year of first three rows
rides['start date']\.head(3)\.dt.year -> in

0 2017
1 2017
2 2017
Name: Start date, dtype: int64 -> out

# See weekdays for first three rides
rides['Start date']\.head(3)\.dt.day_name() -> in

0 Sunday
1 Sunday
2 Monday
Name: Start date, dtype: object

# Shift the indexes forward one, padding with NaT
rides['End date'].shift(1).head(3) -> in

0                           NaT
1     2017-10-01 15:26:26-04:00
2     2017-10-01 17:49:59-04:00
Name: End date, dtype: datetime64[ns, America/New-York] -> out
'''

# Localize the Start date column to America/New_York
rides['Start date'] = rides['Start date'].dt.tz_localize('America/New_York', ambiguous='NaT')

# Print first value
print(rides['Start date'].iloc[0])

# Convert the Start date column to Europe/London
rides['Start date'] = rides['Start date'].dt.tz_convert('Europe/London')

# Print the new value
print(rides['Start date'].iloc[0])


# Add a column for the weekday of the start of the ride
rides['Ride start weekday'] = rides['Start date'].dt.day_name()

# Print the median trip time per weekday
print(rides.groupby('Ride start weekday')['Duration'].median())


# Shift the index of the end date up one; now subract it from the start date
rides['Time since'] = rides['Start date'] - (rides['End date'].shift(1))

# Move from a timedelta to a number of seconds, which is easier to work with
rides['Time since'] = rides['Time since'].dt.total_seconds()

# Resample to the month
monthly = rides.resample('M', on = 'Start date')

# Print the average hours between rides each month
print(monthly['Time since'].mean()/(60*60))