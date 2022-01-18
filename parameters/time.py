# According to NASA
DAYS_PER_YEAR = 365.2422

# Total minutes per day
MINUTES_PER_DAY = int(24 * 60)

# Total minutes in a trading year (24/7)
MINUTES_PER_YEAR = int(DAYS_PER_YEAR * MINUTES_PER_DAY)

# Total seconds in a trading year (24/7)
SECONDS_PER_YEAR = int(MINUTES_PER_YEAR * 60.0)

# Total seconds per day
SECONDS_PER_DAY = int(MINUTES_PER_DAY * 60)

# 1 millisecond as a year fraction
EPSILON_TTM = 0.001 / SECONDS_PER_YEAR
