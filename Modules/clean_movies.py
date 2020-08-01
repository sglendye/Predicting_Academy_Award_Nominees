def write_movies(dir_path):
    import pandas as pd
    import numpy as np
    import re

    data = pd.read_csv(dir_path+"\\"+'data'+"\\"+'tmdb_5000_movies.csv')

    data = data[['budget', 'id', 'release_date', 'revenue', 'runtime', 'title']]

    # We'll nix entries with no budget. There's some good films in here, but there's quite a lot of missing data for these. Most of them, that I can tell from a quick eyeball, aren't likely candidates for an award either
    data = data[data.budget != 0]

    # This might come in handy later
    data = data.rename(columns={'id': 'movie_id'})

    # Before writing our table, we may want to normalize our data. The budgets and revenue may look like ordinary large numbers,
    # but movie fans looking at this will know that the data is in its original numbers and adjusted for inflation.
    # For instance, our highest grossing film of all time, Gone with the Wind, "only" has a revenue of $400,176,459 listed here.

    print(data[data.movie_id == 770])

    # Adjusted for inflation, it should actually be several billion dollars

    # Data from US Bureau of Labor Statistics
    # https://data.bls.gov/timeseries/CUUR0000SA0

    inflation = pd.read_csv(dir_path+"\\"+'data'+"\\"+'inflation_rates.csv')
    #inflation = inflation.set_index('Year')
    #y = inflation.to_dict('dict')


    # Swapping out for just the year to match our inflation data

    data['release_date'] = pd.DatetimeIndex(data['release_date']).year

    # Converting our inflation rate to convert to 2020 dollar values

    inflation = inflation.set_index('Year')
    inflation['CPI'] = (inflation.loc[2020, 'CPI'])/(inflation['CPI'])

    # Merging our inflation rates in
    data = data.merge(inflation, how='left', left_on = 'release_date', right_index = True)

    # Adjusting our values
    data['budget'] = data['budget']*data['CPI']
    data['revenue'] = data['revenue']*data['CPI']
    del data['CPI']

    # Let's check again
    print(data[data.movie_id == 770])

    # Well that's a lot of money... actually a lot more than expected!
    # Using a simple year to year CPI has the unfortunate downside of not capturing films
    # whose revenue is spread out over years (or EIGHT different re-releases in 1947, 1954, 1961, 1967, 1971, 1974, 1989, and 1998).
    # Oh well, we'll work with this for now and see how it performs

    # Storing this for later, we'll need some normalized text to match on because our Academy Awards titles are not quite identical

    def normalize(x):
        x = str(x)
        x = x.lower()
        x = re.sub(' ','', x)
        x = re.sub('[^A-Za-z0-9]+', '', x)
        return x

    data['match'] = data['title'].apply(lambda x: normalize(x))

    return data
