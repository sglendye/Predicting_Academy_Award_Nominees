def write_cast(dir_path):
    import pandas as pd
    import numpy as np
    import json
    data = pd.read_csv(dir_path+"\\"+'data'+"\\"+'tmdb_5000_credits.csv')

    # We should probably filter out empty data. I'm sure an academy award winner would have visible cast and crew anyways
    data = data[data.cast != '[]']
    data = data[data.crew != '[]']

    # Handling the cast columns now. Repeating mostly the same process for parsing the json column
    results = data.cast.apply(json.loads).apply(pd.io.json.json_normalize).pipe(lambda x: pd.concat(x.values))

    # Keeping only the first entry for each film
    movie_ids = results[results.index == 0]
    movie_ids = movie_ids.reset_index()

    # Clearing junk
    del movie_ids['index']

    # Merging our data to get a movie id for every film 
    movie_ids = movie_ids.join(data, how='outer')
    movie_ids = movie_ids[['credit_id', 'movie_id']]
    merged = results.merge(movie_ids, on='credit_id', how='left')

    # Filling in the gaps
    merged['movie_id'] = merged['movie_id'].ffill()

    # We don't have a jobs column here, but we do have an "order" column.
    # After a little bit of research this is definitely the order that the actors are listed in the credits, which *almost* universally begins with leads
    # Let's scoop the top 4 actors from every movie then. This will limit our data set, but still give us a chance to grab the main character, their two best friends, and a villain

    merged = merged[merged.order <= 3]

    # And now a pivot again, on (credits) order this time
    actors = merged.pivot_table(values='name', index='movie_id', columns = 'order', aggfunc='first')

    actors = actors.reset_index()
    actors = actors.rename(columns={actors.columns[1]: 'Actor 1', actors.columns[2]: 'Actor 2', actors.columns[3]: 'Actor 3', actors.columns[4]: 'Actor 4'})

    return actors
