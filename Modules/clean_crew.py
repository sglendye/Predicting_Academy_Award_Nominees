def write_crew(dir_path):
    import pandas as pd
    import numpy as np
    import json
    data = pd.read_csv(dir_path+"\\"+'data'+"\\"+'tmdb_5000_credits.csv')

    # We should probably filter out empty data. I'm sure an academy award winner would have visible cast and crew anyways
    data = data[data.cast != '[]']
    data = data[data.crew != '[]']

    # Parsing our data that is stored as a json
    results = data.crew.apply(json.loads) \
           .apply(pd.io.json.json_normalize)\
           .pipe(lambda x: pd.concat(x.values))

    # Well, we can see that it melted the whole thing down into a few columns. We'll need a movie id to match these up with our films and other data
    # We can also see that each movie received its own index, which was appended by our pipe
    # I'm going to cheat a little here and merge our data back with a trcik matching indices
    # Under the assumption that every film must have at least a zero index, so we can filter on that to generate a dataframe with the same number of films as our original data with our parsed data
    movie_ids = results[results.index == 0]
    movie_ids = movie_ids.reset_index()
    data = data.reset_index()

    # Paring off some junk
    del movie_ids['index']
    del data['index']

    # Merging all of our data back together
    movie_ids = movie_ids.join(data, how='outer')
    movie_ids = movie_ids[['credit_id', 'movie_id']]
    merged = results.merge(movie_ids, on='credit_id', how='left')

    # Using pandas ffill to assign a movie to every entry in our merged dataframe
    merged['movie_id'] = merged['movie_id'].ffill()

    # Checking to see how many different jobs there are.
    # This is *without* duplicates, wow. Probably a lot more than we'll need.
    pd.unique(merged['job'])

    # Freeing up RAM as we go along
    del movie_ids

    # Pivoting all of our data so that each position is a unique column for each film
    positions = merged.pivot_table(values='name', index='movie_id', columns = 'job', aggfunc='first')

    # 418 columns is a bit outrageous to work with in this small project. Perhaps with more resources they mey come in handy, but let's try this with only a few key positions

    # Here's a few

    positions = positions.reset_index()
    jobs2keep = ['movie_id', 'Associate Producer', 'CG Animator', 'Casting', 'Cinematography', 'Choreographer', 'Costume Design', 'Creator', 'Director', 'Music', 'Post-Production Manager', 'Producer', 'Screenplay', 'Set Designer', 'Sound Director', 'Story']

#***************************************
    positions = positions[jobs2keep]

    return positions
