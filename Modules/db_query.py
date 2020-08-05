# Merging all of the data sets. I use a case statement here to binarize each film as either being nominated or not
# The sub query on the join merges the tables on the "match" columns created in the previous modules, which just normalizes the names of 
# the movie titles, since the awards data set did not come with a movie_id. I noticed that I wasn't getting complete matches, so applying 
# some regular expressions to strip special characters and make everything the same case further back seemed to do the trick.


def pull_data(con, curs):
    import pandas as pd
    data = pd.read_sql('''

    SELECT   a.movie_id
           , a.[Actor 1]
           , a.[Actor 2]
           , a.[Actor 3]
           , a.[Actor 4]
           , b.budget
           , b.revenue
           , b.runtime
           , b.title
           , c.[Associate Producer]
           , c.[CG Animator]
           , c.[Casting]
           , c.[Cinematography]
           , c.[Choreographer]
           , c.[Costume Design]
           , c.[Creator]
           , c.[Director]
           , c.[Music]
           , c.[Post-Production Manager]
           , c.[Producer]
           , c.[Screenplay]
           , c.[Set Designer]
           , c.[Sound Director]
           , c.[Story]
           , CAST(
                   CASE
                       WHEN b.nominee IS NOT NULL
                           THEN 1
                       ELSE 0
                    END AS bit) AS nomination
    FROM tmdb_cast AS a
    INNER JOIN
    (
    SELECT   movie_list.movie_id
           , movie_list.budget
           , movie_list.revenue
           , movie_list.runtime
           , movie_list.title
           , academy_awards.film AS nominee
    FROM movie_list
    LEFT JOIN academy_awards
    ON movie_list.match = academy_awards.match
    ) AS b
    ON a.movie_id = b.movie_id
    INNER JOIN tmdb_crew AS c
    ON b.movie_id = c.movie_id
    ''', con)
    
    return data
