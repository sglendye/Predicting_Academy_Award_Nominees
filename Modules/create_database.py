# Writing our tables from the three different data files. This module doesa quick check on the database to see if the tables already exist.
# If they don't, it writes a new one for each file and adds an index to them to help speed up the merges in the second and third exercises.
# If a table does exist, it just passes over it and moves to the next one.
# This check is to allow the user to re-run the full procedure without it returning errors on the first step.

def write_tables(con, curs, dir_path):
    import pandas as pd
    import sqlite3
    import sys
    import os
    sys.path.append(dir_path+'\\'+'modules')
    from clean_awards import write_awards
    from clean_crew import write_crew
    from clean_cast import write_cast
    from clean_movies import write_movies

    count = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='movie_list'", con)
    if len(count.index) == 0:
        data = write_movies(dir_path)
        data.to_sql('movie_list', con, index=False)
        con.commit()
        sql = ("CREATE INDEX idx_list_id ON movie_list (movie_id);")
        curs.execute(sql)
        con.commit()
        sql = ("CREATE INDEX idx_list_match ON movie_list (match);")
        curs.execute(sql)
        con.commit()
    else:
        pass

    count = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='academy_awards'", con)
    if len(count.index) == 0:
        data = write_awards(dir_path)
        data.to_sql('academy_awards', con, index=False)
        con.commit()
        sql = ("CREATE INDEX idx_awards_match ON academy_awards (match);")
        curs.execute(sql)
        con.commit()
    else:
        pass
    
    count = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='tmdb_cast'", con)
    if len(count.index) == 0:
        data = write_cast(dir_path)
        data.to_sql('tmdb_cast', con, index=False)
        con.commit()
        sql = ("CREATE INDEX idx_cast_id ON tmdb_cast (movie_id);")
        curs.execute(sql)
        con.commit()
    else:
        pass

    count = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='tmdb_crew'", con)
    if len(count.index) == 0:
        data = write_crew(dir_path)
        data.to_sql('tmdb_crew', con, index=False)
        con.commit()
        sql = ("CREATE INDEX idx_crew_id ON tmdb_crew (movie_id);")
        curs.execute(sql)
        con.commit()
    else:
        pass
    
