# Importing necessary modules.
#     - os and sys are for easy handling of directory structure for included modules
#     - the create_database modules does as the name implies
#     - the db_query pulls a data set from the resultant database of the above mentioned module
#     - the forest and cat boost modules run their respective models

import pandas as pd
import numpy as np
import sqlite3
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'\\'+'modules')
from create_database import write_tables
from db_query import pull_data
from random_forest_tuned import tuned_forest
from cat_boost_v2 import cat_boost
from random_forest_default import default_forest

# Establishing a connection to our database, or if not found, creating the database for the user
con = sqlite3.connect(dir_path+'\\'+'Academy_Awards_Data.db')
curs = con.cursor()
write_tables(con, curs, dir_path)

# Storing our model results here
metrics = []

# Models with 200 iterations each
data = pull_data(con, curs)
print("-----------------------------------------------------------------------------")
print ("200 iterations")
iters = 200
metrics.append(default_forest(data, iters))
metrics.append(tuned_forest(data, iters))
metrics.append(cat_boost(data, iters))
print('''-----------------------------------------------------------------------------
      
''')

# Models with 500 iterations each
print("-----------------------------------------------------------------------------")
print ("500 iterations")
iters = 500
metrics.append(default_forest(data, iters))
metrics.append(tuned_forest(data, iters))
metrics.append(cat_boost(data, iters))
print('''-----------------------------------------------------------------------------
      
''')

# Models with 1000 iterations each
print("-----------------------------------------------------------------------------")
print ("1000 iterations")
iters = 1000
metrics.append(default_forest(data, iters))
metrics.append(tuned_forest(data, iters))
metrics.append(cat_boost(data, iters))
print('''-----------------------------------------------------------------------------
      
''')

# Storing it all as a dataframe and naming appropriately
df_metrics = pd.DataFrame(metrics)
df_metrics.rename(columns={df_metrics.columns[0]: "Total Percent", df_metrics.columns[1]: "Nomination Percent", df_metrics.columns[2]: "Non-Nomination Percent", df_metrics.columns[3]: "Completion Time"}, inplace = True)

# Adding a column to label our models
models = ['default 200 iterations', 'tuned 200 iterations', 'cat 200 iterations','default 500 iterations', 'tuned 500 iterations', 'cat 500 iterations','default 1000 iterations', 'tuned 1000 iterations', 'cat 1000 iterations' ]
df_metrics.insert(0, "Model", models, True)

