import pandas as pd
import numpy as np
import sqlite3
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path+'\\'+'modules')
from create_database import write_tables
from db_query import pull_data
from random_forest_test import forest
from tuned_forest_test import tuned_forest

con = sqlite3.connect(dir_path+'\\'+'Academy_Awards_Data.db')
curs = con.cursor()
write_tables(con, curs, dir_path)

data = pull_data(con, curs)

#forest(data)
x = tuned_forest(data)
