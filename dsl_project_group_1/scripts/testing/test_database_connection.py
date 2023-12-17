import sys, os

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)
from scripts.auxiliary.database_connector import *
cur = get_cursor_to_cerberus()
algorithm_df = get_df_from_table(cur, 'algorithm', 'algorithm_id, name')