import sys, os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)

from scripts.auxiliary.database_connector import *


def add_completion_time_column_to_simulation_df(cur, simulation_df):
    query = ("SELECT simulation_id, fk_topology_id, max_time "
             "FROM (simulation "
             "INNER JOIN"
             "(SELECT fk_simulation_id, max(completion_time) as max_time "
             "FROM (flowcompletiontime "
             "INNER JOIN (SELECT flow_id FROM flow WHERE arrival_time = 0) AS b on b.flow_id = flowcompletiontime.fk_flow_id) "
             "GROUP BY fk_simulation_id) AS a ON a.fk_simulation_id = simulation.simulation_id)")

    df = get_df_from_sql_query(cur, query)
    simulation_df = simulation_df.join(df.set_index('simulation_id')['max_time'], on='simulation_id', how='left')
    return simulation_df
