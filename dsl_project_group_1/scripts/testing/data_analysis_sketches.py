from scripts.auxiliary.database_connector import *
import scripts.visualization.plot_functions as plot_funct

import sys, os
import pandas as pd
import numpy as np
from icecream import ic
import statistics
import matplotlib.pyplot as plt

# os.path.dirname(__file__) gets you the directory that script is in.
workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)  # append path of folder dsl_project_group_1



mdb_cursor = get_cursor_to_cerberus()

algo_id = 1     # DEFINE WHICH ALGORITHM
topo_id = 1     # DEFINE WHICH TOPOLOGY

algorithm_df = get_df_from_table(mdb_cursor, 'algorithm', 'algorithm_id, name')
simulation_df = get_df_from_table(mdb_cursor, 'simulation')
topology_df = get_df_from_table(mdb_cursor, 'topology')


sample = simulation_df[simulation_df['fk_algorithm_id'] == algo_id]
unique_topology_ids = sample['fk_topology_id'].unique()
sample = sample[sample['fk_topology_id'] == topo_id]
sample_ids = str(list(sample['simulation_id']))[1:-1]





query = f"""
select * from flowcompletiontime where fk_simulation_id in ({sample_ids})
"""
flow_completion_sample = get_df_from_sql_query(mdb_cursor, query)
flow_ids = flow_completion_sample['fk_flow_id']
# sql expects (1, 2) -> convert series to list -> list to str -> remove [] brackets -> add () brackets
list_of_flow_ids = str(list(flow_ids))[1:-1]

query = f"""
select * from flow where flow_id IN ({list_of_flow_ids})
"""
flow_sample = get_df_from_sql_query(mdb_cursor, query)

plot_funct.plot_violin_extended(flow_sample['volume'], '', '', '')
# plot_funct.plot_violin_extended(flow_completion_sample['completion_time'], '', '', '')
group_source = flow_sample.groupby('source_node_id').count().sort_index()
group_source['flow_id'].plot()
plt.show()
group_dest = flow_sample.groupby('destination_node_id').count()
group_dest['flow_id'].plot()
plt.show()
plot_funct.plot_scatter_single(group_source['flow_id'], group_dest['flow_id'], '', '', '', '', False)

