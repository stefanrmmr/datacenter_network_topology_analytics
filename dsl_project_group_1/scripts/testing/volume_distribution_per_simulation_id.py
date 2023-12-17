import sys, os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)
from scripts.auxiliary.database_connector import *
from scripts.visualization.plot_functions import *




def check_for_uniformity(simulation_id: int, cursor, show_output=False):

    # get data from flow table
    flow_df = get_sample_by_simulation_id(cursor, simulation_id, 'flow')

    # group by source node id sum over volumes
    groupby_source_node_summed = flow_df.groupby('source_node_id')['volume'].sum()
    groupby_dest_node_summed = flow_df.groupby('destination_node_id')['volume'].sum()
    groupby_combined = groupby_dest_node_summed + groupby_source_node_summed
    df_summed = pd.concat([groupby_source_node_summed, groupby_dest_node_summed, groupby_combined], axis=1)
    df_summed.columns = ['source', 'destination', 'combined']

    def prep_for_ks(x):
        res = []
        for i, n in zip(range(64), np.array((x/x.sum())*300).astype(int)):
            for _n in range(n):
                res.append(i)
        return res

    source_node_arr = prep_for_ks(groupby_source_node_summed)
    dest_node_arr = prep_for_ks(groupby_dest_node_summed)
    comb_node_arr = prep_for_ks(groupby_combined)


    # Komolgerov Smirnov Test for uniformity
    ks_source = stats.kstest(list(source_node_arr), 'uniform',
                             args=(min(source_node_arr), max(source_node_arr)))
    ks_dest = stats.kstest(dest_node_arr, 'uniform',
                           args=(min(dest_node_arr), max(dest_node_arr)))
    ks_combined = stats.kstest(comb_node_arr, 'uniform',
                               args=(min(comb_node_arr), max(comb_node_arr)))
    ks_list = [ks_source, ks_dest, ks_combined]
    if show_output:
        # plot distributions
        plot_volume_distribution_per_node_stacked(df_summed * (1.25e-10), ks_list, simulation_id, 'combined')
        plot_volume_distribution_per_node(groupby_source_node_summed.sort_values() * (1.25e-10), ks_source, simulation_id, 'source')
        plot_volume_distribution_per_node(groupby_dest_node_summed.sort_values() * (1.25e-10), ks_dest, simulation_id, 'destination')
        # plot_volume_distribution_per_node(groupby_combined.sort_values() * (1.25e-10), ks_combined, simulation_id, 'combined')
    x = pd.Series(dtype=float)
    x['probability of uniformity source nodes'] = ks_source.pvalue
    x['probability of uniformity dest nodes'] = ks_dest.pvalue
    x['probability of uniformity combined nodes'] = ks_combined.pvalue
    return x



# def plot_completion_time_vs_p_value():
cur = get_cursor_to_cerberus()
simulation_df = get_df_from_table(cur, 'simulation')
# get offline cases
query = """select flowgen_id, volume_per_node from flowgenerator where parameter = '{"arrival_time": 0.000000}'"""
offline_flowgen_df = get_df_from_sql_query(cur, query)
simulation_df = simulation_df.join(offline_flowgen_df.set_index('flowgen_id'), on='fk_flow_generator_id', how='inner')

# calculate p-values for uniform distribution
applied_df = simulation_df['simulation_id'].apply(check_for_uniformity, args=(cur,))
simulation_df = pd.concat([simulation_df, applied_df], axis='columns')
plot_histogram_p_values(simulation_df['probability of uniformity source nodes'], 'source')
plot_histogram_p_values(simulation_df['probability of uniformity dest nodes'], 'destination')
plot_histogram_p_values(simulation_df['probability of uniformity combined nodes'], 'combined')
check_for_uniformity(269, cur, show_output=True)
simulation_df.to_csv(os.path.join(workdir, 'data/offline_simulation_df_with_p_values.csv'))

# add max completion times for further analysis
from scripts.auxiliary.add_completion_times_to_simulation_df import *
simulation_df = add_completion_time_column_to_simulation_df(cur, simulation_df)
simulation_df.to_csv(os.path.join(workdir, 'data/offline_simulation_df_with_p_values_and_max_times.csv'))
