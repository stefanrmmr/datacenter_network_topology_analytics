import sys, os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gaussian_kde

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)
from scripts.auxiliary.database_connector import *
from scripts.visualization.plot_functions import *

# get simulation dataframe with aggregated data
simulation_df = pd.read_csv(os.path.join(workdir, 'data/offline_simulation_df_with_p_values_and_max_times.csv'))


# calculate correlation between demand completion time and p-value from KS-Test for Uniformity
def correlation_max_time_vs_p_val(x, p_val_col_name):
    return x[p_val_col_name].corr(x['max_time'])


corr_source = simulation_df.groupby('volume_per_node').apply(
    lambda x: correlation_max_time_vs_p_val(x, 'probability of uniformity source nodes'))
corr_dest = simulation_df.groupby('volume_per_node').apply(
    lambda x: correlation_max_time_vs_p_val(x, 'probability of uniformity dest nodes'))
corr_combined = simulation_df.groupby('volume_per_node').apply(
    lambda x: correlation_max_time_vs_p_val(x, 'probability of uniformity combined nodes'))
total_corr = simulation_df['probability of uniformity source nodes'].corr(simulation_df['max_time'])

plot_scatter_single(corr_source.index, corr_source.values, 'volume per node', 'correlation p-val vs\n completion-time',
                    '', '', reg_line=True)
plot_scatter_single(corr_dest.index, corr_dest.values, 'volume per node', 'correlation p-val vs\n completion-time',
                    '', '', reg_line=True)
plot_scatter_single(corr_combined.index, corr_combined.values, 'volume per node',
                    'correlation p-val vs\n completion-time',
                    '', '', reg_line=True)





plot_scatter_single_w_colormap(simulation_df['probability of uniformity source nodes'],
                               simulation_df['max_time'] * (10 ** -6),
                               simulation_df['volume_per_node'] * (1.25e-10), 'p-value',
                               fr'demand completion time $[s]$', fr'volume per node $[GB]$', '',
                               plot_title=r'p_val_vs_completion_time')


plot_scatter_single(simulation_df['volume_per_node'] * (1.25e-10), simulation_df['probability of uniformity source nodes'], 'volume per node',
                        'p-val',
                        '', '', reg_line=False)
