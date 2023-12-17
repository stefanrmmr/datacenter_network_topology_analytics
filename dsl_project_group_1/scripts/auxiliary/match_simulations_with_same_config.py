import sys, os
import pandas as pd
import numpy as np

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)
from scripts.auxiliary.database_connector import *

def get_unique_config_id_and_seed_column(cur, simulation_df):
    """

    Args:
        cur: MariaDB cursor
        simulation_df: Dataframe containing simulations

    Returns: simulation_df with added column for simulation_id group disregarding with merely differing seeds in the
    flow generator

    """
    flowgen_df = get_df_from_table(cur, 'flowgenerator')
    cols = list(flowgen_df.columns)
    cols = [x for x in cols if (x not in ['flowgen_id', 'seed'])]
    flowgen_df['unique_configs'] = flowgen_df.replace(np.nan, -1).groupby(cols).ngroup()
    simulation_df = simulation_df.join(flowgen_df[['flowgen_id', 'unique_configs', 'seed']].set_index('flowgen_id'),
                                       on='fk_flow_generator_id', how='inner')
    simulation_df['sim_id_disregarding_seed'] = simulation_df.replace(np.nan, -1).groupby(['fk_topology_id', 'fk_algorithm_id', 'unique_configs']).ngroup()
    return simulation_df
