# COMPARISON of performance (demand flow completion time) for different algorithms on the ROTOR NET topology
# group members responsible: Stefan Rummer (03709476)

from scripts.auxiliary.database_connector import get_cursor_to_cerberus
from scripts.auxiliary.database_connector import get_df_from_sql_query
from scripts.auxiliary.database_connector import get_df_from_table
from scripts.visualization.plot_functions import plot_histogram_fct_log
from scripts.visualization.plot_functions import plot_scatter_single
from scripts.visualization.plot_functions import plot_histogram_ttest
from scripts.visualization.plot_functions import plot_algorithm_performance
from scripts.visualization.plot_functions import plot_algorithm_performance_ranked
from scripts.visualization.plot_functions import plot_algorithm_performance_overlap
from scripts.visualization.plot_functions import plot_algorithm_performance_bestof

from icecream import ic
import pandas as pd
import numpy as np
import sys
import os

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# os.path.dirname(__file__) gets you the directory that script is in.
sys.path.append(workdir)  # append path of folder dsl_project_group_1

mdb_cursor = get_cursor_to_cerberus()
topology_df = get_df_from_table(mdb_cursor, 'topology')
algorithm_df = get_df_from_table(mdb_cursor, 'algorithm')
simulation_df = get_df_from_table(mdb_cursor, 'simulation')
flowgenerator_df = get_df_from_table(mdb_cursor, 'flowgenerator')


def demand_compltime_ttest(dataframe, volume, color, cv=1.753):

    # 5 % significance level, alpha = 0.5
    # for len(dataframe) = 16, dof = 15
    # right side t-test, CV = 1,753
    # RIGHT SIDE T-HYPOTHESIS TEST

    mean = dataframe[str(volume)].mean()
    n = len(dataframe)  # < 30, sufficient
    value = dataframe[str(volume)].min()
    std = dataframe[str(volume)].std()
    t = (mean-value)/(std/np.sqrt(n))
    border = ((cv*(std/np.sqrt(n))) - mean)*(-1)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("[HYPOTHESIS TEST for algorithm performance significance]")
    print(f"critical value = {cv} for 5% significance level, n = {n} values.")
    print(f"mean = {mean}, standard deviation = {std}")
    print(f"t- value calculated = {t}")
    print(f" ➔ all demand compl. times below {border} significantly outperform! ")
    if t > cv:
        print(f" ➔ t > cv, REJECT NULL HYPOTHESIS")
    if t <= cv:
        print(f" ➔ t <= cv, ACCEPT NULL HYPOTHESIS")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    plot_histogram_ttest(dataframe[str(volume)], 40, border,
                         'demand completion time [s]',
                         'distribution', f"volume {volume/8000000000} [GB]", color)
    if t > cv:
        return t, cv, border, True  # reject H0
    if t <= cv:
        return t, cv, border, False  # accept H0


def all_algorithms_for_topology(topology_id):
    """
    Args:
        topology_id: representing a network structure
    Returns:
        list of tuples containing existing algo_topo_pairs
    """
    algo_options = simulation_df['fk_algorithm_id'].unique()
    algo_options = list(filter(None, algo_options))
    # find unique algorithm values and remove None values
    algo_available = []  # list for suitable algorithms
    for algorithm_id in algo_options:
        simulation_red_df = simulation_df
        simulation_red_df = simulation_red_df[simulation_red_df['fk_algorithm_id'] == algorithm_id]
        simulation_red_df = simulation_red_df[simulation_red_df['fk_topology_id'] == topology_id]
        if len(simulation_red_df) != 0:  # evaluate all possible algorithm-topology combinations
            algo_available.append(algorithm_id)

    return algo_available


def algorithm_performance(algorithm_id, topology_id, seed, volume, enable_plot=False):
    """
    Args:
        algorithm_id: the selected network algorithm
        topology_id: the selected network topology
        seed: the specific seed for a simulation
        volume: volume per node for a simulation
        enable_plot: display plot?
    Returns:
        fct_min_value: minimum flow completion time
        fct_max_value: demand flow completion time
    """
    algorithm_name_index = algorithm_df[algorithm_df['algorithm_id'] == algorithm_id].index.values
    algorithm_name = str(algorithm_df.iloc[algorithm_name_index]['name'])[5: -26]    # get name of algorithm
    topology_name_index = topology_df[topology_df['topology_id'] == topology_id].index.values
    topology_name = str(topology_df.iloc[topology_name_index]['name'])[5: -26]       # get name of topology

    flowgen_offline = flowgenerator_df[flowgenerator_df['parameter'] == "{\"arrival_time\": 0.000000}"]
    flowgen_offline_ids = flowgen_offline["flowgen_id"].tolist()
    # select the offline flows generated where "parameter" is not <null>
    flowgen_volume = flowgenerator_df[flowgenerator_df['volume_per_node'] == volume]
    flowgen_volume_ids = flowgen_volume["flowgen_id"].tolist()
    # select the flows where "volume_per_node" equals the one selected
    flowgen_seed = flowgenerator_df[flowgenerator_df['seed'] == seed]
    flowgen_seed_ids = flowgen_seed["flowgen_id"].tolist()
    # select the flows where "seed" equals the one selected

    intersection_set = set(flowgen_offline_ids).intersection(set(flowgen_volume_ids), set(flowgen_seed_ids))
    intersection_flowgen_ids = list(intersection_set)  # find flow generator ids that occur in all three lists

    simulation_red_df = simulation_df
    pos = len(simulation_red_df) - 1
    for index in range(len(simulation_red_df)):
        if simulation_red_df.iloc[pos]['fk_flow_generator_id'] not in intersection_flowgen_ids:
            sim_id = simulation_red_df.iloc[pos]['simulation_id']
            # get the unique simulation id for this row element
            index_value = simulation_red_df[simulation_red_df['simulation_id'] == int(sim_id)].index.values
            simulation_red_df = simulation_red_df.drop(labels=index_value, axis=0)
            pos -= 1  # drop row if it is not part of the intersection_flowgen_ids list
        else:         # just skip row if it is part of the intersection_flowgen_ids list
            pos -= 1
    # FILTER FLOWGENERATOR for volume, seed, only select offline simulations

    simulation_red_df = simulation_red_df[simulation_red_df['fk_algorithm_id'] == algorithm_id]
    simulation_red_df = simulation_red_df[simulation_red_df['fk_topology_id'] == topology_id]
    # FILTER SIMULATION TABLE for algorithm_id and topology_id

    simulation_sample_ids = str(list(simulation_red_df['simulation_id']))[1:-1]

    query_fct = f"""select * from flowcompletiontime where fk_simulation_id in ({simulation_sample_ids})"""
    fct_sample_df = get_df_from_sql_query(mdb_cursor, query_fct)  # contains the flowcompletiontimes

    fct_max_value = fct_sample_df['completion_time'].max()
    fct_min_value = fct_sample_df['completion_time'].min()

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"[VOLUME_per_node] = {volume}")
    print(f"[SEED USED] id = {seed} (available for this config)")
    print(f"[TOPOLOGY] id = {topology_id} {topology_name}")
    print(f"[ALGORITHM] id = {algorithm_id} {algorithm_name}")
    print(f"➔ MAX flow completion time: {fct_max_value} (demand completion time)")
    print(f"➔ MIN flow completion time: {fct_min_value} ")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if enable_plot:
        # plot_violin_extended(fct_sample_df['completion_time'], 'flow completion times', 'flows', 'plot title')
        plot_histogram_fct_log(fct_sample_df['completion_time'], 80,
                                  'flow completion time [s]',  # the plot scales micro sec to sec
                                  'distribution [log scale]',
                                  "demand compl.t." + f' = {"{:.2e}".format(fct_max_value)} $\mu$s\n'
                                  f"volume per node" + f' = {"{:.2e}".format(volume)} bit\n'
                                  "algor. id" + f' = {algorithm_id}, '
                                  "topology id" + f' = {topology_id} \n'
                                  f'flows data for seed = {seed}',
                                  f'AlgoAnalysis{algorithm_id}-{topology_id}-{seed}')

    return fct_max_value


def dataframe_dctimes(algorithm_ids, topology_id, seeds, volumes):

    # create dataframe with cols=volume_options, rows=algorithm_options
    # row names: algorithm_ids (index value), column names: volume values

    row = [None] * len(volumes)
    df_dct = pd.DataFrame([row], columns=volumes, index=algorithm_ids)

    for algo in algorithm_ids:
        for vol in volumes:
            try:
                demand_fct = []
                for seed in seeds:
                    try:  # in case the seeds list contains more than one seed, average across demand comp.times
                        demand_fct.append(algorithm_performance(algo, topology_id, int(seed), vol, enable_plot=False))
                    except:
                        pass
                demand_fct = sum(demand_fct)/len(demand_fct)  # calculate mean demand completion time for seeds
                df_dct.at[algo, vol] = demand_fct
            except:
                pass

    return df_dct


flowgen_volume_options = flowgenerator_df['volume_per_node'].unique()
flowgen_volume_options = list(filter(None, flowgen_volume_options))
flowgen_volume_options = [int(x) for x in flowgen_volume_options]
# find unique float values for volumes_per_node and remove None
flowgen_seed_options = flowgenerator_df['seed'].unique()
flowgen_seed_options = [int(x) for x in flowgen_seed_options if str(x) != 'nan']
# find unique seeds values and remove NaN values, convert to integer values
simulation_topo_options = simulation_df['fk_topology_id'].unique()
simulation_topo_options = list(filter(None, simulation_topo_options))
# find unique topologies values and remove None values
simulation_algo_options = simulation_df['fk_algorithm_id'].unique()
simulation_algo_options = list(filter(None, simulation_algo_options))
# find unique algorithm values and remove None values
print(f"➔ Number of VOLUME OPTIONS      available in dataset = {len(flowgen_volume_options)}")
print(f"➔ Number of SEED OPTIONS        available in dataset = {len(flowgen_seed_options)}")
print(f"➔ Number of TOPOLOGY OPTIONS    available in dataset = {len(simulation_topo_options)}")
print(f"➔ Number of ALGORITHMS OPTIONS  available in dataset = {len(simulation_algo_options)}")

algorithms_select = all_algorithms_for_topology(1)
print(f"\n➔ {len(algorithms_select)} Possible Algorithms for Topology:"
      f"\n   ROTOR NET, with algorithm_id = 1\n")
"""for algo in range(len(algorithms_select)):
    print(f"topology 1, algorithm {algorithms_select[algo]}")"""

algorithms_select_sub = []
for x in algorithms_select:
    if x not in [15, 20, 21]:
        algorithms_select_sub.append(x)
# create subset without out liar algorithms

# after initial analysis (df_dct) one can see that ONLY the first four entries of flowgen_volume_options
# actually yield demand completion times, based on the validity of SQL queries trying to access the MariaDb
volumes_select = [8000000000, 20000000000, 32000000000, 64000000000]

"""df_dct = dataframe_dctimes(algorithms_select, 1, flowgen_seed_options, volumes_select)
# dataframe that contains all demand completion times, SAVE in CSV to be used later on
df_dct.to_csv(f"{workdir}/csv_output/dataframe_dct_FINAL.csv")
ic(df_dct)  # display specifications"""

df_dct = pd.read_csv(f"{workdir}/csv_output/dataframe_dct_FINAL.csv")
df_dct.rename(columns={'Unnamed: 0': 'algo_id'}, inplace=True)
df_dct.sort_values(['algo_id'], inplace=True)
# ic(df_dct)

algo_id_list = df_dct['algo_id'].tolist()
algo_id_list = [int(x) for x in algo_id_list]
# order list of algorithm ids, int formatted

"""df_dct_sub = dataframe_dctimes(algorithms_select_sub, 1, flowgen_seed_options, volumes_select)
# dataframe containing identical values like df_dct BUT without outliar algorithms
# use this dataframe for analysis of regression line
# use this dataframe for hypothesis T-TESTs"""

df_dct_sub = df_dct.copy()
df_dct_sub = df_dct_sub[df_dct_sub.algo_id != 15]
df_dct_sub = df_dct_sub[df_dct_sub.algo_id != 20]
df_dct_sub = df_dct_sub[df_dct_sub.algo_id != 21]
ic(df_dct_sub)

# HYPOTHESIS TESTS: whether or not the best algorithm actually performs suitable
demand_compltime_ttest(df_dct_sub, volumes_select[0], 'tum_blue')
demand_compltime_ttest(df_dct_sub, volumes_select[1], 'orange')
demand_compltime_ttest(df_dct_sub, volumes_select[2], 'darkslategrey')
demand_compltime_ttest(df_dct_sub, volumes_select[3], 'teal')


"""# CALCULATE regression line plot without out liar algorithms
list_vol = []
list_dct = []
for vol in volumes_select:
    column_list = list(df_dct_sub[vol])
    for index in range(len(column_list)):
        list_vol.append(vol)
        list_dct.append(column_list[index])
list_dct = [float(x/1000000) for x in list_dct]   # scale to seconds
list_vol = [x/8000000000 for x in list_vol]  # scale to GigaByte
plot_scatter_single(np.array(list_vol), np.array(list_dct), 'volumes [GB]',
                    'demand compl. time [s]', 'datapoints', True)"""


# commented uses of the functions provided in order to generate plots

"""for vol in volumes_select:
    dct_list = df_dct[str(vol)].tolist()
    plot_algorithm_performance(dct_list, algo_id_list, "algorithm ids", "demand completion time",
                               f'volume per node= {"{:.2e}".format(vol)} bit')
    # plot algorithm performances for every single volume"""


"""for vol in volumes_select:
    dct_list = df_dct[str(vol)].tolist()
    plot_algorithm_performance_ranked(dct_list, algo_id_list, "algorithm ids (ranked)", "demand completion time",
                               f'volume per node= {"{:.2e}".format(vol)} bit')
    # plot algorithm performances for every single volume"""

"""plot_algorithm_performance_overlap(df_dct[str(volumes_select[0])].tolist(),
                                   df_dct[str(volumes_select[1])].tolist(),
                                   df_dct[str(volumes_select[2])].tolist(),
                                   df_dct[str(volumes_select[3])].tolist(),
                                   algo_id_list,
                                   "algorithm ids",
                                   "demand compl. time [s]",
                                   "compare dct for volumes")"""

"""plot_algorithm_performance_bestof(df_dct[str(volumes_select[0])].tolist(),
                                   df_dct[str(volumes_select[1])].tolist(),
                                   df_dct[str(volumes_select[2])].tolist(),
                                   df_dct[str(volumes_select[3])].tolist(),
                                   algo_id_list,
                                   "algorithm ids [ranked]",
                                   "demand compl. time [s]",
                                   "top 5 algorithms per volume")"""



# TODO im text zu rotornet auf file in git verweisen bzw den code welcher genutzt wurde um den part zu generieren

# TODO erwähnen für den case source by source weil arrival time == 0.000 beim filtern
# TODO was ist source by source ?

# TODO der seed ist wirklich nur genutzt um in numpy random states zu initialisieren
# TODO gleicher seed --> paired data, ungleicher seed und average über seeds --> stichprobentest vergleich

# TODO T-TEST whether or not there is a statistical significant difference
#  of the model performances for different seeds (seperate t test for each volume)
# TODO normal t test or paired? (if paired: paired(equal var) or pooled approach)?


# TODO für paired data muss die flow generator id gleich sein, aka der seed muss der gleiche sein,
#  andere var alle fix, bis auf zwei versch algorithms

# TODO wenn es zwei unterschiedliche seeds sind aber sonst legiter vergleich dann ist es ein stichproben vergleich

# TODO ranking der algorithmen performance für jedes volumen (bar plot ordered by
