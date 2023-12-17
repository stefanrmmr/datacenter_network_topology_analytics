import sys, os
import pandas as pd
import scipy.stats
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import rc
import seaborn as sns

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)
from scripts.auxiliary.database_connector import *
from scripts.visualization.plot_functions import *
from scripts.auxiliary.match_simulations_with_same_config import *

cur = get_cursor_to_cerberus()
simulation_df = get_df_from_table(cur, 'simulation')
# get offline simulations

query = """select flowgen_id, volume_per_node from flowgenerator where parameter = '{"arrival_time": 0.000000}'"""
offline_flowgen_df = get_df_from_sql_query(cur, query)
simulation_df = simulation_df.join(offline_flowgen_df.set_index('flowgen_id'), on='fk_flow_generator_id', how='inner')


def filter_simulations_by_topology(cur, sim_df, topology):
    query = f"select topology_id from topology where name = '{topology}'"
    topology_ids = get_df_from_sql_query(cur, query)['topology_id']
    return sim_df[sim_df['fk_topology_id'].isin(topology_ids)]


# Rotornet_simulations
rot_simulations_df = filter_simulations_by_topology(cur, simulation_df, 'RotorNetTopologyConfiguration')

# Cerberus_simulations
cerb_simulations_df = filter_simulations_by_topology(cur, simulation_df, 'CerberusTopologyConfiguration')

# Expandernet_simulations
exp_simulations_df = filter_simulations_by_topology(cur, simulation_df, 'ExpanderNetTopologyConfiguration')


def interpolate_total_rate(cur, sim_id, t, show_plot=True):
    df = get_sample_by_simulation_id(cur, sim_id, 'totalflowrate')[['time', 'total_rate']]
    t_new = t[t <= df['time'].max()]
    f = interp1d(np.array(df['time']).astype(float), np.array(df['total_rate']).astype(float))
    y = np.array(interp1d(t_new, np.array(f(t_new)).clip(0))(t_new))
    # y = np.array(f(t_new)).clip(0)
    if show_plot:
        plt.plot(t_new, y)
        plt.xlabel('time [us]')
        plt.ylabel('total_rate')
        plt.show()
    y = np.array(y).clip(0)
    y.resize(len(t))
    return y, df['time'].max()


def calculate_overall_total_rate(cur, sim_ids, t, show_plot=True):
    if not list(sim_ids):
        return None, None, None, None
    accumulated_rates = np.zeros((len(t), len(sim_ids)))
    max_times = np.zeros((1, len(sim_ids)))
    for i, sim_id in enumerate(sim_ids):
        try:
            accumulated_rates[:, i], max_times[0, i] = interpolate_total_rate(cur, sim_id, t, show_plot=False)
        except ValueError:
            print(f"no data for {sim_id}")
            continue
        print(f'{i + 1}/{len(sim_ids)}')

    max_rates = np.array(interp1d(t, np.max(accumulated_rates, axis=1))(t))
    mean_rates = np.array(interp1d(t, np.mean(accumulated_rates, axis=1))(t))
    min_rates = np.array(interp1d(t, np.min(accumulated_rates, axis=1))(t))
    if show_plot:
        # plt.plot(t, max_rates, label='max')
        p = plt.plot(t, mean_rates, label='mean')
        # plt.plot(t, min_rates, label='min')
        plt.fill_between(t, min_rates, max_rates, color=p[0].get_color(), alpha=0.2)
        plt.xlabel('time [us]')
        plt.ylabel('total rate')
        plt.legend()
        plt.show()
    return max_times, max_rates, mean_rates, min_rates


def compare_total_rate_vs_topology(cur, volumes, t, df_rot, df_cerb, df_exp):
    volumes = list(volumes)
    result_df = pd.DataFrame()
    for volume in volumes:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        time_dict = {}
        for df, name, y_offset in zip([df_rot, df_cerb, df_exp], ['RotorNet', 'Cerberus', 'ExpanderNet'], [30, 40, 50]):
            sim_ids = df[df['volume_per_node'] == volume]['simulation_id']
            max_times, max_rates, mean_rates, min_rates = calculate_overall_total_rate(cur, sim_ids, t,
                                                                                       show_plot=False)
            avg_max_time = max_times.mean()
            time_dict[name] = avg_max_time
            if mean_rates is not None:
                # plt.plot(t, max_rates, label='max')
                p = ax.plot(t, mean_rates, label=name)
                # plt.plot(t, min_rates, label='min')
                ax.fill_between(t, min_rates, max_rates, color=p[-1].get_color(), alpha=0.2)
                ax.annotate(f'{name} avg end time', xy=(avg_max_time, 0), xycoords='data',
                            xytext=(0, y_offset), textcoords='offset points',
                            arrowprops=dict(arrowstyle="->", ))
        ax.set_xlabel(r'Time in $\mu s$')
        ax.set_ylabel(r'Total Rate in $MBit/s$')
        ax.set_title(f'volume per node: ${float(volume):.0}$')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(workdir, f'plots/vol_{float(volume):.0}.png'))
        plt.show()
        s = pd.Series(time_dict)
        s.name = volume
        print(s)
        result_df = result_df.append(s)
    return result_df


def compare_simulations_within_one_topology(cur, volumes, t, topology_name, simulation_df):
    possible_topologies = ['RotorNet', 'Cerberus', 'ExpanderNet']
    if topology_name not in possible_topologies:
        raise ValueError(f'Select one of {possible_topologies}')
    sims_by_topology_df = filter_simulations_by_topology(cur, simulation_df, topology_name + 'TopologyConfiguration')

    sims_by_topology_df = get_unique_config_id_and_seed_column(cur, sims_by_topology_df)
    volumes = list(volumes)
    result_df = pd.DataFrame()
    for volume in volumes:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        time_dict = {}
        for sim_id_ds in list(sims_by_topology_df['sim_id_disregarding_seed'].unique()):
            selected_simulations = sims_by_topology_df[(sims_by_topology_df['sim_id_disregarding_seed'] == sim_id_ds) &
                                                       (sims_by_topology_df['volume_per_node'] == volume)]

            if not selected_simulations.empty:
                sim_ids = selected_simulations['simulation_id']
                topology_id = list(selected_simulations['fk_topology_id'])[0]
                algorithm_id = list(selected_simulations['fk_algorithm_id'])[0]
                max_times, max_rates, mean_rates, min_rates = calculate_overall_total_rate(cur, sim_ids, t,
                                                                                           show_plot=False)
                if not np.all((max_times == 0)):
                    label = f'top. id: {topology_id}, algo. id: {algorithm_id}'
                    time_dict[label] = max_times.tolist()[0]
                    if mean_rates is not None:
                        # plt.plot(t, max_rates, label='max')
                        p = ax.plot(t, mean_rates, label=label)
                        # plt.plot(t, min_rates, label='min')
                        ax.fill_between(t, min_rates, max_rates, color=p[-1].get_color(), alpha=0.2)
                        # ax.annotate(f'{str(sim_id_ds)} avg end time', xy=(avg_max_time, 0), xycoords='data',
                        #             xytext=(0, y_offset), textcoords='offset points',
                        #             arrowprops=dict(arrowstyle="->", ))
        if not len(time_dict) == 0:
            ax.set_xlabel(r'Time in $\mu s$')
            ax.set_ylabel(r'Total Rate in $MBit/s$')
            ax.set_title(f'{topology_name}, volume per node: ${float(volume):.0}$')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(workdir, f'plots/vol_{float(volume):.0}_{topology_name}_time_series.png'))
            plt.show()

            time_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in time_dict.items()]))
            sns.boxplot(data=time_df)
            plt.ylabel(r'completion time in $\mu s$')
            if len(time_df.columns) > 2:
                plt.xticks(rotation=90)
            plt.title(f'{topology_name}, volume per node: ${float(volume):.0}$')
            plt.tight_layout()
            plt.savefig(os.path.join(workdir, f'plots/vol_{float(volume):.0}_{topology_name}_boxplot.png'))
            plt.show()

    return result_df


t = np.linspace(0, 2e6, 1000)
volumes = simulation_df['volume_per_node'].unique()
# res = compare_total_rate_vs_topology(cur, volumes, t, rot_simulations_df, cerb_simulations_df, exp_simulations_df)
compare_simulations_within_one_topology(cur, volumes, t, 'Cerberus', simulation_df)
compare_simulations_within_one_topology(cur, volumes, t, 'RotorNet', simulation_df)
compare_simulations_within_one_topology(cur, volumes, t, 'ExpanderNet', simulation_df)

