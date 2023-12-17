import numpy as np

from scripts.auxiliary.database_connector import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.patches as mpatches

from scripts.visualization.plot_functions import *
# os.path.dirname(__file__) gets you the directory that script is in.
workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)  # append path of folder dsl_project_group_1

W = 3* 3.487
plt.rcParams.update({
    "text.usetex": True,
    'figure.figsize': (W, W/(4/3)),
    "font.family": "serif",
    "font.sans-serif": ["Libertine"],
    'font.size': 3*10,
    'axes.labelsize': 3*11,  # -> axis labels
    'legend.fontsize': 3*11,
})
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

mdb_cursor = get_cursor_to_cerberus()
# query = ("""SELECT simulation_id, fk_topology_id, fk_algorithm_id, max_time, topology.name, flowgenerator.parameter,
# flowgenerator.volume_per_node,flowgenerator.number_flows
# FROM simulation
# INNER JOIN topology on simulation.fk_topology_id = topology.topology_id
# INNER JOIN flowgenerator on simulation.fk_flow_generator_id = flowgenerator.flowgen_id
# INNER JOIN (SELECT fk_simulation_id, max(completion_time) as max_time
# FROM flowcompletiontime
# GROUP BY fk_simulation_id) as a on a.fk_simulation_id = simulation.simulation_id""")
#
# """Scheinbar nur 781 Simulation IDs"""
#
# df = get_df_from_sql_query(mdb_cursor,query)
# """Work on dataframe"""
# df['parameter'].replace({None:0,'{"arrival_time": 0.000000}':1},inplace=True)
# df['volume_per_node'].replace({None:0},inplace=True)
# df['number_flows'].replace({np.nan:0},inplace=True)
# df.to_csv('/home/dslgroup1/maresa/data/summary2.csv', index= False)
#
df = pd.read_csv('/home/dslgroup1/maresa/data/summary.csv')

simulation_df = get_df_from_table(mdb_cursor, 'simulation')
simulation_df = get_unique_config_id_and_seed_column(mdb_cursor, simulation_df)
offline = df[df['parameter'] == 1]
online = df[df['parameter'] == 0]

"""Offline cases check maximum times for every topology"""

demands = offline['volume_per_node'].unique()

times = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id'])['max_time'].mean().to_frame(name = 'avg_time').reset_index()
summary = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id','simulation_id'])['max_time'].mean().to_frame(name = 'max time').reset_index()
# on_summary = online.groupby(by=['number_flows','fk_topology_id','fk_algorithm_id','simulation_id'])['max_time'].mean().to_frame(name = 'max time').reset_index()

simulation_count = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id'])['simulation_id'].count().to_frame(name = 'count').reset_index()
# get seed from simulation_df (Leopold's function)
summary = pd.merge(summary,simulation_df[['simulation_id', 'seed']], on = 'simulation_id')

volumes = times['volume_per_node'].to_list()
topology_ids = times['fk_topology_id']
top_cerb = []
top_rot = []
top_exp = []
avg_demand_times = times['avg_time']
# normalize the avg times to plot -> divide by max time
markersize = 150
# size_time = [(i/max(avg_demand_times))*markersize for i in avg_demand_times]
size_time = [(i/1e4) for i in avg_demand_times]

for i in topology_ids:
    if offline.loc[offline['fk_topology_id']==i]['name'].values.all() == 'CerberusTopologyConfiguration':
        top_cerb.append(i)
        top_rot.append(np.nan)
        top_exp.append(np.nan)
    elif offline.loc[offline['fk_topology_id']==i]['name'].values.all() == 'RotorNetTopologyConfiguration':
        top_cerb.append(np.nan)
        top_rot.append(i)
        top_exp.append(np.nan)
    elif offline.loc[offline['fk_topology_id']==i]['name'].values.all() == 'ExpanderNetTopologyConfiguration':
        top_cerb.append(np.nan)
        top_rot.append(np.nan)
        top_exp.append(i)

def find_color(i):
    if i in top_cerb:
        return 'blue'
    elif i in top_rot:
        return 'green'
    else:
        return 'orange'


x_axis = list(zip(times['fk_topology_id'],times['fk_algorithm_id']))
categories = [str(i) for i in x_axis]
color_list = []
for i in x_axis:
    if i[0] in top_cerb:
        color_list.append('blue')
    elif i[0] in top_rot:
        color_list.append('green')
    else:
        color_list.append('orange')

# plt.figure(figsize=(20, 16))
# plt.scatter(categories, volumes, s = size_time, c = color_list)
# # for i,text in enumerate(simulation_count['count']):
# #     plt.annotate(text, (categories[i], volumes[i]))
# plt.xlabel("Topology ID", size=14)
# plt.ylabel("Volume per node", size=14)
# plt.grid(True)
# # plt.legend()
# plt.savefig('/home/dslgroup1/maresa/plots/test1.png')
# plt.show()

# plt.figure(figsize=(14, 8))
# # plt.scatter(topology_ids, volumes, color='blue',s = size_time)
# plt.scatter(top_cerb,volumes, color='blue',s = size_time, label = 'Cerberus')
# plt.scatter(top_rot,volumes, color='green',s = size_time, label = 'RotorNet')
# plt.scatter(top_exp,volumes, color='orange',s = size_time, label = 'Expander')
# plt.xlabel("Topology ID", size=14)
# plt.ylabel("Volume per node", size=14)
# plt.grid(True)
# plt.legend()
# plt.savefig('/home/dslgroup1/maresa/plots/overview_topology2.pdf')
# plt.show()

def z_value_for_comparison(pop1,pop2):
    x1 = np.mean(pop1)
    print("X1",x1)
    x2 = np.mean(pop2)
    print("X2", x2)
    sig1 = np.std(pop1)
    sig2 = np.std(pop2)
    n1 = len(pop1)
    n2 = len(pop2)
    # mu1- mu2 = 0
    nom = x1-x2
    denom = np.sqrt((sig1**2/n1) + (sig2**2/n2))
    z = nom / denom
    return z


def compare_populations(pop1, pop2, pop3, name1, name2, name3):
    p_values = {}
    means = {}
    z1 = z_value_for_comparison(pop1, pop2)
    p_values[f"{name1}-{name2}"] = stats.norm.sf(z1)
    means[name1] = np.mean(pop1)

    z2 = z_value_for_comparison(pop1, pop3)
    p_values[f"{name1}-{name3}"] = stats.norm.sf(z2)
    means[name2] = np.mean(pop2)

    z3 = z_value_for_comparison(pop2, pop3)
    p_values[f"{name2}-{name3}"] = stats.norm.sf(z3)
    means[name3] = np.mean(pop3)

    return p_values,means


""" Create paired data when seed/volume is the same but topology is different"""
# create paired data between two pairs

chosen_volumes = [64000000000.00000, 160000000000.00000, 384000000000.00000]


def find_common_values(list1,list2):
    set1 = set(list1)
    set2 = set(list2)

    return list(set1.intersection(set2))


def find_volumes_per_node(id,alg):
    unique_values = summary.loc[(summary['fk_topology_id'] == id) &
                                (summary['fk_algorithm_id'] == alg)]['volume_per_node'].unique()
    return list(unique_values)



def compare(top_A,alg_A,top_B,alg_B,vol):
    #get times according to combination - paired data
    difference = []
    timeA = []
    timeB = []
    for v in vol:
        rowA = summary.loc[(summary['fk_topology_id'] == top_A) &
                           (summary['fk_algorithm_id'] == alg_A) &
                           (summary['volume_per_node'] == v)]
        rowB = summary.loc[(summary['fk_topology_id'] == top_B) &
                           (summary['fk_algorithm_id'] == alg_B) &
                           (summary['volume_per_node'] == v)]
        # print(rowA)
        # print(rowB)

        for indexa, rowa in rowA.iterrows():
            for indexb, rowb in rowB.iterrows():
                if rowa['seed'] == rowb['seed']:
                    # print(rowa['seed'],rowb['seed'])
                    timeA.append(rowa['max time']* 1e-3)
                    timeB.append(rowb['max time']* 1e-3)

    # print(timeA)
    # print(timeB)

    difference = np.subtract(timeA, timeB)
    return difference


test = compare(19,1,21,7,chosen_volumes)



def inference(distribution):
    """Inference on a given distribution (t-test)"""
    n = len(distribution)
    s = np.std(distribution)
    x = np.mean(distribution)

    se = s/np.sqrt(n)
    t_score = x / se
    # print("t score",t_score)
    df = n-1
    p_value = stats.t.sf(np.abs(t_score),df) * 2
    # print("p-value:", p_value)
    t_star = stats.t.ppf(1-0.025,df)
    ci = [x - t_star * se, x + t_star * se]
    return x,se,p_value,ci,n

# x = np.linspace(stats.t.ppf(0.01, df), stats.t.ppf(0.99, df), 100)
# plt.plot(x, stats.t.pdf(x, df),'r-', lw=5, alpha=0.6, label='t pdf')
# plt.show()

def analysis(a,b):
    volumes = find_common_values(find_volumes_per_node(a[0],a[1]),find_volumes_per_node(b[0],b[1]))
    dist = compare(a[0],a[1],b[0],b[1],volumes)
    if dist.size ==0:
        return np.nan
    mean, se, p, ci, n = inference(dist)
    # plot distribution and save figure
    # plt.hist(dist)
    # plt.hist(dist,bins = n,edgecolor = 'black')
    # plt.axvline(mean,0, color = 'red', label = 'mean = {:.2f}'.format(mean))
    # plt.xlabel("Difference in max completion time")
    # plt.ylabel("Count")
    # plt.title(f"{a} vs {b}")
    # plt.legend()
    # plt.show()
    # plt.savefig(f'/home/dslgroup1/maresa/plots/{a}vs{b}.png')


    # print("Mean difference", mean)
    # print("Standard Error difference", se)
    # print("Sample Size", n)
    # print("P-value", p)
    # print("Confidence Interval", ci)
    return ci, mean, se, p, n, dist

"""Extract all confidence interval by comparing every possible configuration 
with every other possible configuration
"""
# configs = sorted(list(set(x_axis)))
# confidence = np.zeros((len(configs),len(configs),2))
# # confidence = np.zeros((len(x_axis),len(x_axis),2))
# n = len(configs)
# for i in range(n):
#     for j in range(i,n):
#          if i != j:
#             confidence[i,j] = analysis(configs[i],configs[j])

# np.save('/home/dslgroup1/maresa/data/confidence.npy',confidence)
# np.save('/home/dslgroup1/maresa/data/axis.npy',configs)

"""Plot the most interesting configurations"""
c1,mean1, se1, p1, n1, dist1 = analysis((21,7),(23,6))
c2, mean2, se2, p2, n2, dist2 = analysis((19, 1),(23,6))
c3, mean3, se3, p3, n3, dist3 = analysis((21, 7),(19, 1))

#set number of bins across most extenve dataset (dist3)
nbins = np.linspace(min([min(dist1),min(dist2),min(dist3)]),max([max(dist1),max(dist2),max(dist3)]),50)
width = 5
height = width / 1.618


fig, axs = plt.subplots(3,1, sharex=True, constrained_layout=True, dpi = 500)


axs[0].hist(dist1 ,bins = nbins, edgecolor = 'black')
axs[0].axvline(mean1, color = 'red', label = '$\mu$ = {:.2f}'.format(mean1))
axs[0].axvline(0, color = 'white', alpha = 0, label = f'n = {n1}')
axs[0].axvline(0, color = 'white', alpha = 0, label = f'p = {p1:0.4f}')
# axs[0].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c1[0]:0.1f},{c1[1]:0.1f}]')
axs[0].set_ylabel('Count')
axs[0].set_title('(21,7) RotorNet - (23,6) Cerberus', fontsize = 3*11)

axs[1].hist(dist2,bins = nbins, edgecolor = 'black')
axs[1].axvline(mean2, color = 'red', label = '$\mu$ = {:.2f}'.format(mean2))
axs[1].axvline(0, color = 'white', alpha = 0, label = f'n = {n2}')
axs[1].axvline(0, color = 'white', alpha = 0, label = f'p = {p2:0.4f}')
# axs[1].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c2[0]:0.1f},{c2[1]:0.1f}]')
axs[1].set_ylabel('Count')
axs[1].set_title('(19,1) Expander - (23,6) Cerberus', fontsize = 3*11)

axs[2].hist(dist3, bins=nbins, edgecolor='black')
axs[2].axvline(mean3, color='red', label='$\mu$ = {:.2f}'.format(mean3))
axs[2].axvline(0, color = 'white', alpha = 0, label = f'n = {n3}')
axs[2].axvline(0, color = 'white', alpha = 0, label = f'p = {p3:0.4f}')
# axs[2].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c3[0]:0.1f},{c3[1]:0.1f}]')
axs[2].set_ylabel('Count')
axs[2].set_title('(21,7) RotorNet - (19,1) Expander', fontsize=3*11)

plt.xlabel('Time Difference in ms')

for ax in axs:
    #remove top and right graph boundary
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #print legend without frame
    ax.legend(frameon=True,
              facecolor='w',
              labelspacing=0.2, handlelength=0.5,
              handletextpad=0.3)

plt.savefig('/home/dslgroup1/maresa/plots/3_way_comparison1.png')
plt.show()

# c1,mean1, se1, p1, n1, dist1 = analysis((4,1),(15,6))
# c2, mean2, se2, p2, n2, dist2 = analysis((12, 4),(15,6))
# c3, mean3, se3, p3, n3, dist3 = analysis((12, 4),(4, 1))
#
# #set number of bins across most extenve dataset (dist3)
# nbins = np.linspace(min([min(dist3),min(dist2),min(dist1)]),max([max(dist3),max(dist2),max(dist1)]),60)
# width = 3.487
# height = 2.8*(width / 1.618)
#

# fig, axs = plt.subplots(3,1, sharex=True, constrained_layout=True, dpi = 500)
#
#
# axs[0].hist(dist1 ,bins = nbins, edgecolor = 'black')
# axs[0].axvline(mean1, color = 'red', label = '$\mu$ = {:.2f}'.format(mean1))
# axs[0].axvline(0, color = 'white', alpha = 0, label = f'n = {n1}')
# axs[0].axvline(0, color = 'white', alpha = 0, label = f'p = {p1:0.4f}')
# # axs[0].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c1[0]:0.1f},{c1[1]:0.1f}]')
# axs[0].set_ylabel('Count')
# axs[0].set_title('(4,1) RotorNet - (15,6) Cerberus', fontsize=11)
#
# axs[1].hist(dist2,bins = nbins, edgecolor = 'black')
# axs[1].axvline(mean2, color = 'red', label = '$\mu$ = {:.2f}'.format(mean2))
# axs[1].axvline(0, color = 'white', alpha = 0, label = f'n = {n2}')
# axs[1].axvline(0, color = 'white', alpha = 0, label = f'p = {p2:0.4f}')
# # axs[1].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c2[0]:0.1f},{c2[1]:0.1f}]')
# axs[1].set_ylabel('Count')
# axs[1].set_title('(12, 4) Expander - (15,6) Cerberus', fontsize = 11)
#
# axs[2].hist(dist3, bins=nbins, edgecolor='black')
# axs[2].axvline(mean3, color='red', label='$\mu$ = {:.2f}'.format(mean3))
# axs[2].axvline(0, color = 'white', alpha = 0, label = f'n = {n3}')
# axs[2].axvline(0, color = 'white', alpha = 0, label = f'p = {p3:0.4f}')
# # axs[2].axvline(0, color = 'white', alpha = 0, label = f'CI = [{c3[0]:0.1f},{c3[1]:0.1f}]')
# axs[2].set_ylabel('Count')
# axs[2].set_title('(12, 4) Expander - (4, 1) RotorNet',fontsize = 11)
#
# plt.xlabel('Time Difference in ms')
#
# for i,ax in enumerate(axs):
#     #remove top and right graph boundary
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     #print legend without frame
#     ax.legend(frameon = False,
#               labelspacing = 0.2, handlelength = 0.5,
#               handletextpad = 0.3)
#
# plt.savefig('/home/dslgroup1/maresa/plots/3_way_comparison1_2.png')
# plt.show()