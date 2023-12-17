import numpy as np

from scripts.auxiliary.database_connector import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import scipy.stats as stats


from scripts.visualization.plot_functions import *
# os.path.dirname(__file__) gets you the directory that script is in.
workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)  # append path of folder dsl_project_group_1

W = 3*3.487
plt.rcParams.update({
    "text.usetex": True,
    'figure.figsize': (W, W/(4/3)),
    "font.family": "serif",
    "font.sans-serif": ["Libertine"],
    'font.size': 3*10,
    'axes.labelsize': 3*11,  # -> axis labels
    'legend.fontsize': 3*11,
})



df = pd.read_csv('/home/dslgroup1/maresa/data/summary.csv')

# simulation_df = get_df_from_table(mdb_cursor, 'simulation')
offline = df[df['parameter'] == 1]
online = df[df['parameter'] == 0]

demands = offline['volume_per_node'].unique()

times = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id'])['max_time'].mean().to_frame(name = 'avg_time').reset_index()
summary = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id','simulation_id'])['max_time'].mean().to_frame(name = 'max time').reset_index()

simulation_count = offline.groupby(by=['volume_per_node','fk_topology_id','fk_algorithm_id'])['simulation_id'].count().to_frame(name = 'count').reset_index()

volumes = times['volume_per_node'].to_list()
volumes = [i * 1.25 *1e-10 for i in volumes]
topology_ids = times['fk_topology_id']
top_cerb = []
top_rot = []
top_exp = []
avg_demand_times = times['avg_time']
# normalize the avg times to plot -> divide by max time
# markersize = 50
# size_time = [(i/max(avg_demand_times))*markersize for i in avg_demand_times]
# size_time = [(i/1e4) for i in avg_demand_times]

def radius(size_time):
    minimum = min(size_time)
    print(minimum)
    maximum = max(size_time)
    print(maximum)
    markersize = 10
    rad = [(i / minimum) * markersize for i in size_time]
    return rad

def radius2(size_time):
    size_time = [i * 1e-6 for i in size_time]
    minimum = min(size_time)
    print(minimum)
    maximum = max(size_time)
    print(maximum)
    a = 10
    markersize = 5
    rad = [(a/i) * markersize for i in size_time]
    return rad


size_time = radius2(avg_demand_times)

'''Sort into groups to color marker depending on relation'''

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


x_axis = list(zip(times['fk_topology_id'],times['fk_algorithm_id']))
categories = [str(i) for i in x_axis]
# create color list to match a color to each marker
color_list = []
for i in x_axis:
    if i[0] in top_cerb:
        color_list.append('blue')
    elif i[0] in top_rot:
        color_list.append('green')
    else:
        color_list.append('orange')


width = 9
height = width / 1.618
plt.figure(dpi = 500)
plt.grid(True)
plt.scatter(categories, volumes, s=size_time, c=color_list, alpha=0.9, edgecolors='k')
# for i,text in enumerate(simulation_count['count']):
#     plt.annotate(text, (categories[i], volumes[i]))
# plt.xlabel("(Topology ID, Algorithm ID)", size=11)
# plt.xticks(rotation=45)
plt.axhline(8,0,1, c='navy', alpha = 0.5)
# plt.annotate("CI",(35.7,8), fontsize = 14)
plt.tick_params(bottom=False, labelbottom=False)
plt.ylabel("Volume per node in GB")
ax = plt.gca()

# Create a Rectangle patch
rect = Rectangle((19.5,0),7,20,linewidth=2,edgecolor='crimson',facecolor='none',alpha = 0.5)
plt.annotate("Group 2",(20,21))
rect1 = Rectangle((29.5,6),7,60,linewidth=2,edgecolor='purple',facecolor='none',alpha = 0.5)
plt.annotate("Group 1",(23,58))
# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect1)

legend_elements = [Line2D([0], [0], marker='o', color='w', label='RotorNet',markerfacecolor='green', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Cerberus',markerfacecolor='blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Expander',markerfacecolor='orange', markersize=10),
                   Line2D([0], [0], marker='o',color='w', label='min 0.223 s ',markerfacecolor='darkgrey', markersize=15),
                   Line2D([0], [0],marker='o', color='w', label='max 8.074 s',markerfacecolor='darkgrey', markersize=7),
                   ]
plt.legend(handles=legend_elements, loc='best', facecolor = 'w', framealpha = 1)
plt.savefig('/home/dslgroup1/maresa/plots/overview_topology1.png', bbox_inches='tight',pad_inches = 0.1)
plt.show()