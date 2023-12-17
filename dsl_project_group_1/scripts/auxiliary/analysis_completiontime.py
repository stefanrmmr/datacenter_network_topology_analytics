import sys
import os
import pandas as pd
import numpy as np
from scripts.auxiliary.database_connector import *
import matplotlib.pyplot as plt
import scipy.stats as stats


from scripts.visualization.plot_functions import *
# os.path.dirname(__file__) gets you the directory that script is in.
workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)  # append path of folder dsl_project_group_1

mdb_cursor = get_cursor_to_cerberus()

"""STEP 1: AGGREGATE DATA: GET ALL OFFLINE CASES, GET MAX COMPLETION TIME PER SIMULATION ID AND COMBINE WITH TOPOLOGY ID"""

query = ("SELECT simulation_id, fk_topology_id, max_time "
         "FROM (simulation "
         "INNER JOIN"
         "(SELECT fk_simulation_id, max(completion_time) as max_time "
         "FROM (flowcompletiontime "
         "INNER JOIN (SELECT flow_id FROM flow WHERE arrival_time = 0) AS b on b.flow_id = flowcompletiontime.fk_flow_id) "
         "GROUP BY fk_simulation_id) AS a ON a.fk_simulation_id = simulation.simulation_id)")

mdb_cursor.execute(query)
# DB is a dataframe containing offline simulations, with the simulation ID,
# the topology ID and the maximum flow completion time
db = get_df_from_cursor(mdb_cursor)
print(db)
overall_mean = db["max_time"].mean()

# Get distinct Topology Names
query = "SELECT DISTINCT name FROM topology"
mdb_cursor.execute(query)
names = [i[0] for i in mdb_cursor.fetchall()]
ids = {}
query = "select topology_id from topology where name = 'RotorNetTopologyConfiguration'"
mdb_cursor.execute(query)
ids[names[0]] =[j[0] for j in mdb_cursor.fetchall()]
query = "select topology_id from topology where name = 'CerberusTopologyConfiguration'"
mdb_cursor.execute(query)
ids[names[1]] =[j[0] for j in mdb_cursor.fetchall()]
query = "select topology_id from topology where name = 'ExpanderNetTopologyConfiguration'"
mdb_cursor.execute(query)
ids[names[2]] =[j[0] for j in mdb_cursor.fetchall()]

# ids contains the 3 distinct Topology Names and the corresponding topology ids
# sim contains all simulation ids belonging to the 3 main topologies.
sim = {}

for key in ids.keys():
    query = f"select simulation_id from simulation where fk_topology_id in {tuple(ids[key])}"
    mdb_cursor.execute(query)
    sim[key] = [i[0] for i in mdb_cursor.fetchall()]



"""Check whether ANOVA can be done"""
"""Normality: Groups should have a normal distribution -- check histogram plots"""
"""Variability: Groups should have an about equal variability -- check_variability should
                be small"""
leng = []
variability = []
max_times = {}

for key in sim.keys():
    times = db[db['simulation_id'].isin(sim[key])]['max_time'].astype(float)
    times = times.tolist()
    max_times[key] = times
    leng.append(len(times))

    plot_histogram_single(times,'fd','times','count',None,'hist')
    variability.append(np.var(times))

check_variability = np.var(variability)
plt.hist(max_times['RotorNetTopologyConfiguration'], bins='fd')
plt.show()
print(f"Variability is too large. The variability on the variabilities is {check_variability}")
print(f"Furthermore the number of samples in the groups are not equal as the lengths are:"
      f"{leng}")

"""Sample the same amount out of the groups to retry ANOVA, goal: do a ANOVA one way f_test"""



"""Compare the average between 2 topologies each
1. RotorNet vs Cerberus"""

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


rot = max_times['RotorNetTopologyConfiguration']
cerb = max_times['CerberusTopologyConfiguration']
expand = max_times['ExpanderNetTopologyConfiguration']
z1 = z_value_for_comparison(rot,cerb)
p_value1 = stats.norm.sf(z1)
print("P_value 1 Rot vs Cerb", p_value1)

z2 = z_value_for_comparison(rot,expand)
p_value2 = stats.norm.sf(z2)
print("P_value 2, Rot vs Expand ", p_value2)

z3 = z_value_for_comparison(cerb,expand)
p_value3 = stats.norm.sf(z3)
print("P_value 3, Cerv vs Expand ", p_value3)

# p_value2 = stats.ttest_ind(rot,cerb)

# groups = db.groupby("fk_topology_id")
# size = []
# variability = []
# for i in range(1,len(groups)):
#     group_1 = groups.get_group(i)
#     size.append(len(group_1))
#     plt.hist(group_1["max_time"],bins="fd")
#     plt.show()
#     variability.append(np.var(group_1["max_time"].astype(float)))
#
# print(group_1)

"""STEP 2: Group by Topologies to get the AVERAGE demand completion time per topology,
 averaging over the number of simulations"""
# group by topology ID and return mean of max_time
# mean = db.groupby("fk_topology_id")["max_time"].mean()
# print(mean)

"""Hypothesis Testing: Using the average demand completion time
H0 = There is no impact of the topology on the demand completion time, the mean is the same for all topologies
H_A = There is impact of the topology on the demand completion time, one mean is different
ANOVA Conditions:
"""


