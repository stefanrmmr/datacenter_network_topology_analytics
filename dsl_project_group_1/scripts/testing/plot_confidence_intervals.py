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

x = np.load('/home/dslgroup1/leopold/data/confidence.npy')
axis = np.load('/home/dslgroup1/leopold/data/axis.npy')
x = x[[x[0]!=1 for x in axis]]
axis = axis[[x[0]!=1 for x in axis]]

names = [str(x) for x in axis]
plot_funct.plot_confidence_intervall(x[0], names)