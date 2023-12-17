from scripts.auxiliary.database_connector import *
import scripts.visualization.plot_functions as plot_funct

import sys, os
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import pprint
#
# width = 3.487
# height = width / (4/3)1.618

tum_blue = '#3070b3'  # define TUM corporate branded blue for plots

W = 2*3.487
plt.rcParams.update({
    "text.usetex": True,
    'figure.figsize': (W, W/(4/3)),
    "font.family": "serif",
    "font.sans-serif": ["Libertine"],
    'font.size': 2*10,
    'axes.labelsize': 2*11,  # -> axis labels
    'legend.fontsize': 2*11,
})


def plot_confidence_intervall(matrix,list_of_names):
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)
    y_ticks = range(len(list_of_names))
    bar_height = 0.6

    for row,y in zip(matrix,y_ticks):
        for i,element in enumerate(row):
            wd = element[1]-element[0]
            ax.barh(y,width=element[1]-element[0], left=element[0])
            # plt.annotate(list_of_names[i],(element[1], y))
            # ax.vlines(interval, ymin=y-bar_height*0.9, ymax=y+bar_height*0.9, color=tum_blue)
    plt.axvline(x=0, color='red')
    # ax.barh(0, width=0, color='white', height=bar_height, hatch='//', edgecolor=tum_blue, label='lower completion time')
    # ax.barh(0, width=0, color=tum_blue, height=bar_height, edgecolor=tum_blue, label='higher completion time')
    plt.yticks(y_ticks, list_of_names)
    plt.xlabel('difference of completion time')
    plt.ylabel('configurations')
    # ax.legend()
    fig.tight_layout()
    plt.savefig('/home/dslgroup1/maresa/plots/confidence_wo_top1.png')
    plt.show()


def plot_confidence_intervall2(list_of_intervalls, list_of_names):
    fig = plt.figure(dpi=500)
    ax = fig.add_subplot(111)
    combi = [x for x in zip(list_of_intervalls, list_of_names) if ~np.isnan(x[0]).all()]
    bar_height = 0.3
    for y, (interval, name) in enumerate(combi):
        if list_of_intervalls.all() == 0:
            break
        interval_below_zero = (min(interval[0], 0), min(interval[1], 0))
        below_zero_width = abs(abs(interval_below_zero[1]) - abs(interval_below_zero[0]))
        interval_above_zero = (max(interval[0], 0), max(interval[1], 0))
        above_zero_width = abs(abs(interval_above_zero[1]) - abs(interval_above_zero[0]))
        if below_zero_width > 0:
            ax.barh(y, width=below_zero_width, left=interval_below_zero[0], color='white', height=bar_height,
                    hatch='//', edgecolor=tum_blue)
        if above_zero_width > 0:
            ax.barh(y, width=above_zero_width, left=interval_above_zero[0], color=tum_blue, height=bar_height,
                    edgecolor=tum_blue)
        ax.vlines(interval, ymin=y - bar_height * 0.9, ymax=y + bar_height * 0.9, color=tum_blue)
    plt.axvline(x=0, color='red')
    ax.barh(0, width=0, color='white', height=bar_height, hatch='//', edgecolor=tum_blue, label='faster')
    ax.barh(0, width=0, color=tum_blue, height=bar_height, edgecolor=tum_blue, label='slower')
    plt.yticks(range(len(combi)), [x[1] for x in combi])
    plt.xlabel('$\Delta$ (23,6) completion time [s]')
    plt.ylabel('Confidence Interval')
    ax.legend(frameon=False, loc = 'upper right')
    fig.tight_layout()
    plt.margins(x=0.1,y=0.01)
    plt.savefig('/home/dslgroup1/maresa/plots/ci_236.png')
    plt.show()

pp = pprint.PrettyPrinter(width=100, compact=True, indent = 3)
x = np.load('/home/dslgroup1/maresa/data/confidence.npy')
x = x* 1e-6
# pp.pprint(x[0,0,0])
# print(len(x[0]))
axis = np.load('/home/dslgroup1/maresa/data/axis2.npy')
# get index without topology 1

idx =0
for i in axis:
    if i[0] != 1:
        break
    idx += 1

axis = axis[(idx-1):-1]
# x = x[[x[0]!=1 for x in axis]]
# axis = axis[[x[0]!=1 for x in axis]]
# index([23,6])
names = [str(x) for x in axis]

plot_confidence_intervall2(x[(idx-1):34,35,:], names)
