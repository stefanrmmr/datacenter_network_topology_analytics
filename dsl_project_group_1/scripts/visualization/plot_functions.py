# FUNCTIONS for Graphs and visual Outputs [DSML Group 1]
# group members responsible: Stefan Rummer (03709476)

from numpy.polynomial.polynomial import polyfit
import matplotlib.ticker as mpl_ticker
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib import rc
import pandas as pd
import matplotlib as mpl
import numpy as np
import statistics
import sys
import os
from matplotlib import rc
from matplotlib import rc
from scipy.stats import gaussian_kde


rc('font', **{'family': 'serif', 'sans-serif': ['Libertine']})
rc('text', usetex=True)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)
# plt.rc('figure', titlesize=10)

workdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(workdir)  # append path of folder dsl_project_group_1

tum_blue = '#3070b3'  # define TUM corporate branded blue for plots

width = 3.487
height = width / 1.618


def plot_scatter_single(arr_x, arr_y, x_label, y_label,
                        data_label, reg_line):
    """
    Args:
        arr_x: array containing #n x values
        arr_y: array containing #n y values
        x_label: X axis label
        y_label: Y axis label
        data_label: legend description of datapoints
        plot_title: header title for plot
        reg_line: plot regression line?
    Returns:
        scatter plot
    """
    plt.rc('legend', fontsize=6)

    fig = plt.figure(dpi=400, figsize=(width, height))

    ax = fig.add_subplot(111)
    ax.scatter(arr_x, arr_y, edgecolors='black', c=tum_blue, label=str(data_label))

    #ax.set_title(str(plot_title))
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.grid(linestyle='--')

    if reg_line:
        # plot lin regression fitted ideal line
        idx = np.isfinite(arr_x) & np.isfinite(arr_y)
        b, m = polyfit(arr_x[idx], arr_y[idx], 1)
        ax.plot(arr_x, b + m * arr_x, '-', color="orange", label='linear approximation')

    ax.set_xticklabels([])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])  # define custom ticks
    ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])  # name custom tick labels
    ax.legend(loc=2)
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    plt.tight_layout()
    fig.savefig(f"{workdir}/plots/scatter_algoperformance_{time_of_analysis}.png")
    fig.show()


def plot_scatter_duo(arr_xl, arr_yl, arr_xr, arr_yr, x_label, y_label,
                     data_label, plot_title_l, plot_title_r, reg_line):
    """
    Args:
        arr_xl: array containing #n x values    AX_LEFT
        arr_yl: array containing #n y values    AX_LEFT
        arr_xr: array containing #n x values    AX_RIGHT
        arr_yr: array containing #n y values    AX_RIGHT
        x_label: X axis label
        y_label: Y axis label
        data_label: datapoints description for both plots
        plot_title_l: header title for plot     AX_LEFT
        plot_title_r: header title for plot     AX_RIGHT
        reg_line: plot regression line?
    Returns:
        2x scatter plots side by side
    """

    fig = plt.figure(dpi=400, figsize=(width * 2, height))
    ax_left = fig.add_subplot(121)  # left plot     AX_LEFT
    ax_right = fig.add_subplot(122)  # right plot    AX_RIGHT

    x_min = min(min(arr_xl), min(arr_xr)) - min(min(arr_xl), min(arr_xr)) * 0.2
    x_max = max(max(arr_xl), max(arr_xr)) + max(max(arr_xl), max(arr_xr)) * 0.2
    y_min = min(min(arr_yl), min(arr_yr)) - min(min(arr_yl), min(arr_yr)) * 0.2
    y_max = max(max(arr_yl), max(arr_yr)) + max(max(arr_yl), max(arr_yr)) * 0.2

    ax_left.scatter(arr_xl, arr_yl, edgecolors='black', c=tum_blue, label=str(data_label))
    ax_left.legend(loc='upper left')
    ax_left.set_title(str(plot_title_l))
    ax_left.set_xlabel(str(x_label))
    ax_left.set_ylabel(str(y_label))
    ax_left.set_xlim(x_min, x_max)
    ax_left.set_ylim(y_min, y_max)
    ax_left.grid(linestyle='--')

    ax_right.scatter(arr_xr, arr_yr, edgecolors='black', c='teal', label=str(data_label))
    ax_right.legend(loc='upper left')
    ax_right.set_title(str(plot_title_r))
    ax_right.set_xlabel(str(x_label))
    ax_right.set_ylabel(str(y_label))
    ax_right.set_xlim(x_min, x_max)
    ax_right.set_ylim(y_min, y_max)
    ax_right.grid(linestyle='--')

    if reg_line:
        # plot lin regression fitted ideal line     AX_LEFT
        bl, ml = polyfit(arr_xl, arr_yl, 1)
        ax_left.plot(arr_xl, bl + ml * arr_xl, '-', color="black")
        # plot lin regression fitted ideal line     AX_RIGHT
        br, mr = polyfit(arr_xr, arr_yr, 1)
        ax_left.plot(arr_xr, br + mr * arr_xr, '-', color="black")

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/scatter_duo_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/scatter_duo_{time_of_analysis}.png")
    fig.show()


def plot_histogram_single(arr_x, n_bins, x_label, y_label, data_label, plot_title):
    """
    Args:
        arr_x: array containing #n x values
        n_bins: number of separate columns in histogram ("resolution")
        x_label: X axis label
        y_label: Y axis label
        data_label: datapoints description
        plot_title: header title for plot
    Returns:
        histogram plot
    """

    fig = plt.figure(dpi=400, figsize=(width, height))

    ax = fig.add_subplot(111)
    ax.hist(arr_x, bins=n_bins, density=True, edgecolor='black', facecolor=tum_blue, label=str(data_label))
    ax.xaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())

    ax.set_title(str(plot_title))
    ax.set_xlabel(str(x_label), weight='bold')
    ax.set_ylabel(str(y_label), weight='bold')
    ax.grid(linestyle='--')
    ax.legend()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.tight_layout()
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.pdf")
    fig.show()


def plot_histogram_ttest(arr_x, n_bins, border, x_label, y_label, label, color):
    """
    Args:
        arr_x: array containing #n x values
        n_bins: number of separate columns in histogram ("resolution")
        border: indicator for significant outperformance
        x_label: X axis label
        y_label: Y axis label
    Returns:
        histogram plot
    """
    plt.rc('legend', fontsize=6)

    if color == 'tum_blue':
        color = tum_blue

    arr = list(arr_x)
    arr = [x/1000000 for x in arr]
    border = border/1000000

    fig = plt.figure(dpi=500, figsize=(width, height))
    ax = fig.add_subplot(111)
    ax.hist(arr, bins=n_bins, density=True, facecolor=color, label=label)
    ax.xaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())

    ax.axvline(border, label=fr'critical value border', color='orange')
    ax.axvspan(0.75, border, facecolor='None', hatch='//', edgecolor='orange', alpha=0.5,
               label=f'reject $H_0$, significant\n outperformance')

    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.grid(linestyle='--', color="lightgrey", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/HISTO_TTEST_{time_of_analysis}.png")
    fig.show()

def plot_histogram_fct_log(arr_x, n_bins, x_label, y_label, data_label, plot_title):
    """
    Args:
        arr_x: array containing #n x values
        n_bins: number of separate columns in histogram ("resolution")
        x_label: X axis label
        y_label: Y axis label
        data_label: datapoints description
        plot_title: header title for plot
    Returns:
        histogram plot
    """

    plt.rc('legend', fontsize=8)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)

    fig = plt.figure(dpi=400, figsize=(width, height))

    ax = fig.add_subplot(111)
    arr_x = [entry/1000000 for entry in arr_x]
    ax.hist(arr_x, bins=n_bins, density=True, edgecolor='black', facecolor=tum_blue, label=str(data_label))
    ax.xaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.set_yscale('log', base=10)
    ax.set_yticklabels([])
    # ax.set_title(plot_title)
    ax.set_xlabel(str(x_label), weight='bold')
    ax.set_ylabel(str(y_label), weight='bold')
    ax.grid(color='lightgrey', linestyle='--')
    ax.legend()

    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.tight_layout()
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.png")
    fig.show()


def plot_algorithm_performance(arr_height_l, arr_labels, x_label, y_label, plot_title):

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('figure', titlesize=8)
    #plt.rc('axes', labelsize=8)

    fig = plt.figure(dpi=500, figsize=(width, height))
    ax = fig.add_subplot(111)

    bars = arr_labels
    x_pos = np.arange(len(bars))
    x_pos = x_pos - 1  # TODO does this work ?
    y_max = max(arr_height_l) + max(arr_height_l) * 0.2

    # AX_LEFT Create bar plot for demand completion times
    ax.bar(x_pos, arr_height_l, align="center", width=0.8,
           color=tum_blue, edgecolor="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    #ax.set_xticklabels([])
    # ax.set_ylim(0, y_max)
    ax.set_ylim(0, 9000000)
    ax.grid(axis='y', linestyle='--', color="lightgrey")

    ax.set_xticks(x_pos)  # define custom ticks
    ax.set_xticklabels(bars)  # name custom tick labels
    ax.xaxis.set_tick_params(rotation=45)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/algoperformance_{time_of_analysis}.png")
    fig.show()


def plot_algorithm_performance_ranked(arr_height_l, arr_labels, x_label, y_label, plot_title):

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('figure', titlesize=8)
    #plt.rc('axes', labelsize=8)

    data_tuples = list(zip(arr_height_l, arr_labels))
    df = pd.DataFrame(data_tuples, columns=['height', 'label'])
    df.sort_values(['height'], inplace=True)

    arr_height_l = df['height'].tolist()
    arr_labels = df['label'].tolist()

    fig = plt.figure(dpi=500, figsize=(width, height))
    ax = fig.add_subplot(111)

    bars = arr_labels
    x_pos = np.arange(len(bars))
    x_pos = x_pos - 1  # TODO does this work ?
    y_max = max(arr_height_l) + max(arr_height_l) * 0.2

    # AX_LEFT Create bar plot for demand completion times
    ax.bar(x_pos, arr_height_l, align="center", width=0.7,
           color=tum_blue, edgecolor="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    #ax.set_xticklabels([])
    # ax.set_ylim(0, y_max)
    ax.set_ylim(0, 9000000)
    ax.grid(axis='y', linestyle='--', color="lightgrey", linewidth=0.5)

    ax.set_xticks(x_pos)  # define custom ticks
    ax.set_xticklabels(bars)  # name custom tick labels
    ax.xaxis.set_tick_params(rotation=45)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/algoperformance_RANKED{time_of_analysis}.png")
    fig.show()


def plot_algorithm_performance_overlap(arr_vol1, arr_vol2, arr_vol3, arr_vol4,
                                       arr_labels, x_label, y_label, plot_title):

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('figure', titlesize=8)
    plt.rc('legend', fontsize=6)
    #plt.rc('axes', labelsize=8)

    fig = plt.figure(dpi=500, figsize=(width, height))
    ax = fig.add_subplot(111)

    bars = arr_labels
    x_pos = np.arange(len(bars))
    print(x_pos)
    x_pos = x_pos - 1  # TODO does this work ?

    x_pos_shift1 = [x+0.15 for x in x_pos]
    x_pos_shift2 = [x+0.3 for x in x_pos]
    x_pos_shift3 = [x+0.45 for x in x_pos]

    arr_vol1 = [x/1000000 for x in arr_vol1]
    arr_vol2 = [x/1000000 for x in arr_vol2]
    arr_vol3 = [x/1000000 for x in arr_vol3]
    arr_vol4 = [x/1000000 for x in arr_vol4]

    # AX Create bar plot for demand completion times
    ax.bar(x_pos_shift3, arr_vol4, align="center", width=0.3, color='teal', label="vol. 8.0 [GB]")
    ax.bar(x_pos_shift2, arr_vol3, align="center", width=0.3, color='darkslategrey', label="vol. 4.0 [GB]")
    ax.bar(x_pos_shift1, arr_vol2, align="center", width=0.3, color='orange', label="vol. 2.5 [GB]")
    ax.bar(x_pos, arr_vol1, align="center", width=0.3, color=tum_blue, label="vol. 1.0 [GB]")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(plot_title)
    ax.legend(loc=2)
    #ax.set_xticklabels([])
    # ax.set_ylim(0, y_max)
    ax.set_ylim(0, 9)
    ax.grid(axis='y', linestyle='--', color="lightgrey", linewidth=0.5)

    ax.set_xticks(x_pos_shift2)  # define custom ticks
    ax.set_xticklabels(bars)  # name custom tick labels
    ax.xaxis.set_tick_params(rotation=45)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/algoperformance_OVERLAP{time_of_analysis}.png")
    fig.show()


def plot_algorithm_performance_bestof(arr_vol1, arr_vol2, arr_vol3, arr_vol4,
                                       arr_labels, x_label, y_label, plot_title):

    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('figure', titlesize=8)
    plt.rc('legend', fontsize=6)

    fig = plt.figure(dpi=500, figsize=(width, height))
    ax = fig.add_subplot(111)

    df4 = pd.DataFrame(list(zip(arr_vol4, arr_labels)), columns=['height', 'label'])
    df4.sort_values(['height'], inplace=True)
    df4 = df4.head(5)
    df3 = pd.DataFrame(list(zip(arr_vol3, arr_labels)), columns=['height', 'label'])
    df3.sort_values(['height'], inplace=True)
    df3 = df3.head(5)
    df2 = pd.DataFrame(list(zip(arr_vol2, arr_labels)), columns=['height', 'label'])
    df2.sort_values(['height'], inplace=True)
    df2 = df2.head(5)
    df1 = pd.DataFrame(list(zip(arr_vol1, arr_labels)), columns=['height', 'label'])
    df1.sort_values(['height'], inplace=True)
    df1 = df1.head(5)

    df1_list_dct = df1['height'].tolist()
    df1_list_dct = [x/1000000 for x in df1_list_dct]
    df1_list_labels = df1['label'].tolist()
    df1_list_labels.append("")

    df2_list_dct = df2['height'].tolist()
    df2_list_dct = [x/1000000 for x in df2_list_dct]
    df2_list_labels = df2['label'].tolist()
    df2_list_labels.append("")

    df3_list_dct = df3['height'].tolist()
    df3_list_dct = [x/1000000 for x in df3_list_dct]
    df3_list_labels = df3['label'].tolist()
    df3_list_labels.append("")

    df4_list_dct = df4['height'].tolist()
    df4_list_dct = [x/1000000 for x in df4_list_dct]
    df4_list_labels = df4['label'].tolist()

    bars = df1_list_labels + df2_list_labels + df3_list_labels + df4_list_labels
    x_pos = np.arange((len(df1_list_dct)*4 + 3))
    #x_pos = x_pos - 1  # TODO does this work ?

    ax.bar(x_pos[18:23], df4_list_dct, align="center", width=0.8, color='teal', label="vol. 8.0 [GB], top 5 algo.")
    ax.bar(x_pos[12:17], df3_list_dct, align="center", width=0.8, color='darkslategrey', label="vol. 4.0 [GB], top 5 algo.")
    ax.bar(x_pos[6:11], df2_list_dct, align="center", width=0.8, color='orange', label="vol. 2.5 [GB], top 5 algo.")
    ax.bar(x_pos[0:5], df1_list_dct, align="center", width=0.8, color=tum_blue, label="vol. 1.0 [GB], top 5 algo.")
    # ax.bar_label(df1_list_labels, padding=3)

    x_pos_shift1 = [x+0.15 for x in x_pos]
    x_pos_shift2 = [x+0.3 for x in x_pos]
    x_pos_shift3 = [x+0.45 for x in x_pos]

    # AX Create bar plot for demand completion times
    #ax.bar(x_pos_shift3, arr_vol4, align="center", width=0.3, color='teal', label="vol. 64 Gbit")
    #ax.bar(x_pos_shift2, arr_vol3, align="center", width=0.3, color='darkslategrey', label="vol. 32 Gbit")
    #ax.bar(x_pos_shift1, arr_vol2, align="center", width=0.3, color='orange', label="vol. 20 Gbit")
    #ax.bar(x_pos, arr_vol1, align="center", width=0.3, color=tum_blue, label="vol. 8 Gbit")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(plot_title)
    ax.legend(loc=2)
    ax.set_xticklabels([])
    ax.grid(axis='y', linestyle='--', color="lightgrey", linewidth=0.5)

    ax.set_xticks(x_pos)  # define custom ticks
    ax.set_xticklabels(bars)  # name custom tick labels
    ax.xaxis.set_tick_params(rotation=45)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/algoperformance_BESTOF{time_of_analysis}.png")
    fig.show()


def plot_histogram_duo(arr_xl, arr_xr, n_bins, x_label, y_label,
                       data_label, plot_title_l, plot_title_r):
    """
    Args:
        arr_xl: array containing #n x values    AX_LEFT
        arr_xr: array containing #n y values    AX_RIGHT
        n_bins: number of separate columns in histogram ("resolution")
        x_label: X axis label
        y_label: Y axis label
        data_label: datapoints description for both plots
        plot_title_l: header title for plot     AX_LEFT
        plot_title_r: header title for plot     AX_RIGHT
    Returns:
        2x histograms plots side by side
    """

    fig = plt.figure(dpi=400, figsize=(width * 2, height))
    ax_left = fig.add_subplot(121)  # left plot     AX_LEFT
    ax_right = fig.add_subplot(122)  # right plot    AX_RIGHT

    ax_left.hist(arr_xl, bins=n_bins, density=True, facecolor=tum_blue, label=str(data_label))
    ax_left.legend(loc='upper left')
    ax_left.set_title(str(plot_title_l))
    ax_left.set_xlabel(str(x_label))
    ax_left.set_ylabel(str(y_label))
    ax_left.grid(linestyle='--')

    ax_right.hist(arr_xr, bins=n_bins, density=True, facecolor='teal', label=str(data_label))
    ax_right.legend(loc='upper left')
    ax_right.set_title(str(plot_title_r))
    ax_right.set_xlabel(str(x_label))
    ax_right.set_ylabel(str(y_label))
    ax_right.grid(linestyle='--')

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/histogram_duo_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/histogram_duo_{time_of_analysis}.png")
    fig.show()


def plot_violin_extended(arr_x, x_label, data_label, plot_title):
    """
    Args:
        arr_x: array containing #n x values
        x_label: X axis label
        data_label: datapoints description
        plot_title: header title for plot
    Returns:
        violin plot overlay with boxplot, scatter of datapoints, mean value indicator
        (the real deal!, the ultimate plot experience)
    """

    mean_value = statistics.mean(arr_x)
    mean_value_str = str('{:0.2f}'.format(mean_value))

    fig = plt.figure(dpi=400, figsize=(width, height))
    ax = fig.add_subplot(111)

    # create violin plot
    parts = ax.violinplot(arr_x, points=100, vert=False,
                          showmeans=False, showextrema=False, showmedians=False)
    # color the violin plot body
    for pc in parts['bodies']:
        pc.set_facecolor("olive")
        pc.set_edgecolor('black')
        pc.set_alpha(0.2)

    meanprops = dict(linestyle='-', linewidth=2, color='black')
    medianprops = dict(linestyle='-.', linewidth=0, color='firebrick')
    buckets = np.random.uniform(low=0.97, high=1.03, size=(len(arr_x),))

    ax.scatter(arr_x, buckets, edgecolors='black', color=tum_blue, alpha=0.7, label=data_label)
    ax.boxplot(arr_x, medianprops=medianprops, meanprops=meanprops,
               showmeans=True, meanline=True, vert=False, showfliers=False)

    # plot lines for mean indicator and plot separation
    ax.plot([mean_value, mean_value], [0.7, 0.75], color=tum_blue, linewidth=3)
    ax.plot([mean_value, mean_value], [0.65, 1.3], color=tum_blue, linewidth=1)
    ax.plot([min(arr_x) - min(arr_x * 0.2), max(arr_x) + max(arr_x) * 0.2],
            [1.3, 1.3], color="grey", linewidth=1)

    # add text label with information regarding Mean Value
    ax.text(-1.028, 1.338, f' {mean_value_str} mean value ', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=tum_blue, boxstyle='round'))

    # titles, axis formatting, output
    ax.set_title(f"{plot_title}", pad=15, weight='bold')
    ax.set_xlabel(f"{x_label}", fontsize=10, weight='bold')
    ax.set_xlim(min(arr_x) - min(arr_x * 0.2), max(arr_x) + max(arr_x) * 0.2)
    ax.set_ylim(0.7, 1.6)
    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.grid(linestyle='--')
    ax.xaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    ax.legend(loc='upper left', prop={'size': 9})

    # PLOT final output
    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.png")
    fig.show()


def plot_bars_duo(arr_height_l, arr_height_r, arr_labels,
                  x_label="server racks", y_label="y axis:", plot_title="bar_plot", ticks=False):
    """
    Args:
        arr_height_l: array containing #n height values
        arr_height_r: array containing #n height values
        arr_labels: #n X axis labels
        x_label: x axis label
        y_label: y axis label
        plot_title: header title for plot
        ticks: display x axis ticks? (boolean)
    Returns:
        Bar plot for the #n values
    """

    fig = plt.figure(dpi=250, figsize=(width * 2, height))
    ax_left = fig.add_subplot(121)  # left plot     AX_LEFT
    ax_right = fig.add_subplot(122)  # right plot    AX_RIGHT

    bars = arr_labels
    x_pos = np.arange(len(bars))
    x_pos = x_pos - 1  # TODO does this work ?
    y_max = max(max(arr_height_l), max(arr_height_r)) + \
            max(max(arr_height_l), max(arr_height_r)) * 0.2

    # AX_LEFT Create bar plot for the mean sentiments
    rects = ax_left.bar(x_pos, arr_height_l, align="center", width=0.8, color=tum_blue, edgecolor="black")
    # ax_left.bar_label(rects, padding=3)    # add label on top of bar
    ax_left.set_xlabel(f"{x_label} [n={str(len(arr_height_l))} bars]", fontsize=10, weight='bold')
    ax_left.set_ylabel(f"{y_label} workload distribution", fontsize=10, weight='bold')
    ax_left.set_title(f"{plot_title}", fontsize=15, weight='bold')
    ax_left.set_xticklabels([])
    ax_left.set_ylim(0, y_max)
    # ax_left.grid(linestyle='--')

    # AX_RIGHT Create bar plot for the mean sentiments
    rects = ax_right.bar(x_pos, arr_height_r, align="center", width=0.8, color=tum_blue, edgecolor="black")
    # ax_right.bar_label(rects, padding=3)    # add label on top of bar
    ax_right.set_xlabel(f"{x_label} [n={str(len(arr_height_r))} bars]", fontsize=10, weight='bold')
    # ax_right.set_ylabel(f"{y_label}", fontsize=10, weight='bold')
    ax_right.set_title(f"{plot_title}", fontsize=15, weight='bold')
    ax_right.set_xticklabels([])
    ax_right.set_ylim(0, y_max)
    # ax_right.grid(linestyle='--')

    if ticks:  # only plot ticks if
        ax_right.set_xticks(x_pos)  # define custom ticks
        ax_right.set_xticklabels(bars, fontsize=5)  # name custom tick labels
        ax_right.xaxis.set_tick_params(rotation=45)
        ax_left.set_xticks(x_pos)  # define custom ticks
        ax_left.set_xticklabels(bars, fontsize=5)  # name custom tick labels
        ax_left.xaxis.set_tick_params(rotation=45)

    fig.tight_layout()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.png")
    fig.show()


def plot_confidence_intervall(list_of_intervalls, list_of_names):
    fig = plt.figure(dpi=400, figsize=(width, height * 4))
    ax = fig.add_subplot(111)
    y_ticks = range(len(list_of_names))
    bar_height = 0.6
    for interval, name, y in zip(list_of_intervalls, list_of_names, y_ticks):
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
    ax.barh(0, width=0, color='white', height=bar_height, hatch='//', edgecolor=tum_blue, label='lower completion time')
    ax.barh(0, width=0, color=tum_blue, height=bar_height, edgecolor=tum_blue, label='higher completion time')
    plt.yticks(y_ticks, list_of_names)
    plt.xlabel('difference of completion time')
    plt.ylabel('configurations')
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_volume_distribution_per_node(volume_per_node, ks, sim_id, name):
    fig, ax = plt.subplots(1, 1, dpi=400, sharex='col', figsize=(width, height))
    ax.bar(range(64), volume_per_node, label=f"p-value: ${ks.pvalue:.2}$")
    plt.axhline(y=np.mean(volume_per_node), color='orange')
    ax.set_xlabel(r"Nodes", weight='bold')
    ax.set_ylabel(r"Summed Volume $[GB]$", weight='bold')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.legend()
    ax.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{workdir}/plots/{name}_node_vol_distribution_sim_{sim_id}.pdf")
    plt.show()

def plot_volume_distribution_per_node_stacked(df_w_volumes, list_ks, sim_id, name):
    fig, ax = plt.subplots(1, 1, dpi=400, sharex='col', figsize=(width, height))
    df_w_volumes = df_w_volumes.sort_values('combined')
    ax.bar(range(64), df_w_volumes['destination'], label=f"combined p-value: ${list_ks[2].pvalue:.2}$", bottom=df_w_volumes['source'])
    ax.bar(range(64), df_w_volumes['source'], label=f"source p-value: ${list_ks[0].pvalue:.2}$")
    plt.axhline(y=np.mean(df_w_volumes['source']), color='black', linestyle='--')
    plt.axhline(y=np.mean(df_w_volumes['combined']), color='black')
    ax.set_xlabel(r"Nodes", weight='bold')
    ax.set_ylabel(r"Summed Volume $[GB]$", weight='bold')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    ax.legend()
    ax.grid(linestyle='--')
    plt.tight_layout()
    plt.savefig(f"{workdir}/plots/{name}_node_vol_distribution_sim_{sim_id}.pdf")
    plt.savefig(f"{workdir}/plots/{name}_node_vol_distribution_sim_{sim_id}.png")
    plt.show()

def plot_histogram_p_values(arr_x, name):
    """
    Args:
        name: on of [source, destination, combined]
        arr_x: array containing #n x values
    Returns:
        histogram plot
    """
    print(f'reject for {len(arr_x[arr_x < 0.1])}/{len(arr_x)}')
    fig = plt.figure(dpi=400, figsize=(width, height))
    alpha = 0.1
    ax = fig.add_subplot(111)
    ax.hist(arr_x, bins='sqrt', density=True, edgecolor='black', facecolor=tum_blue)
    ax.xaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mpl_ticker.ScalarFormatter())
    ax.axvline(alpha, label=fr'$\alpha = {alpha}$', color='orange')
    ax.axvspan(0, alpha, facecolor='None', hatch='//', edgecolor='orange', alpha=0.5,
               label=f'reject $H_0$ for {len(arr_x[arr_x < 0.1])}/{len(arr_x)} \nsimulations')
    ax.legend()
    ax.set_xlabel(f'p-value of uniform distribution of {name}', weight='bold')
    ax.set_ylabel('frequency', weight='bold')
    ax.grid(linestyle='--')
    fig.tight_layout()
    fig.savefig(f"{workdir}/plots/p_values_{name}_node_distributions.pdf")
    fig.savefig(f"{workdir}/plots/p_values_{name}_node_distributions.png")
    fig.show()

def plot_scatter_single_w_colormap(arr_x, arr_y, arr_color, x_label, y_label, color_label,
                                   data_label, plot_title, reg_line=False, legend=False):
    """
    Args:
        legend: show legend [boolean]
        arr_x: array containing #n x values
        arr_y: array containing #n y values
        x_label: X axis label
        y_label: Y axis label
        data_label: legend description of datapoints
        plot_title: header title for plot
        reg_line: plot regression line?
    Returns:
        scatter plot
    """
    bounds = np.unique(arr_color)
    cmap = plt.cm.get_cmap('viridis_r', len(bounds))  # define the colormap

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    x_min = min(arr_x) - abs(min(arr_x)) * 0.2 - 0.02
    x_max = max(arr_x) + abs(max(arr_x)) * 0.2
    y_min = min(arr_y) - abs(min(arr_y)) * 0.2
    y_max = max(arr_y) + abs(max(arr_y)) * 0.2

    fig = plt.figure(dpi=400, figsize=(width, height))

    ax = fig.add_subplot(111)
    sc = ax.scatter(arr_x, arr_y, c=arr_color, label=str(data_label), alpha=0.5, s=8, cmap=cmap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, spacing='proportional', ticks=bounds[bounds % 8 == 0], boundaries=bounds, format='%1i')

    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(color_label, rotation=270)
    # ax.set_yscale('log', base=10)
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(linestyle='--')

    if reg_line:
        # plot lin regression fitted ideal line
        idx = np.isfinite(arr_x) & np.isfinite(arr_y)
        b, m = polyfit(arr_x[idx], arr_y[idx], 1)
        ax.plot(arr_x, b + m * arr_x, '-', color="orange", label='linear approximation')
    if legend:
        ax.legend()
    plt.tight_layout()
    fig.savefig(f"{workdir}/plots/{plot_title}.pdf")
    fig.savefig(f"{workdir}/plots/{plot_title}.png")
    fig.show()

def plot_scatter_single_log(arr_x, arr_y, x_label, y_label,
                            data_label, plot_title, reg_line):
    """
    Args:
        arr_x: array containing #n x values
        arr_y: array containing #n y values
        x_label: X axis label
        y_label: Y axis label
        data_label: legend description of datapoints
        plot_title: header title for plot
        reg_line: plot regression line?
    Returns:
        scatter plot
    """
    xy = np.vstack([arr_x, arr_y])
    z = gaussian_kde(xy)(xy)
    x_min = min(arr_x) - abs(min(arr_x)) * 0.2 - 0.02
    x_max = max(arr_x) + abs(max(arr_x)) * 0.2
    y_min = min(arr_y) - abs(min(arr_y)) * 0.2
    y_max = max(arr_y) + abs(max(arr_y)) * 0.2

    fig = plt.figure(dpi=400, figsize=(width, height))

    ax = fig.add_subplot(111)
    sc = ax.scatter(arr_x, arr_y, c=z, label=str(data_label), alpha=0.5, s=8)
    ax.set_xscale('log', base=10)
    ax.set_title(str(plot_title))
    ax.set_xlabel(str(x_label))
    ax.set_ylabel(str(y_label))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(linestyle='--')

    if reg_line:
        # plot lin regression fitted ideal line
        idx = np.isfinite(arr_x) & np.isfinite(arr_y)
        b, m = polyfit(arr_x[idx], arr_y[idx], 1)
        ax.plot(arr_x, b + m * arr_x, '-', color="orange", label='linear approximation')
    ax.legend()
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    plt.tight_layout()
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.pdf")
    fig.savefig(f"{workdir}/plots/{plot_title}_{time_of_analysis}.png")
    fig.show()