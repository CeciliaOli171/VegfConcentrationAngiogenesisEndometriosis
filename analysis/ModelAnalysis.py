## Libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statistics as stats
import seaborn as sns
  
import matplotlib
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.ticker import LinearLocator 
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from scipy.spatial import ConvexHull
from scipy.stats import linregress
import scipy.integrate as integrate

import pandas as pd

import os, sys
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from ParametersSensitivityRunner import ParametersSensitivityRunner

runner = ParametersSensitivityRunner

## Parameters 
Tcycle = 13.0
time = 2000
Tend = 1989
timestep = 120
TotalTestSourceNb = 40
TotalTestNb = 40
dim = 2
ref_point = 40
dL = 150-ref_point

list_40_seeds = [i+200 for i in range(40)]

# VEGF concentration parameters
c_0 = 0.1
Kc = 1.5e-2
lambdaSprout = 0.08
l_max = 150

# Psprout parameters 
cmax = 0.8
cmin = 0.3
Pmax = 0.98
Pmin = 0.5

n = (1/np.log(cmax/cmin))*np.log((Pmax/Pmin)*(1-Pmin)/(1-Pmax))
K = cmax*((1-Pmax)/Pmax)**(1/n)

# Path to Data  
main_file_path = "/hpc/coli171/Results/PaperAngiogenesisModel2025/PaperModel2025Analysis2D/"
main_file_data_path = "/hpc/coli171/Results/PaperAngiogenesisModel2025/PaperModel2025Analysis2D/Data/FullDataSet.csv" # "/analysis/FullDataSet.csv" 


## Settings
GraphVEGFConcentration = False
GraphVEGFGradient = False 
GraphPsprout = False
GraphPsproutPosition = False

GraphNbBranchesBoxPlots = False
GraphTimeNormFirstVesselTip = True
GraphTimeNormFirstVesselTipTowardLesion = True

GraphNbCellsPlaneBoxPlots = False 
GraphDensityCellInsideLesionComparedToTotalCellsBoxPlots = False

GraphFirstTimeCellReachingLesionBoxPlot = False
GraphErrorFirstTimeCellReachingLesionPositionLesionBoxPlots = False

GraphVascularisation = False


## Plots

# VEGF concentration in steady-state
if GraphVEGFConcentration:
    fig = plt.subplots(figsize = (12,8), dpi = 300)
    x = np.linspace(0,150,1000)
    y1 = [(0.1 - c_0)*np.exp(-Kc*item)+c_0 for item in x]
    y2 = [(0.2 - c_0)*np.exp(-Kc*item)+c_0 for item in x]
    y5 = [(0.5 - c_0)*np.exp(-Kc*item)+c_0 for item in x]
    y10 = [(1.0 - c_0)*np.exp(-Kc*item)+c_0 for item in x]
    plt.plot(x, y1, label= r'$c_{max} = 0.1$', color = 'xkcd:pale blue', linewidth=5.0)
    plt.plot(x, y2, label= r'$c_{max} = 0.2$', color = 'xkcd:lightblue', linewidth=5.0)
    plt.plot(x, y5, label= r'$c_{max} = 0.5$', color = 'xkcd:blue', linewidth=5.0)
    plt.plot(x, y10, label= r'$c_{max} = 1.0$', color = 'xkcd:royal blue', linewidth=5.0)
    plt.xlabel(r'$x$', fontsize=26) 
    plt.ylabel(r'$c(x)$', fontsize=26)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.legend(fontsize = 24)
    plt.show()
    plt.savefig(main_file_path + "Figures/SteadyStateCaseVEGFConcentration.png")

if GraphVEGFGradient:
    fig = plt.subplots(figsize = (12,8), dpi = 300)
    x = np.linspace(0,150,1000)
    y1 = [-Kc*(0.1 - c_0)*np.exp(-Kc*item) for item in x]
    y2 = [-Kc*(0.2 - c_0)*np.exp(-Kc*item) for item in x]
    y5 = [-Kc*(0.5 - c_0)*np.exp(-Kc*item) for item in x]
    y10 = [-Kc*(1.0 - c_0)*np.exp(-Kc*item) for item in x]
    plt.plot(x, y1, label= r'$c_{max} = 0.1$', color = 'xkcd:lavender', linewidth=5.0)
    plt.plot(x, y2, label= r'$c_{max} = 0.2$', color = 'xkcd:wisteria', linewidth=5.0)
    plt.plot(x, y5, label= r'$c_{max} = 0.5$', color='xkcd:purple', linewidth=5.0)
    plt.plot(x, y10, label=r'$c_{max} = 1.0$', color='xkcd:royal purple', linewidth=5.0)
    plt.xlabel(r'$x$', fontsize=26) 
    plt.ylabel(r'$\nabla c(x)[0]$', fontsize=26)
    plt.xticks(fontsize = 24)
    plt.yticks(fontsize = 24)
    plt.legend(fontsize = 24)

    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax = plt.gca()
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_position((0.0, 1.02))  # Move it to the top
    offset_text.set_horizontalalignment('left')
    offset_text.set_fontsize(24)

    plt.show()
    plt.savefig(main_file_path + "Figures/SteadyStateCaseVEGFGradient.png")


# Psprout
if GraphPsprout:
    fig = plt.subplots(figsize = (12,8), dpi = 300)
    sourceterm = np.linspace(0,1.0,100)
    sourcetermticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    Psprout = [lambdaSprout*(item)**n/(K**n + (item)**n) for item in sourceterm]
    plt.plot(sourceterm, Psprout, color='xkcd:orange', linewidth=5.0)
    plt.xlabel(r'$c$', fontsize=26) 
    plt.ylabel(r'$P_{sprout}$', fontsize=26)
    plt.xticks(sourcetermticks, fontsize = 24)
    plt.yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], fontsize = 24)
    plt.show()
    plt.savefig(main_file_path + "Figures/SteadyStateCasePsproutSourceTerm.png")

if GraphPsproutPosition:
    fig = plt.subplots(figsize = (12,8), dpi = 300)
    x = np.linspace(0, 150, 1000)
    Psprout1 = [lambdaSprout*c_0**n/(K**n + c_0**n) for item in x]
    Psprout2 = [lambdaSprout*((0.2 - c_0)*np.exp(-Kc*item)+c_0)**n/(K**n + ((0.1 - c_0)*np.exp(-Kc*item)+c_0)**n) for item in x]
    Psprout5 = [lambdaSprout*((0.5 - c_0)*np.exp(-Kc*item)+c_0)**n/(K**n + ((0.5 - c_0)*np.exp(-Kc*item)+c_0)**n) for item in x]
    Psprout10 = [lambdaSprout*((1.0 - c_0)*np.exp(-Kc*item)+c_0)**n/(K**n + ((1.0 - c_0)*np.exp(-Kc*item)+c_0)**n) for item in x]
    plt.plot(x, Psprout1, label=r'$c_{max} = 0.1$', color='xkcd:beige', linewidth=5.0)
    plt.plot(x, Psprout2, label=r'$c_{max} = 0.2$', color='xkcd:tan', linewidth=5.0)
    plt.plot(x, Psprout5, label=r'$c_{max} = 0.5$', color='xkcd:light brown', linewidth=5.0)
    plt.plot(x, Psprout10, label=r'$c_{max} = 1.0$', color='xkcd:brown', linewidth=5.0)
    plt.xlabel(r'$x$', fontsize=26) 
    plt.ylabel(r'$P_{sprout}$', fontsize=26)
    plt.xticks(fontsize = 24)
    plt.yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], fontsize = 24)
    plt.legend(fontsize = 24)
    plt.show()
    plt.savefig(main_file_path + "Figures/SteadyStateCasePsproutPosition.png")


# Number of branches 
if GraphNbBranchesBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    data_file = pd.read_csv(main_file_data_path)

    glob_vals  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    local_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

    hyp_col = 'vegf hyp (total number branches)'
    src_col = 'source term (total number branches)'
    pos_col = 'average vegf (total number branches)'

    df = data_file.copy()
    src = df[src_col].round(1)

    is_global = (df[hyp_col] == 'Global Hypothesis') & src.isin(glob_vals)
    is_local  = (df[hyp_col] == 'Local Hypothesis') & src.isin(local_vals)

    df[pos_col] = np.select(
        [is_global, is_local],
        [src, (0.5 * src + 0.05).round(2)],
        default=np.nan,
    )

    new_df = df.dropna(subset=[pos_col]).copy()
    new_df[pos_col] = new_df[pos_col].round(1)      

    palette = {'Global Hypothesis': "#008080", 'Local Hypothesis': "#C00000"}
    order = ['Local Hypothesis', 'Global Hypothesis']

    sns.boxplot(data=new_df, x=pos_col, y='total number branches', hue=hyp_col, hue_order=order, palette=palette, fill=False, whis=np.inf, showcaps=True, showfliers=False, native_scale=True, ax=ax)
    sns.stripplot(data=new_df, x=pos_col, y='total number branches', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=5, alpha=0.6, native_scale=True, legend=False, ax=ax)

    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.25, 1.05)
    sns.move_legend(ax, "best", title=None, fontsize=20) 
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Number of Branches', fontsize = 20)
    plt.savefig(main_file_path + "Figures/NbBranchesBoxPlots.png")

if GraphTimeNormFirstVesselTip:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    df = pd.read_csv(main_file_data_path)

    dt = 13.0
    src_col = 'source term (network velocity)'
    col = 'network velocity'
    hyp_col = 'vegf hyp (network velocity)'

    palette = {'Global hypothesis': "#008080", 'Local hypothesis': "#C00000"}
    order = ['Local hypothesis', 'Global hypothesis']

    df["hypothesis"] = df[hyp_col].str.replace(" Hypothesis", " hypothesis", regex=False)   
    rows = []
    for hyp, s in zip(df["hypothesis"], df[col]):
        y = runner.parse(s)
        rows.append(pd.DataFrame({"time": np.arange(len(y)) * dt, "norm": np.abs(y)*10, "hypothesis": hyp}))
    long_df = pd.concat(rows, ignore_index=True)

    sns.lineplot(data=long_df, x="time", y="norm", hue="hypothesis", hue_order=order, palette=palette, estimator="mean", errorbar=("pi", 100), linewidth=2, ax=ax)

    ax.set_xlim(0, 2000)
    ax.set_xlabel(r'Time (h)', fontsize = 26)
    ax.set_ylabel(r'Network velocity $(\mu m \cdot h^{-1})$', fontsize = 20)
    ax.tick_params(axis = 'both', labelsize = 20)
    sns.move_legend(ax, "best", title=None, fontsize=14)
    fig.savefig(main_file_path + "Figures/TimeReachingFirstVesselTip.png")

if GraphTimeNormFirstVesselTipTowardLesion:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    df = pd.read_csv(main_file_data_path)

    dt = 13.0
    src_col = 'source term (network velocity toward lesion)'
    col = 'network velocity toward lesion'
    hyp_col = 'vegf hyp (network velocity toward lesion)'

    palette = {'Global hypothesis': "#008080", 'Local hypothesis': "#C00000"}
    order = ['Local hypothesis', 'Global hypothesis']

    df["hypothesis"] = df[hyp_col].str.replace(" Hypothesis", " hypothesis", regex=False)   
    rows = []
    for hyp, s in zip(df["hypothesis"], df[col]):
        y = runner.parse(s)
        rows.append(pd.DataFrame({"time": np.arange(len(y)) * dt, "norm": np.abs(y)*10, "hypothesis": hyp}))
    long_df = pd.concat(rows, ignore_index=True)

    sns.lineplot(data=long_df, x="time", y="norm", hue="hypothesis", hue_order=order, palette=palette, estimator="mean", errorbar=("pi", 100), linewidth=2, ax=ax)

    ax.set_xlim(0, 2000)
    ax.set_xlabel(r'Time (h)', fontsize = 26)
    ax.set_ylabel(r'Network velocity toward lesion $(\mu m \cdot h^{-1})$', fontsize = 20)
    ax.tick_params(axis = 'both', labelsize = 20)
    sns.move_legend(ax, "best", title=None, fontsize=14)
    fig.savefig(main_file_path + "Figures/TimeReachingFirstVesselTipTowardLesion.png")


# Number of cells after a certain plane depending on source term for different Psprout 
if GraphNbCellsPlaneBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    data_file = pd.read_csv(main_file_data_path)

    glob_vals  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    local_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

    hyp_col = 'vegf hyp (cells inside lesion)'
    src_col = 'source term (cells inside lesion)'
    pos_col = 'average vegf (cells inside lesion)'

    df = data_file.copy()
    src = df[src_col].round(1)

    is_global = (df[hyp_col] == 'Global Hypothesis') & src.isin(glob_vals)
    is_local  = (df[hyp_col] == 'Local Hypothesis') & src.isin(local_vals)

    df[pos_col] = np.select(
        [is_global, is_local],
        [src, (0.5 * src + 0.05).round(2)],
        default=np.nan,
    )

    new_df = df.dropna(subset=[pos_col]).copy()
    new_df[pos_col] = new_df[pos_col].round(1)
    zero = new_df['cells inside lesion'] == 0      

    palette = {'Global Hypothesis': "#008080", 'Local Hypothesis': "#C00000"}
    order = ['Local Hypothesis', 'Global Hypothesis']

    sns.boxplot(data=new_df, x=pos_col, y='cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, fill=False, whis=np.inf, showcaps=True, showfliers=False, native_scale=True, ax=ax)
    sns.stripplot(data=new_df[~zero], x=pos_col, y='cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=5, alpha=0.6, native_scale=True, legend=False, ax=ax)
    sns.stripplot(data=new_df[zero], x=pos_col, y='cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=6, alpha=0.6, marker='x', linewidth=1.9, edgecolor='face', native_scale=True, legend=False, ax=ax)

    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.25, 1.05)
    sns.move_legend(ax, "best", title=None, fontsize=20) 
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Number of Vessel Segments Inside Lesion', fontsize = 20)    
    plt.savefig(main_file_path + "Figures/CellsAfterPlaneBoxPlots.png")

if GraphDensityCellInsideLesionComparedToTotalCellsBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    data_file = pd.read_csv(main_file_data_path)

    glob_vals  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    local_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

    hyp_col = 'vegf hyp (% network cells inside lesion)'
    src_col = 'source term (% network cells inside lesion)'
    pos_col = 'average vegf (% network cells inside lesion)'

    df = data_file.copy()
    src = df[src_col].round(1)

    is_global = (df[hyp_col] == 'Global Hypothesis') & src.isin(glob_vals)
    is_local  = (df[hyp_col] == 'Local Hypothesis') & src.isin(local_vals)

    df[pos_col] = np.select([is_global, is_local], [src, (0.5 * src + 0.05).round(2)], default=np.nan)

    new_df = df.dropna(subset=[pos_col]).copy()
    new_df[pos_col] = new_df[pos_col].round(1)  
    zero = new_df['cells inside lesion'] == 0    

    palette = {'Global Hypothesis': "#008080", 'Local Hypothesis': "#C00000"}
    order = ['Local Hypothesis', 'Global Hypothesis']

    sns.boxplot(data=new_df, x=pos_col, y='percentage of network cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, fill=False, whis=np.inf, showcaps=False, showfliers=False, native_scale=True, ax=ax)
    sns.stripplot(data=new_df[~zero], x=pos_col, y='percentage of network cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=5, alpha=0.6, native_scale=True, legend=False, ax=ax)
    sns.stripplot(data=new_df[zero], x=pos_col, y='percentage of network cells inside lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=6, alpha=0.6, marker='x', linewidth=1.9, edgecolor='face', native_scale=True, legend=False, ax=ax)

    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.25, 1.05)
    sns.move_legend(ax, "best", title=None, fontsize=20) 
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel("Proportion of Networks' Vessel Segments Inside Lesion (%)", fontsize = 17)
    plt.savefig(main_file_path + "Figures/CellDensityInsideLesionComparedToTotalCells.png")


# Times
if GraphFirstTimeCellReachingLesionBoxPlot:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    data_file = pd.read_csv(main_file_data_path)

    glob_vals  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    local_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

    hyp_col = 'vegf hyp (time reaching lesion)'
    src_col = 'source term (time reaching lesion)'
    pos_col = 'average vegf (time reaching lesion)'

    df = data_file.copy()
    src = df[src_col].round(1)

    is_global = (df[hyp_col] == 'Global Hypothesis') & src.isin(glob_vals)
    is_local  = (df[hyp_col] == 'Local Hypothesis') & src.isin(local_vals)

    df[pos_col] = np.select(
        [is_global, is_local],
        [src, (0.5 * src + 0.05).round(2)],
        default=np.nan,
    )

    new_df = df.dropna(subset=[pos_col]).copy()
    new_df[pos_col] = new_df[pos_col].round(1)  
    zero = new_df['time reaching lesion'] == 2000

    palette = {'Global Hypothesis': "#008080", 'Local Hypothesis': "#C00000"}
    order = ['Local Hypothesis', 'Global Hypothesis']

    sns.boxplot(data=new_df, x=pos_col, y='time reaching lesion', hue=hyp_col, hue_order=order, palette=palette, fill=False, whis=np.inf, showcaps=True, showfliers=False, native_scale=True, ax=ax)
    sns.stripplot(data=new_df[~zero], x=pos_col, y='time reaching lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=5, alpha=0.6, native_scale=True, legend=False, ax=ax)
    sns.stripplot(data=new_df[zero], x=pos_col, y='time reaching lesion', hue=hyp_col, hue_order=order, palette=palette, dodge=True, size=6, alpha=0.6, marker='x', linewidth=1.9, edgecolor='face', native_scale=True, legend=False, ax=ax)

    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlim(0.25, 1.05)
    sns.move_legend(ax, "best", title=None, fontsize=20) 
    ax.tick_params(axis = 'both', labelsize = 20)
    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Time For Network to Reach Lesion (hours)', fontsize = 20)
    # ax.set_yticks([1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
    # ax.set_ylim(1200, 2015)
    plt.savefig(main_file_path + "Figures/TimeFirstCellReachingLesionBoxPlot" + ".png")

if GraphErrorFirstTimeCellReachingLesionPositionLesionBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (10,10), dpi = 300, layout='constrained')

    k_average = 10
    k_analyticalapproxpde = 2*k_average-1
    k_constant = k_average

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_width = 2
    group_gap = 0.1

    DistanceLesionVessel = [40, 30, 20]
    DistanceLesionVesselTicks = [110, 120, 130]
    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pos_constant = [i + 0.5 + box_width/2 for i in DistanceLesionVesselTicks]
    pos_analyticalapproxpde = [i - 0.5 - box_width/2 for i in DistanceLesionVesselTicks]
    position_total = []

    timefirstreachinglesion_analyticalapproxpde_data = []
    timefirstreachinglesion_constant_data = []
    timefirstreachinglesion_total = []

    vegfhyp_total = []

    timereachinglesion_analyticalapproxpde = []
    timereachinglesion_constant = []

    # loop over all the files 
    for m in range(3):
        i = DistanceLesionVessel[m]
        time_analyticalapproxpde = []
        time_constant = []

        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k_analyticalapproxpde)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k_constant)

            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, i, dim)
            timeseed_constant = runner.TimeFirstReachingPlane(file_nodescoordinates_constant, i, dim)

            time_analyticalapproxpde.append(timeseed_analyticalapproxpde)

            timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, i, dim))
            position_total.append(pos_analyticalapproxpde[m])
            vegfhyp_total.append("Local Hypothesis")

            time_constant.append(timeseed_constant)
            timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_constant, i, dim))
            position_total.append(pos_constant[m])
            vegfhyp_total.append("Global Hypothesis")

        timereachinglesion_analyticalapproxpde.append(time_analyticalapproxpde)
        timereachinglesion_constant.append(time_constant)

        df = pd.DataFrame({'position':position_total, 'timefirstreachinglesion':timefirstreachinglesion_total,'vegfhyp':vegfhyp_total})
        df = df.copy()
        df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
        positions_sorted = np.sort(df['xpos'].unique())
        min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.5
        jitter_width = min(0.25 * min_gap, 0.5)  # cap jitter so points don't overlap neighbors
        rng = np.random.default_rng(42)
        df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

        sns.scatterplot(data=df, x='xpos_jit', y='timefirstreachinglesion', hue='vegfhyp', alpha=1.0, s=40, palette=["#C00000", "#008080"])

    width = 2
    position_constant = [item + 0.5 + width/2 for item in DistanceLesionVesselTicks]
    position_analyticalapproxpde = [item - 0.5 - width/2 for item in DistanceLesionVesselTicks]

    box_colors = ['#C00000', '#008080']
    bp_analyticalapproxpde = ax.boxplot(timereachinglesion_analyticalapproxpde, positions = position_analyticalapproxpde, widths=width, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, whis=np.inf) 
    runner.style_boxplot(bp_analyticalapproxpde, color_analyticalapproxpde)
    bp_constant = ax.boxplot(timereachinglesion_constant, positions = position_constant, widths=width, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, whis=np.inf) 
    runner.style_boxplot(bp_constant, color_constant)

    ax.set_xlabel(r'Distance Lesion / Vessel ($\mu m$)', fontsize = 24)
    ax.set_ylabel(r'Time For Network To Reach Lesion (hours)', fontsize = 24)
    custom_lines = [plt.Line2D([0], [0], color=box_colors[0], lw=4), plt.Line2D([0], [0], color=box_colors[1], lw=4)]
    ax.legend(custom_lines, ['Local Hypothesis', 'Global Hypothesis'], loc='lower right', fontsize = 24)  
    ax.set_xticks(DistanceLesionVesselTicks)
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.set_ylim(1400, 2025)
    plt.savefig(main_file_path + "Figures/ErrorFirstTimeCellReachingLesionPositionLesion" + str(k_average) + ".png")


# Vascularisation
if GraphVascularisation:
    # plot 
    fig, ax = plt.subplots(figsize = (12,8), dpi = 300, layout='constrained')

    # loop over all the files 
    vascularisation_analyticalapproxpde = []
    vascularisation_constant = []

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    source_total = []
    vascularisation_total = []
    vegfhyp_total = []

    ref_point_test = 20
    ref_point_test_str = 150-ref_point_test

    for k in range(1,11):
        time_constant = []
        time_analyticalapproxpde = []
        vascularisationvalue_constant = 0
        vascularisationvalue_analyticalapproxpde = 0
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point_test, dim)
            timeseed_constant = runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point_test, dim)

            # remove the not vascularised network
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    time_analyticalapproxpde.append(timeseed_analyticalapproxpde)
                    vascularisationvalue_analyticalapproxpde += 1
            if timeseed_constant < 1989.0:
                time_constant.append(timeseed_constant)
                vascularisationvalue_constant += 1

        vascularisation_analyticalapproxpde.append(vascularisationvalue_analyticalapproxpde*100/TotalTestNb)
        source_total.append(sourceterm[k-1])
        vascularisation_total.append(vascularisationvalue_analyticalapproxpde*100/TotalTestNb)
        vegfhyp_total.append("Local Hypothesis")
        vascularisation_constant.append(vascularisationvalue_constant*100/TotalTestNb)
        source_total.append(sourceterm[k-1])
        vascularisation_total.append(vascularisationvalue_constant*100/TotalTestNb)
        vegfhyp_total.append("Global Hypothesis")

    for k in range(11,21):
        time_analyticalapproxpde = []
        vascularisationvalue_analyticalapproxpde = 0
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point_test, dim)

            # remove the not vascularised network
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    time_analyticalapproxpde.append(timeseed_analyticalapproxpde)
                    vascularisationvalue_analyticalapproxpde += 1

        vascularisation_analyticalapproxpde.append(vascularisationvalue_analyticalapproxpde*100/TotalTestNb)
        source_total.append(sourceterm[k-1])
        vascularisation_total.append(vascularisationvalue_analyticalapproxpde*100/TotalTestNb)
        vegfhyp_total.append("Local Hypothesis")

    SourceTermTicks = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    barWidth = 0.025
    position_constant = [item + barWidth/2 for item in SourceTermTicks]
    position_analyticalapproxpde = [item - barWidth/2 for item in SourceTermTicks]

    vascularisation_constant_notall = [vascularisation_constant[j] for j in range(2,10)]
    vascularisation_analyticalapproxpde_notall = [vascularisation_analyticalapproxpde[j] for j in range(4,20)]
    vascularisation_analyticalapproxpde_even = [vascularisation_analyticalapproxpde[2*i] for i in range(8)]

    bp_analyticalapproxpde = ax.bar(position_analyticalapproxpde, vascularisation_analyticalapproxpde_even, width=barWidth, color = '#C00000', label = 'Local Hypothesis') 
    bp_constant = ax.bar(position_constant, vascularisation_constant_notall, width=barWidth, color = '#008080', label = 'Global Hypothesis') 

    df = pd.DataFrame({'source':source_total, 'timefirstreachinglesion':vascularisation_total,'vegfhyp':vegfhyp_total})
    df.to_csv(main_file_path + "Data/VascularisationDistance" + str(ref_point_test_str), index=False)
   
    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Realisations with Lesion Vascularised \n (%)', fontsize = 26)
    ax.legend(loc='lower left', fontsize = 24)
    ax.set_xticks(SourceTermTicks)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis = 'both', labelsize = 24)
    plt.savefig(main_file_path + "Figures/VascularisationBarPlot" + str(ref_point_test) + ".png")
