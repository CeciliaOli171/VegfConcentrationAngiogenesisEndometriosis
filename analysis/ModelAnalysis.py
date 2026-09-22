## Libraries 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import statistics as stats
import seaborn as sns
  
from matplotlib import cm
from matplotlib.ticker import LinearLocator 
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
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
Tend = 2000
timestep = 120
TotalTestSourceNb = 40
TotalTestNb = 40-12
dim = 2
ref_point = 40
dL = 110

list_40_seeds = [i+200 for i in range(40)]
list_40_seeds.remove(220) # for now 
list_40_seeds.remove(221) # for now 
list_40_seeds.remove(222) # for now 
list_40_seeds.remove(227) # for now 
list_40_seeds.remove(229) # for now 
list_40_seeds.remove(230) # for now 
list_40_seeds.remove(233) # for now 
list_40_seeds.remove(235) # for now 
list_40_seeds.remove(236) # for now 
list_40_seeds.remove(237) # for now 
list_40_seeds.remove(238) # for now 
list_40_seeds.remove(239) # for now 

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

# Paths 
# HPC
main_file_path = "/hpc/coli171/Results/PaperAngiogenesisModel2025/PaperModel2025Analysis2D/" 


## Settings
TestBaseline = False

GraphVEGFConcentration = False
GraphVEGFGradient = False 
GraphPsprout = False
GraphPsproutPosition = False

GraphAnastomosisVesselTips = True
GraphTimeNormFirstVesselTip = False
GraphNbBranchesBoxPlots = False
GraphNbBranchesANDExpectedNbBranchesSameGraph = False

GraphNbCellsPlaneBoxPlots = False 
GraphDensityCellInsideLesionComparedToTotalCellsBoxPlots = False

GraphFirstTimeCellReachingLesionBoxPlot = False
GraphErrorFirstTimeCellReachingLesionPositionLesionBoxPlots = False

GraphVascularisation = False


## Functions 
def Psprout(x, c_max, Kc, GlobalOrLocal):
    if GlobalOrLocal == 1:
        c_0 = 0.1
        vegf = (c_max-c_0)*np.exp(-Kc*x)+c_0
        Psprout = lambdaSprout*vegf**n/(K**n + vegf**n)
    else:
        Psprout = lambdaSprout*c_max**n/(K**n + c_max**n)

    return Psprout

def Panastomosis(GlobalOrLocal, Pcomputation):
    if GlobalOrLocal == 1:
        Panastomosis = Pcomputation
    else:
        Panastomosis = Pcomputation

    return Panastomosis

def ExpectationSprouting(x, c_max, Kc, GlobalOrLocal):
    # case of Psprout depending on VEGF
    Sk = Psprout(l_max-0.3*x, c_max, Kc, GlobalOrLocal) # vessel segment division hyp
    #Sk = Psprout(l_max-0.3*0.5*x, c_max, Kc, GlobalOrLocal) # tip cell division hyp
    return Sk 

def ExpectationAnastomosis(x, c_max, Kc, GlobalOrLocal, Pcomputation):
    #Ak = Panastomosis(GlobalOrLocal, Pcomputation)*Psprout(l_max-0.3*x, c_max, Kc, GlobalOrLocal) # vessel segment division hyp
    Ak = Panastomosis(GlobalOrLocal, Pcomputation)
    return Ak 

def ExpectationBranches(t, c_max, Kc, GlobalOrLocal, Pcomputation):
    v = 0.68
    Bn = 2*(1+ExpectationSprouting(v*t, c_max, Kc, GlobalOrLocal)-ExpectationAnastomosis(v*t, c_max, Kc, GlobalOrLocal, Pcomputation))**t - 1
    return Bn


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


# Number of branches (computational and analytical) 
if GraphAnastomosisVesselTips:
    # plot 
    fig, ax = plt.subplots(1, 2, figsize = (12,8), dpi = 300, layout='constrained')

    # colours and colourmaps
    cmap_analyticalapproxpde = LinearSegmentedColormap.from_list('red_transparent', [(1.0, 0.0, 0.0, 0.0), (0.75, 0.0, 0.0, 1.0)])
    cmap_constant = LinearSegmentedColormap.from_list('gold_transparent', [(1.0, 1.0, 1.0, 1.0), (0.0, 0.5, 0.5, 1.0)])

    # we represent the percentage of tip cells going through anastomosis at each time steps 
    # we have two different plots: one for the constant case and one for the steady-state of PDE case 
    # for each cmax, we represent the percentage of the ratio anastomosis/tip cells previous time for each time step (except t=0)
    # one cmax = one colour
    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = np.linspace(0, Tend, int(Tend/13)+1)

    # Storage for all data points across all (j, k)
    all_tip_cells_analyticalapproxpde = []
    all_anastomosis_analyticalapproxpde = []
    all_tip_cells_constant = []
    all_anastomosis_constant = []

    for k in range(1,11):
        anastomosis_analyticalapproxpde = []
        anastomosis_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"
            file_anastomosis_constant = file_path_constant + "/results_from_time_0/results.vizanastomosis"

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_cellmutation_constant = file_path_constant + "/results_from_time_0/results.vizmutationstates"

            tipcells_analyticalapproxpde = runner.VesselTipsPerTimeStep(file_cellmutation_analyticalapproxpde)
            anastomosis_analyticalapproxpde = runner.AnastomosisPerTimeStep(file_anastomosis_analyticalapproxpde)
            tipcells_constant = runner.VesselTipsPerTimeStep(file_cellmutation_constant)
            anastomosis_constant = runner.AnastomosisPerTimeStep(file_anastomosis_constant)

            all_tip_cells_analyticalapproxpde.extend(tipcells_analyticalapproxpde)
            all_anastomosis_analyticalapproxpde.extend(anastomosis_analyticalapproxpde)
            all_tip_cells_constant.extend(tipcells_constant)
            all_anastomosis_constant.extend(anastomosis_constant)

    for k in range(11,21):
        anastomosis_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"

            tipcells_analyticalapproxpde = runner.VesselTipsPerTimeStep(file_cellmutation_analyticalapproxpde)
            anastomosis_analyticalapproxpde = runner.AnastomosisPerTimeStep(file_anastomosis_analyticalapproxpde)

            all_tip_cells_analyticalapproxpde.extend(tipcells_analyticalapproxpde)
            all_anastomosis_analyticalapproxpde.extend(anastomosis_analyticalapproxpde)

    all_tip_cells_constant = np.round(np.array(all_tip_cells_constant, dtype=np.float64)).astype(np.int64)
    all_anastomosis_constant = np.round(np.array(all_anastomosis_constant, dtype=np.float64)).astype(np.int64)
    all_tip_cells_analyticalapproxpde = np.round(np.array(all_tip_cells_analyticalapproxpde, dtype=np.float64)).astype(np.int64)
    all_anastomosis_analyticalapproxpde = np.round(np.array(all_anastomosis_analyticalapproxpde, dtype=np.float64)).astype(np.int64)

    idx_constant = np.argsort(all_tip_cells_constant)
    idx_analyticalapproxpde = np.argsort(all_tip_cells_analyticalapproxpde)

    all_tip_cells_sorted_constant = all_tip_cells_constant[idx_constant]
    all_anastomosis_sorted_constant = all_anastomosis_constant[idx_constant]
    all_tip_cells_sorted_analyticalapproxpde = all_tip_cells_analyticalapproxpde[idx_analyticalapproxpde]
    all_anastomosis_sorted_analyticalapproxpde = all_anastomosis_analyticalapproxpde[idx_analyticalapproxpde]

    ux_constant, inv_constant = np.unique(all_tip_cells_sorted_constant, return_inverse=True)
    means_constant = np.array([all_anastomosis_sorted_constant[inv_constant == i].mean() for i in range(len(ux_constant))])
    ux_analyticalapproxpde, inv_analyticalapproxpde = np.unique(all_tip_cells_sorted_analyticalapproxpde, return_inverse=True)
    means_analyticalapproxpde = np.array([all_anastomosis_sorted_analyticalapproxpde[inv_analyticalapproxpde == i].mean() for i in range(len(ux_analyticalapproxpde))])

    # ---- CONSTANT DATA: square-binned colormap ----
    # Define ranges like your hexbin 'extent'
    xmin1, xmax1 = 0, max(all_tip_cells_constant)
    ymin1, ymax1 = 0, max(all_anastomosis_constant)

    # Compute 2D histogram with square bins (gridsize -> bins)
    H1, xedges1, yedges1 = np.histogram2d(
        all_tip_cells_constant,
        all_anastomosis_constant,
        bins=(xmax1,ymax1),                             # match gridsize=(30,9)
        range=[[xmin1, xmax1], [ymin1, ymax1]]
    )

    # Apply mincnt=1 by masking zeros
    H1_masked = np.ma.masked_where(H1 < 1, H1)

    # Draw squares via pcolormesh, log-normalized like bins='log'
    mesh1 = ax[1].pcolormesh(
        xedges1, yedges1, H1_masked.T,           # transpose to align orientation
        cmap=cmap_constant,
        norm=LogNorm(),                          # equivalent to bins='log'
        shading='auto'                           # fills full squares
    )

    # ---- ANALYTICAL APPROX PDE DATA: square-binned colormap ----
    xmin0, xmax0 = 0, max(all_tip_cells_analyticalapproxpde)
    ymin0, ymax0 = 0, max(all_anastomosis_analyticalapproxpde)

    H0, xedges0, yedges0 = np.histogram2d(
        all_tip_cells_analyticalapproxpde,
        all_anastomosis_analyticalapproxpde,
        bins=(xmax0, ymax0),                            # match gridsize=(10,3)
        range=[[xmin0, xmax0], [ymin0, ymax0]]
    )

    H0_masked = np.ma.masked_where(H0 < 1, H0)

    mesh0 = ax[0].pcolormesh(
        xedges0, yedges0, H0_masked.T,
        cmap=cmap_analyticalapproxpde,
        norm=LogNorm(),
        shading='auto'
    )

    weights_constant = []
    for k in range(len(all_anastomosis_constant)):
        if (all_tip_cells_constant[k] <= 0):
            weights_constant.append(1)
        elif (all_anastomosis_constant[k] == 0):
            weights_constant.append(1)
        else:
            weights_constant.append(4)
    sqrt_weights_constant = np.sqrt(weights_constant)

    all_tip_cells_constant_weighted = all_tip_cells_constant * sqrt_weights_constant
    all_anastomosis_constant_weighted = all_anastomosis_constant * sqrt_weights_constant

    m_constant, p_constant, r_value, p_value, std_err = linregress(
        all_tip_cells_constant_weighted,
        all_anastomosis_constant_weighted
    )
    ax[1].plot(
        np.linspace(0, xmax1, xmax1),
        [m_constant * item + p_constant for item in np.linspace(0, xmax1, xmax1)],
        '--k', linewidth=5.0, label=fr'$P_{{a, ECM}} = {m_constant:.3g}$'
    )

    legend_box1 = patches.FancyBboxPatch(
        (0.05, 0.73), 0.75, 0.22, transform=ax[1].transAxes,
        boxstyle="round,pad=0.02", facecolor='white',
        edgecolor='black', linewidth=1, alpha=0.25
    )
    ax[1].add_patch(legend_box1)
    ax[1].text(0.55, 0.92, 'Density',
            transform=ax[1].transAxes, fontsize=18,
            verticalalignment='center', horizontalalignment='right')
    mean_marker = Line2D([0.2], [0.83], marker='o', markersize=5, markerfacecolor='xkcd:black',markeredgecolor='xkcd:black', linestyle='None', transform=ax[0].transAxes)
    mean_marker.set_clip_path(legend_box1.get_path(), transform=legend_box1.get_transform())
    ax[1].add_line(mean_marker)
    ax[1].text(0.67, 0.83, 'Mean values', transform=ax[1].transAxes, fontsize=18, verticalalignment='center', horizontalalignment='right')
    line = Line2D([0.15, 0.15+0.1], [0.75, 0.75], color='xkcd:black', ls='--', lw=3, transform=ax[0].transAxes)
    line.set_clip_path(legend_box1.get_path(), transform=legend_box1.get_transform())
    ax[1].add_line(line)
    ax[1].text(0.8, 0.75, 'Linear Regression', transform=ax[1].transAxes, fontsize=18, verticalalignment='center', horizontalalignment='right')

    cax1 = inset_axes(ax[1], width="100%", height="45%", loc='upper left', bbox_to_anchor=(0.075, 0.85, 0.25, 0.1), bbox_transform=ax[1].transAxes, borderpad=0)
    cb1 = fig.colorbar(mesh1, cax=cax1, orientation='horizontal')  # use mesh0
    cb1.ax.tick_params(labelsize=18)
    cb1.set_label('')

    legend_box0 = patches.FancyBboxPatch(
        (0.05, 0.87), 0.5, 0.08, transform=ax[0].transAxes,
        boxstyle="round,pad=0.02", facecolor='white',
        edgecolor='black', linewidth=1, alpha=0.25
    )
    ax[0].add_patch(legend_box0)
    ax[0].text(0.55, 0.92, 'Density', transform=ax[0].transAxes, fontsize=18, verticalalignment='center', horizontalalignment='right')

    cax0 = inset_axes(ax[0], width="100%", height="45%", loc='upper left', bbox_to_anchor=(0.075, 0.85, 0.25, 0.1), bbox_transform=ax[0].transAxes, borderpad=0)
    cb0 = fig.colorbar(mesh0, cax=cax0, orientation='horizontal')  # use mesh1
    cb0.ax.tick_params(labelsize=18)

    ax[1].scatter(ux_constant.tolist(), means_constant, label = 'Mean anastomosis values', color = 'xkcd:black')
    #ax[1].plot(ux_analyticalapproxpde.tolist(), means_analyticalapproxpde, label = 'Mean anastomosis values', color = 'xkcd:black')

    print(m_constant)

    ax[1].set_ylim(0, ymax1+1)
    ax[0].set_ylim(0, ymax0+1)
    ax[0].set_title('Local Hypothesis', fontsize = 24)
    ax[1].set_title('Global Hypothesis', fontsize = 24)
    ax[0].set_xlabel('Number of Vessel Tips', fontsize = 26)
    ax[1].set_xlabel('Number of Vessel Tips', fontsize = 26)
    ax[0].set_ylabel('Anastomosis Events', fontsize = 26)
    ax[0].tick_params(axis = 'both', labelsize = 24)
    ax[1].tick_params(axis = 'both', labelsize = 24)
    plt.savefig(main_file_path + "Figures/AnastomosisVesselTips.png")

if GraphTimeNormFirstVesselTip:
    # plot 
    fig, ax = plt.subplots(figsize = (12,8), dpi = 300, layout='constrained')

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = np.linspace(0, Tend, int(Tend/Tcycle)+1)

    normfirsttipcell_analyticalapproxpde = []
    normfirsttipcell_constant = []

    slopes_velocity = []

    for k in range(1,11):
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_cellmutation_constant = file_path_constant + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            valuenormfirsttipcell_analyticalapproxpde = runner.NormFirstVesselTip(file_nodescoordinates_analyticalapproxpde, file_cellmutation_analyticalapproxpde, dim)
            valuenormfirsttipcell_constant = runner.NormFirstVesselTip(file_nodescoordinates_constant, file_cellmutation_constant, dim)

            slope_analyticalapproxpde, intercept = np.polyfit(t, valuenormfirsttipcell_analyticalapproxpde, 1)
            slope_constant, intercept = np.polyfit(t, valuenormfirsttipcell_constant, 1)
            slopes_velocity.append(slope_analyticalapproxpde)
            slopes_velocity.append(slope_constant)

            # scatter plot 
            if(j == 210 and k == 3):
                ax.plot(t, valuenormfirsttipcell_analyticalapproxpde, color='#C00000', label = 'Local Hypothesis')
                ax.plot(t, valuenormfirsttipcell_constant, color='#008080', label = 'Global Hypothesis')
            else:
                ax.plot(t, valuenormfirsttipcell_analyticalapproxpde, color='#C00000')
                ax.plot(t, valuenormfirsttipcell_constant, color='#008080')

    velocity_value = np.mean(slopes_velocity)
    velocity = [item*velocity_value for item in t]
    ax.plot(t, velocity, color='xkcd:black', label=fr'$v = {velocity_value:.2g} \; \mu m \cdot h^{-1}$', linewidth = 5.0)

    ax.set_xlabel(r'$t$ (hours)', fontsize = 30)
    ax.set_ylabel(r'Distance $(\mu m)$', fontsize = 26)
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.legend(loc='upper left', fontsize = 24)
    plt.show()
    plt.savefig(main_file_path + "Figures/TimeReachingFirstVesselTip.png")

if GraphNbBranchesANDExpectedNbBranchesSameGraph:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_colors = [color_analyticalapproxpde, color_constant] 
    box_width = 0.03
    group_gap = 1.0

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sourceterm_analyticalapproxpde = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    average_analyticalapproxpde = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] 
    pos_constant = [i + box_width/2 + 0.0015 for i in sourceterm]
    pos_analyticalapproxpde = [i-box_width/2 - 0.0015 for i in average_analyticalapproxpde]
    position_total = []

    # loop over all the files 
    totalnumberbranches_analyticalapproxpde_average = np.zeros(20)
    totalnumberbranches_constant_average = np.zeros(10)
    totalnumberbranches_analyticalapproxpde_data = []
    totalnumberbranches_constant_data = []
    totalnumberbranches_total = []

    vegfhyp_total = []

    for k in range(1, 11):
        totalnumberbranches_analyticalapproxpde = []
        totalnumberbranches_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_branchesnumber_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizbranchnumber"
            file_branchesnumber_constant = file_path_constant + "/results_from_time_0/results.vizbranchnumber"
            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"
            file_anastomosis_constant = file_path_constant + "/results_from_time_0/results.vizanastomosis"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"
            
            totalnumberbranches_constant.append(runner.TotalNumberBranches(file_branchesnumber_constant, file_anastomosis_constant))
            totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_constant, file_anastomosis_constant))
            position_total.append(pos_constant[k-1])
            vegfhyp_total.append("ECM Hypothesis")

            if(k%2==1):
                totalnumberbranches_analyticalapproxpde.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                position_total.append(pos_analyticalapproxpde[k-1])
                vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # totalnumberbranches_analyticalapproxpde_average[k-1] = stats.mean(totalnumberbranches_analyticalapproxpde)
        # totalnumberbranches_constant_average[k-1] = stats.mean(totalnumberbranches_constant)

        # data lists
        totalnumberbranches_constant_data.append(totalnumberbranches_constant)
        totalnumberbranches_analyticalapproxpde_data.append(totalnumberbranches_analyticalapproxpde)

    for k in range(11, 21):
        totalnumberbranches_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_branchesnumber_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizbranchnumber"
            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            if(k%2==1):
                totalnumberbranches_analyticalapproxpde.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                position_total.append(pos_analyticalapproxpde[k-1])
                vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # totalnumberbranches_analyticalapproxpde_average[k-1] = stats.mean(totalnumberbranches_analyticalapproxpde)
        
        # data lists
        totalnumberbranches_analyticalapproxpde_data.append(totalnumberbranches_analyticalapproxpde)

    df = pd.DataFrame({'position':position_total, 'totalnumberbranches':totalnumberbranches_total,'vegfhyp':vegfhyp_total})
    df = df.copy()
    df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
    positions_sorted = np.sort(df['xpos'].unique())
    min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
    jitter_width = min(0.25 * min_gap, 0.05)  # cap jitter so points don't overlap neighbors
    rng = np.random.default_rng(42)
    df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

    yGlobal = [ExpectationBranches(Tend/Tcycle, item, Kc, 0, 0.041) for item in sourceterm]
    yLocal = [ExpectationBranches(Tend/Tcycle, item, Kc, 1, 0) for item in sourceterm]
 
    ax.plot(sourceterm, yLocal, label = 'Expected Number Branches (Local Hypothesis)', color = 'xkcd:black')
    ax.plot(sourceterm, yGlobal, label = 'Expected Number Branches (Global Hypothesis)', color = 'xkcd:black')
    sns.scatterplot(data=df, x='xpos_jit', y='totalnumberbranches', hue='vegfhyp', alpha=1.0, s=40, palette=["#008080", "#C00000"])

    legend_handles = [
        Line2D([0], [0], color=color_analyticalapproxpde, lw=3, label="Local Hypothesis"),
        Line2D([0], [0], color=color_constant, lw=3, label="Global Hypothesis"),
        ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper left', fontsize = 20)

    ax.set_xticks(sourceterm)
    ax.set_xlim(0.05, 1.05)
    ax.set_xticklabels([str(x) for x in sourceterm])
    ax.tick_params(axis = 'both', labelsize = 20)

    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Nb of Branches', fontsize = 20)
    plt.savefig(main_file_path + "Figures/ExpectedNbBranchesBothHypotheses.png")

if GraphNbBranchesBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_colors = [color_analyticalapproxpde, color_constant] 
    box_width = 0.03
    group_gap = 1.0

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sourceterm_analyticalapproxpde = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    average_analyticalapproxpde = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] 
    pos_constant = [i + box_width/2 + 0.0015 for i in sourceterm]
    pos_analyticalapproxpde = [i-box_width/2 - 0.0015 for i in average_analyticalapproxpde]
    position_total = []

    # loop over all the files 
    totalnumberbranches_analyticalapproxpde_average = np.zeros(20)
    totalnumberbranches_constant_average = np.zeros(10)
    totalnumberbranches_analyticalapproxpde_data = []
    totalnumberbranches_constant_data = []
    totalnumberbranches_total = []

    vegfhyp_total = []

    for k in range(1, 11):
        totalnumberbranches_analyticalapproxpde = []
        totalnumberbranches_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_branchesnumber_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizbranchnumber"
            file_branchesnumber_constant = file_path_constant + "/results_from_time_0/results.vizbranchnumber"
            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"
            file_anastomosis_constant = file_path_constant + "/results_from_time_0/results.vizanastomosis"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"
            
            totalnumberbranches_constant.append(runner.TotalNumberBranches(file_branchesnumber_constant, file_anastomosis_constant))
            totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_constant, file_anastomosis_constant))
            position_total.append(pos_constant[k-1])
            vegfhyp_total.append("ECM Hypothesis")

            if(k%2==1):
                totalnumberbranches_analyticalapproxpde.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                position_total.append(pos_analyticalapproxpde[k-1])
                vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # totalnumberbranches_analyticalapproxpde_average[k-1] = stats.mean(totalnumberbranches_analyticalapproxpde)
        # totalnumberbranches_constant_average[k-1] = stats.mean(totalnumberbranches_constant)

        # data lists
        totalnumberbranches_constant_data.append(totalnumberbranches_constant)
        totalnumberbranches_analyticalapproxpde_data.append(totalnumberbranches_analyticalapproxpde)

    for k in range(11, 21):
        totalnumberbranches_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_branchesnumber_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizbranchnumber"
            file_anastomosis_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizanastomosis"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            if(k%2==1):
                totalnumberbranches_analyticalapproxpde.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                totalnumberbranches_total.append(runner.TotalNumberBranches(file_branchesnumber_analyticalapproxpde, file_anastomosis_analyticalapproxpde))
                position_total.append(pos_analyticalapproxpde[k-1])
                vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # totalnumberbranches_analyticalapproxpde_average[k-1] = stats.mean(totalnumberbranches_analyticalapproxpde)
        
        # data lists
        totalnumberbranches_analyticalapproxpde_data.append(totalnumberbranches_analyticalapproxpde)
 
    # ax.plot(pos_constant, totalnumberbranches_constant_average, ls='--', label = 'ECM Hypothesis', color='#008080')
    # ax.plot(pos_analyticalapproxpde, totalnumberbranches_analyticalapproxpde_average, label = 'Lesion Hypothesis', ls='--', color='#C00000')

    bp_A = ax.boxplot(totalnumberbranches_constant_data, positions=pos_constant, widths=box_width, vert=True, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, meanline = True, meanprops=dict(color=color_constant, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_A, color_constant)

    totalnumberbranches_analyticalapproxpde_evendata = [totalnumberbranches_analyticalapproxpde_data[2*i] for i in range(10)]
    pos_analyticalapproxpde_even = [pos_analyticalapproxpde[2*i] for i in range(10)]
    bp_B = ax.boxplot(totalnumberbranches_analyticalapproxpde_evendata, positions=pos_analyticalapproxpde_even, widths=box_width, vert=True, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, meanline = True, meanprops=dict(color=color_analyticalapproxpde, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_B, color_analyticalapproxpde)

    df = pd.DataFrame({'position':position_total, 'totalnumberbranches':totalnumberbranches_total,'vegfhyp':vegfhyp_total})
    df = df.copy()
    df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
    positions_sorted = np.sort(df['xpos'].unique())
    min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
    jitter_width = min(0.25 * min_gap, 0.05)  # cap jitter so points don't overlap neighbors
    rng = np.random.default_rng(42)
    df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

    sns.scatterplot(data=df, x='xpos_jit', y='totalnumberbranches', hue='vegfhyp', alpha=1.0, s=40, palette=["#008080", "#C00000"])

    legend_handles = [
        Line2D([0], [0], color=color_analyticalapproxpde, lw=3, label="Local Hypothesis"),
        Line2D([0], [0], color=color_constant, lw=3, label="Global Hypothesis"),
        ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper left', fontsize = 20)

    ax.set_xticks(sourceterm)
    ax.set_xlim(0.05, 1.05)
    ax.set_xticklabels([str(x) for x in sourceterm])
    ax.tick_params(axis = 'both', labelsize = 20)

    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Nb of Branches', fontsize = 20)
    plt.savefig(main_file_path + "Figures/NbBranchesBoxPlots.png")


# Number of cells after a certain plane depending on source term for different Psprout 
if GraphNbCellsPlaneBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_colors = [color_analyticalapproxpde, color_constant] 
    box_width = 0.03
    group_gap = 1.0

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sourceterm_analyticalpproxpde = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    average_analyticalapproxpde = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] 
    pos_constant = [i + box_width/2 + 0.0015 for i in sourceterm]
    pos_analyticalapproxpde = [i-box_width/2 - 0.0015 for i in average_analyticalapproxpde]
    position_total = []

    # loop over all the files 
    cellsafterplane_analyticalapproxpde_average = np.zeros(20)
    cellsafterplane_constant_average = np.zeros(10)
    cellsafterplane_analyticalapproxpde_data = []
    cellsafterplane_constant_data = []
    cellsafterplane_total = []

    vegfhyp_total = []

    for k in range(1,11):
        cellsafterplane_analyticalapproxpde = []
        cellsafterplane_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_cellmutation_constant = file_path_constant + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)
            timeseed_constant = runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point, dim)

            if timeseed_constant < 1989.0:
                cellsafterplane_constant.append(runner.NbCellsAfterPlane(file_nodescoordinates_constant, ref_point, dim))
                cellsafterplane_total.append(runner.NbCellsAfterPlane(file_nodescoordinates_constant, ref_point, dim))
                position_total.append(pos_constant[k-1])
                vegfhyp_total.append("ECM Hypothesis")
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    cellsafterplane_analyticalapproxpde.append(runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    cellsafterplane_total.append(runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")
        
        # average of the results for one source term 
        # cellsafterplane_analyticalapproxpde_average[k-1] = stats.mean(cellsafterplane_analyticalapproxpde)
        # cellsafterplane_constant_average[k-1] = stats.mean(cellsafterplane_constant)

        # data lists
        cellsafterplane_constant_data.append(cellsafterplane_constant)
        cellsafterplane_analyticalapproxpde_data.append(cellsafterplane_analyticalapproxpde)

    for k in range(11,21):
        cellsafterplane_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)

            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    cellsafterplane_analyticalapproxpde.append(runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    cellsafterplane_total.append(runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # cellsafterplane_analyticalapproxpde_average[k-1] = stats.mean(cellsafterplane_analyticalapproxpde)

        # data lists
        cellsafterplane_analyticalapproxpde_data.append(cellsafterplane_analyticalapproxpde)

    # ax.plot(pos_constant, cellsafterplane_constant_average, ls='--', label = 'ECM Hypothesis', color='#008080')
    # ax.plot(pos_analyticalapproxpde, cellsafterplane_analyticalapproxpde_average, label = 'Lesion Hypothesis', ls='--', color='#C00000')

    bp_A = ax.boxplot(cellsafterplane_constant_data, positions=pos_constant, widths=box_width, vert=True, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, meanline = True, meanprops=dict(color=color_constant, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_A, color_constant)

    cellsafterplane_analyticalapproxpde_evendata = [cellsafterplane_analyticalapproxpde_data[2*i] for i in range(10)]
    pos_analyticalapproxpde_even = [pos_analyticalapproxpde[2*i] for i in range(10)]
    bp_B = ax.boxplot(cellsafterplane_analyticalapproxpde_evendata, positions=pos_analyticalapproxpde_even, widths=box_width, vert=True, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, meanline = True, meanprops=dict(color=color_analyticalapproxpde, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_B, color_analyticalapproxpde)

    df = pd.DataFrame({'position':position_total, 'cellsafterplane':cellsafterplane_total,'vegfhyp':vegfhyp_total})
    df = df.copy()
    df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
    positions_sorted = np.sort(df['xpos'].unique())
    min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
    jitter_width = min(0.25 * min_gap, 0.05)  # cap jitter so points don't overlap neighbors
    rng = np.random.default_rng(42)
    df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

    sns.scatterplot(data=df, x='xpos_jit', y='cellsafterplane', hue='vegfhyp', alpha=1.0, s=40, palette=["#008080", "#C00000"])

    legend_handles = [
        Line2D([0], [0], color=color_analyticalapproxpde, lw=3, label="Local Hypothesis"),
        Line2D([0], [0], color=color_constant, lw=3, label="Global Hypothesis"),
        ]
    ax.legend(handles=legend_handles, frameon=False, loc='upper left', fontsize = 20)

    ax.set_xticks(sourceterm)
    ax.set_xlim(0.05, 1.05)
    ax.set_xticklabels([str(x) for x in sourceterm])
    ax.tick_params(axis = 'both', labelsize = 16)

    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Nb of Vessel Segments Inside Lesion', fontsize = 20)    
    plt.savefig(main_file_path + "Figures/CellsAfterPlaneBoxPlots.png")

if GraphDensityCellInsideLesionComparedToTotalCellsBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_colors = [color_analyticalapproxpde, color_constant] 
    box_width = 0.03
    group_gap = 1.0

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sourceterm_analyticalapproxpde = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    average_analyticalapproxpde = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] 
    pos_constant = [i + box_width/2 + 0.0015 for i in sourceterm]
    pos_analyticalapproxpde = [i - box_width/2 - 0.0015 for i in average_analyticalapproxpde]
    position_total = []

    # loop over all the files 
    celldensitytotal_analyticalapproxpde_average = np.zeros(20)
    celldensitytotal_constant_average = np.zeros(10)
    celldensitytotal_analyticalapproxpde_data = []
    celldensitytotal_constant_data = []
    celldensitytotal_total = []

    vegfhyp_total = []

    for k in range(1,11):
        celldensitytotal_analyticalapproxpde = []
        celldensitytotal_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_cellmutation_constant = file_path_constant + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            celldensitytotal_constant.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_constant, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_constant))
            celldensitytotal_analyticalapproxpde.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_analyticalapproxpde))

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)
            timeseed_constant = runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point, dim)

            # remove the not vascularised network
            if timeseed_constant < 1989.0:
                celldensitytotal_total.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_constant, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_constant))
                position_total.append(pos_constant[k-1])
                vegfhyp_total.append("ECM Hypothesis")
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    celldensitytotal_total.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_analyticalapproxpde))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        celldensitytotal_analyticalapproxpde_average[k-1] = stats.mean(celldensitytotal_analyticalapproxpde)
        celldensitytotal_constant_average[k-1] = stats.mean(celldensitytotal_constant)

        # data lists
        celldensitytotal_analyticalapproxpde_data.append(celldensitytotal_analyticalapproxpde)
        celldensitytotal_constant_data.append(celldensitytotal_constant)

    for k in range(11,21):
        celldensitytotal_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            celldensitytotal_analyticalapproxpde.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_analyticalapproxpde))

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)

            # remove the not vascularised network
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    celldensitytotal_total.append(100*runner.NbCellsAfterPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)/runner.TotalNumberCells(file_cellmutation_analyticalapproxpde))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")

        # average of the results for one source term 
        # celldensitytotal_analyticalapproxpde_average[k-1] = stats.mean(celldensitytotal_analyticalapproxpde)

        # data lists
        celldensitytotal_analyticalapproxpde_data.append(celldensitytotal_analyticalapproxpde)

    # ax.plot(pos_constant, celldensitytotal_constant_average, ls = '--', label = 'ECM Hypothesis', color='#008080')
    # ax.plot(pos_analyticalapproxpde, celldensitytotal_analyticalapproxpde_average, ls = '--', label = 'Lesion Hypothesis', color='#C00000')

    bp_A = ax.boxplot(celldensitytotal_constant_data, positions=pos_constant, widths=box_width, vert=True, patch_artist=True, showmeans=False, meanline=True, manage_ticks=False, showfliers=False, meanprops=dict(color=color_constant, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_A, color_constant)

    celldensitytotal_analyticalapproxpde_evendata = [celldensitytotal_analyticalapproxpde_data[2*i] for i in range(10)]
    pos_analyticalapproxpde_even = [pos_analyticalapproxpde[2*i] for i in range(10)]
    bp_B = ax.boxplot(celldensitytotal_analyticalapproxpde_evendata, positions=pos_analyticalapproxpde_even, widths=box_width, vert=True, patch_artist=True, showmeans=False, meanline = True, manage_ticks=False, showfliers=False, meanprops=dict(color=color_analyticalapproxpde, linewidth=2, ls="--"), whis=np.inf)
    runner.style_boxplot(bp_B, color_analyticalapproxpde)

    df = pd.DataFrame({'position':position_total, 'celldensitytotal':celldensitytotal_total,'vegfhyp':vegfhyp_total})
    df = df.copy()
    df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
    positions_sorted = np.sort(df['xpos'].unique())
    min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
    jitter_width = min(0.25 * min_gap, 0.05)  # cap jitter so points don't overlap neighbors
    rng = np.random.default_rng(42)
    df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

    sns.scatterplot(data=df, x='xpos_jit', y='celldensitytotal', hue='vegfhyp', alpha=1.0, s=40, palette=["#008080", "#C00000"])

    legend_handles = [
        Line2D([0], [0], color=color_analyticalapproxpde, lw=3, label="Local Hypothesis"),
        Line2D([0], [0], color=color_constant, lw=3, label="Global Hypothesis"),]
    ax.legend(handles=legend_handles, frameon=False, loc='upper right', fontsize = 20)

    ax.set_xticks(sourceterm)
    ax.set_xlim(0.05, 1.05)
    ax.set_xticklabels([str(x) for x in sourceterm])
    ax.tick_params(axis = 'both', labelsize = 20)

    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel("Proportion of Networks' Vessel Segments Inside Lesion (%)", fontsize = 20)
    plt.savefig(main_file_path + "Figures/CellDensityInsideLesionComparedToTotalCells.png")


# Times
if GraphFirstTimeCellReachingLesionBoxPlot:
    # plot 
    fig, ax = plt.subplots(figsize = (14,8), dpi = 300, layout='constrained')

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_width = 0.03
    group_gap = 1.0

    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sourceterm_analyticalapproxpde = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    average_analyticalapproxpde = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05] #[np.float64(0.1), np.float64(0.14896667220480278), np.float64(0.19793334440960558), np.float64(0.24690001661440839), np.float64(0.29586668881921113), np.float64(0.34483336102401396), np.float64(0.39380003322881674), np.float64(0.4427667054336195), np.float64(0.4917333776384223), np.float64(0.5407000498432251)]
    pos_constant = [i+ box_width/2 + 0.0015 for i in sourceterm]
    pos_analyticalapproxpde = [i-box_width/2 - 0.0015 for i in average_analyticalapproxpde]
    position_total = []

    # loop over all the files 
    timefirstreachinglesion_analyticalapproxpde_average = np.zeros(20)
    timefirstreachinglesion_constant_average = np.zeros(10)
    timefirstreachinglesion_analyticalapproxpde_data = []
    timefirstreachinglesion_constant_data = []
    timefirstreachinglesion_total = []

    vegfhyp_total = []

    for k in range(1,11):
        timefirstreachinglesion_analyticalapproxpde = []
        timefirstreachinglesion_constant = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)
            file_path_constant = main_file_path + "CoupledModel2DConstant/CoupledModel2DConstantSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"
            file_cellmutation_constant = file_path_constant + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_constant = file_path_constant + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)
            timeseed_constant = runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point, dim)

            timefirstreachinglesion_analyticalapproxpde.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
            timefirstreachinglesion_constant.append(runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point, dim))

            if timeseed_constant < 1989.0:
                timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_constant, ref_point, dim))
                position_total.append(pos_constant[k-1])
                vegfhyp_total.append("ECM Hypothesis")
            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")

         # average of the results for one source term 
        timefirstreachinglesion_analyticalapproxpde_average[k-1] = stats.mean(timefirstreachinglesion_analyticalapproxpde)
        timefirstreachinglesion_constant_average[k-1] = stats.mean(timefirstreachinglesion_constant)

        # data lists
        timefirstreachinglesion_constant_data.append(timefirstreachinglesion_constant)
        timefirstreachinglesion_analyticalapproxpde_data.append(timefirstreachinglesion_analyticalapproxpde)

    for k in range(11,21):
        timefirstreachinglesion_analyticalapproxpde = []
        for j in list_40_seeds:
            file_path_analyticalapproxpde = main_file_path + "CoupledModel2DAnalyticalApproxPde/CoupledModel2DAnalyticalApproxPdeSeed" + str(j) + "Source" + str(k)

            file_cellmutation_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.vizmutationstates"
            file_nodescoordinates_analyticalapproxpde = file_path_analyticalapproxpde + "/results_from_time_0/results.viznodes"

            timeseed_analyticalapproxpde = runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim)
            timefirstreachinglesion_analyticalapproxpde.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))

            if timeseed_analyticalapproxpde < 1989.0:
                if(k % 2 == 1):
                    timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, ref_point, dim))
                    position_total.append(pos_analyticalapproxpde[k-1])
                    vegfhyp_total.append("Lesion Hypothesis")

         # average of the results for one source term 
        timefirstreachinglesion_analyticalapproxpde_average[k-1] = stats.mean(timefirstreachinglesion_analyticalapproxpde)

        # data lists
        timefirstreachinglesion_analyticalapproxpde_data.append(timefirstreachinglesion_analyticalapproxpde)

    # ax.plot(pos_constant, timefirstreachinglesion_constant_average, marker = '.', markersize = 6, label = 'ECM Hypothesis', color='#008080')
    # ax.plot(pos_analyticalapproxpde, timefirstreachinglesion_analyticalapproxpde_average, label = 'Lesion Hypothesis', marker = '.', markersize = 6, color='#C00000')

    bp_A = ax.boxplot(timefirstreachinglesion_constant_data, positions=pos_constant, widths=box_width, vert=True, patch_artist=True, showmeans=False, meanline = True, manage_ticks=False, showfliers=False, whis=np.inf)
    runner.style_boxplot(bp_A, color_constant)

    timefirstreachinglesion_analyticalapproxpde_evendata = [timefirstreachinglesion_analyticalapproxpde_data[2*i] for i in range(10)]
    pos_analyticalapproxpde_even = [pos_analyticalapproxpde[2*i] for i in range(10)]
    bp_B = ax.boxplot(timefirstreachinglesion_analyticalapproxpde_evendata, positions=pos_analyticalapproxpde_even, widths=box_width, vert=True, patch_artist=True, showmeans=False, meanline = True, manage_ticks=False, showfliers=False, whis=np.inf)
    runner.style_boxplot(bp_B, color_analyticalapproxpde)

    df = pd.DataFrame({'position':position_total, 'timefirstreachinglesion':timefirstreachinglesion_total,'vegfhyp':vegfhyp_total})
    df = df.copy()
    df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
    positions_sorted = np.sort(df['xpos'].unique())
    min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
    jitter_width = min(0.25 * min_gap, 0.04)  # cap jitter so points don't overlap neighbors
    rng = np.random.default_rng(42)
    df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

    sns.scatterplot(data=df, x='xpos_jit', y='timefirstreachinglesion', hue='vegfhyp', alpha=1.0, s=40, palette=["#008080", "#C00000"])

    legend_handles = [
        Line2D([0], [0], color=color_analyticalapproxpde, lw=3, label="Local Hypothesis"),
        Line2D([0], [0], color=color_constant, lw=3, label="Global Hypothesis"),
        ]
    ax.legend(handles=legend_handles, frameon=False, loc='lower right', fontsize = 20)

    ax.set_xticks(sourceterm)
    ax.set_xlim(0.05, 1.05)
    ax.set_xticklabels([str(x) for x in sourceterm])
    ax.tick_params(axis = 'both', labelsize = 16)

    ax.set_xlabel(r'$c_{average}$', fontsize = 26)
    ax.set_ylabel('Time For Network to Reach Lesion (hours)', fontsize = 20)
    ax.set_yticks([1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])
    ax.set_ylim(1300, 2015)
    plt.savefig(main_file_path + "Figures/TimeFirstCellReachingLesionBoxPlot.png")

if GraphErrorFirstTimeCellReachingLesionPositionLesionBoxPlots:
    # plot 
    fig, ax = plt.subplots(figsize = (10,10), dpi = 300, layout='constrained')

    k_analyticalapproxpde = 19
    k_constant = 10
    k_average = 10

    color_analyticalapproxpde='#C00000'
    color_constant='#008080'
    box_width = 0.02
    group_gap = 0.1

    DistanceLesionVessel = [40, 30, 20]
    DistanceLesionVesselTicks = [1.10, 1.20, 1.30]
    sourceterm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    pos_constant = [i + 0.005 + box_width/2 for i in DistanceLesionVesselTicks]
    pos_analyticalapproxpde = [i - 0.005 - box_width/2 for i in DistanceLesionVesselTicks]
    position_total = []

    timefirstreachinglesion_analyticalapproxpde_average = np.zeros(10)
    timefirstreachinglesion_constant_average = np.zeros(10)
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

            # remove the not vascularised network
            if timeseed_analyticalapproxpde < 1989.0:
                time_analyticalapproxpde.append(timeseed_analyticalapproxpde)

                timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_analyticalapproxpde, i, dim))
                position_total.append(pos_analyticalapproxpde[m])
                vegfhyp_total.append("Lesion Hypothesis")
            if timeseed_constant < 1989.0:
                time_constant.append(timeseed_constant)

                timefirstreachinglesion_total.append(runner.TimeFirstReachingPlane(file_nodescoordinates_constant, i, dim))
                position_total.append(pos_constant[m])
                vegfhyp_total.append("ECM Hypothesis")

        timereachinglesion_analyticalapproxpde.append(time_analyticalapproxpde)
        timereachinglesion_constant.append(time_constant)

        df = pd.DataFrame({'position':position_total, 'timefirstreachinglesion':timefirstreachinglesion_total,'vegfhyp':vegfhyp_total})
        df = df.copy()
        df['xpos'] = df['position'].astype(float) if df['position'].dtype == object else df['position']
        positions_sorted = np.sort(df['xpos'].unique())
        min_gap = np.diff(positions_sorted).min() if len(positions_sorted) > 1 else 0.05
        jitter_width = min(0.25 * min_gap, 0.05)  # cap jitter so points don't overlap neighbors
        rng = np.random.default_rng(42)
        df['xpos_jit'] = df['xpos'] + rng.uniform(-jitter_width, jitter_width, size=len(df))

        sns.scatterplot(data=df, x='xpos_jit', y='timefirstreachinglesion', hue='vegfhyp', alpha=1.0, s=40, palette=["#C00000", "#008080"])

    width = 0.02
    position_constant = [item + 0.005 + width/2 for item in DistanceLesionVesselTicks]
    position_analyticalapproxpde = [item - 0.005 - width/2 for item in DistanceLesionVesselTicks]

    box_colors = ['#C00000', '#008080']
    bp_analyticalapproxpde = ax.boxplot(timereachinglesion_analyticalapproxpde, positions = position_analyticalapproxpde, widths=width, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, whis=np.inf) 
    runner.style_boxplot(bp_analyticalapproxpde, color_analyticalapproxpde)
    bp_constant = ax.boxplot(timereachinglesion_constant, positions = position_constant, widths=width, patch_artist=True, showmeans=False, manage_ticks=False, showfliers=False, whis=np.inf) 
    runner.style_boxplot(bp_constant, color_constant)

    ax.set_xlabel('Distance Lesion / Vessel (mm)', fontsize = 24)
    ax.set_ylabel('Time For Network To Reach Lesion (hours)', fontsize = 24)
    custom_lines = [plt.Line2D([0], [0], color=box_colors[0], lw=4), plt.Line2D([0], [0], color=box_colors[1], lw=4)]
    ax.legend(custom_lines, ['Local Hypothesis', 'Global Hypothesis'], loc='lower right', fontsize = 24)  
    ax.set_xticks(DistanceLesionVesselTicks)
    ax.tick_params(axis = 'both', labelsize = 24)
    ax.set_ylim(1475, 2025)
    plt.savefig(main_file_path + "Figures/ErrorFirstTimeCellReachingLesionPositionLesion" + str(k_average) + ".png")


# Vascularisation
if GraphVascularisation:
    # plot 
    fig, ax = plt.subplots(figsize = (12,8), dpi = 300, layout='constrained')

    # loop over all the files 
    vascularisation_analyticalapproxpde = []
    vascularisation_constant = []

    ref_point_test = 20

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
                time_analyticalapproxpde.append(timeseed_analyticalapproxpde)
                vascularisationvalue_analyticalapproxpde += 1
            if timeseed_constant < 1989.0:
                time_constant.append(timeseed_constant)
                vascularisationvalue_constant += 1

        vascularisation_analyticalapproxpde.append(vascularisationvalue_analyticalapproxpde*100/TotalTestNb)
        vascularisation_constant.append(vascularisationvalue_constant*100/TotalTestNb)

    SourceTermTicks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    barWidth = 0.025
    position_constant = [item + barWidth/2 for item in SourceTermTicks]
    position_analyticalapproxpde = [item - barWidth/2 for item in SourceTermTicks]

    bp_analyticalapproxpde = ax.bar(position_analyticalapproxpde, vascularisation_analyticalapproxpde, width=barWidth, color = '#C00000', label = 'Local Hypothesis') 
    bp_constant = ax.bar(position_constant, vascularisation_constant, width=barWidth, color = '#008080', label = 'Global Hypothesis') 
   
    ax.set_xlabel(r'$c_{max}$', fontsize = 26)
    ax.set_ylabel('Realisations with Lesion Vascularised \n (%)', fontsize = 26)
    ax.legend(loc='lower left', fontsize = 24)
    ax.set_xticks(SourceTermTicks)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.tick_params(axis = 'both', labelsize = 24)
    plt.savefig(main_file_path + "Figures/VascularisationBarPlot" + str(ref_point_test) + ".png")
