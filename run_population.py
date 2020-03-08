# common imports
from __future__ import division

# python packages
import numpy as np
import matplotlib.pyplot as plt
import casadi as cs
from matplotlib import gridspec

# local imports
from local_imports import LimitCycle as lc
from local_imports import PlotOptions as plo
from local_imports import Utilities as uts
from local_models.malaria_pop_model import param, y0in, malaria_model

def plot_L_F(ts, L, F, ax, light='DD'):
    """ plots bars for light (black-white) and feeding (geen-white)
    cycles at the top 
    
    Activity is one of 'DD', 'LD', 'FB'
    """

    ld = L(ts) # lgiht: 1 when light, 0 when dark
    ff = F(ts) # feeding: 1 when fed, 0 when fasted
    lighton = np.ma.masked_where(ld == 0, np.zeros(len(ts)))
    lightoff = np.ma.masked_where(ld > 0.005, np.zeros(len(ts)))
    feedon = np.ma.masked_where(ff > 0.005, np.zeros(len(ts)))
    feedoff = np.ma.masked_where(ff == 0, np.zeros(len(ts)))

    ax.fill_between(ts/24, feedon+2.15, feedon+2, color='gray',
                    label='Feeding')
    if light=='DD':
        ax.fill_between(ts/24, feedon+2.3, feedon+2.15, color='orange',
                    label='Activity')
    if light=='LD':
        ax.fill_between(ts/24, lightoff+2.3, lightoff+2.15, color='orange',
                    label='Activity')
    f1 = ax.fill_between(ts/24, lightoff+2, lightoff+1.85, color='black',
                    label='Dark')
    #f2 = ax.fill_between(ts/24, lighton+2, lighton+1.85, color='white')

    #for fi in [f1, f2]:
    #    fi.set_edgecolor('k')

# key: here are the four models we are testing

# model type          signal   malaria intrinsic
models = {"Model 1": ['food', False],
          "Model 2": ['brain', False],
          "Model 3": ['food', True],
          "Model 4": ['brain', True]}

# experiments            geno  light feeding
experiments = {"Case1": ['WT', 'DD', 'AdLib'],
               "Case2": ['WT', 'LD', 'AdLib'],
               "Case3": ['WT', 'LD', 'SpreadOut'],
               "Case4": ['FB', 'DD', 'AdLib'],
               "Case5": ['YY', 'DD', 'AdLib']
               }


# single-figure: just-in-time
plo.PlotOptions(ticks='in')
fig = plt.figure(figsize=(4.5,8.5))
gs = gridspec.GridSpec(5,2, height_ratios=(1.9,1,1,1,1))
yl = [True, False]
for mi, model in enumerate(['Model 1', 'Model 2']):
    a2 = mi
    ylab = yl[mi]
    for ci, case in enumerate(['Case1', 'Case2', 'Case3', 'Case4', 'Case5']):
        # set up the model once
        signal = models[model][0]
        osc = models[model][1]
        geno = experiments[case][0]
        lcyc = experiments[case][1]
        fcyc = experiments[case][2]
        ODEs, L, F = malaria_model(lcyc, signal, fcyc, geno, osc)
        model4_case1 = lc.Oscillator(ODEs, param, y0=y0in)
        ts, states = model4_case1.int_odes(200)

        # plot
        
        ax = plt.subplot(gs[ci, mi])
        plot_L_F(ts, L, F, ax, light=lcyc)
        ax.plot(ts/24, states[:,5::4]/0.28, color='pink', alpha=0.1)
        ax.plot(ts/24, states[:,0]/0.28, color='f', label='Mouse Clock')
        ax.plot(ts/24, states[:,5::4].mean(1)/0.28, ls=':', lw=1.5, color='h', label='Parasite mean')
        ax.set_ylim([0,2.6])
        ax.set_yticks([0,0.5,1.])
        ax.set_xlim([0,8])
        if ylab:
            ax.set_ylabel('Conc. (AU)        ')
        if osc:
            ost = 'intrinsic'
        else:
            ost = 'just-in-time'
        ax.text(0.1, 2.4, "Condition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
        if ci ==0:
            ax.set_ylim(0,4.1)
            plt.legend(ncol=1)
    ax.set_xlabel('Time (days)')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/many/just_in_time.pdf')


# single-figure: endogenous oscillator
fig = plt.figure(figsize=(4.5,8.5))
gs = gridspec.GridSpec(5,2, height_ratios=(1.9,1,1,1,1))
yl = [True, False]
for mi, model in enumerate(['Model 3', 'Model 4']):
    a2 = mi
    ylab = yl[mi]
    for ci, case in enumerate(['Case1', 'Case2', 'Case3', 'Case4', 'Case5']):
        # set up the model once
        signal = models[model][0]
        osc = models[model][1]
        geno = experiments[case][0]
        lcyc = experiments[case][1]
        fcyc = experiments[case][2]
        ODEs, L, F = malaria_model(lcyc, signal, fcyc, geno, osc)
        model4_case1 = lc.Oscillator(ODEs, param, y0=y0in)
        ts, states = model4_case1.int_odes(200)

        # plot
        
        ax = plt.subplot(gs[ci, mi])
        plot_L_F(ts, L, F, ax, light=lcyc)
        ax.plot(ts/24, states[:,5::4]/0.28, color='pink', alpha=0.1)
        ax.plot(ts/24, states[:,0]/0.28, color='f', label='Mouse Clock')
        ax.plot(ts/24, states[:,5::4].mean(1)/0.28, ls=':', lw=1.5, color='h', label='Parasite mean')
        ax.set_ylim([0,2.6])
        ax.set_yticks([0,0.5,1.])
        ax.set_xlim([0,8])
        if ylab:
            ax.set_ylabel('Conc. (AU)        ')
        if osc:
            ost = 'intrinsic'
        else:
            ost = 'just-in-time'
        ax.text(0.1, 2.4, "Condition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
        if ci ==0:
            ax.set_ylim(0,4.1)
            plt.legend(ncol=1)
    ax.set_xlabel('Time (days)')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/many/endogenous.pdf')












# plo.PlotOptions(ticks='in')
# # run all models
# for model in models.keys():
#     for case in experiments.keys():
#         # set up the model once
#         signal = models[model][0]
#         osc = models[model][1]
#         geno = experiments[case][0]
#         lcyc = experiments[case][1]
#         fcyc = experiments[case][2]
#         ODEs, L, F = malaria_model(lcyc, signal, fcyc, geno, osc)
#         model4_case1 = lc.Oscillator(ODEs, param, y0=y0in)
#         ts, states = model4_case1.int_odes(200)

#         # plot
#         fig = plt.figure()
#         ax = plt.subplot()
#         plot_L_F(ts, L, F, ax, light=lcyc)
#         ax.plot(ts/24, states[:,5::4]/0.28, color='pink', alpha=0.1)
#         ax.plot(ts/24, states[:,0]/0.28, color='f', label='Mouse Clock')
#         ax.plot(ts/24, states[:,5::4].mean(1)/0.28, ls=':', lw=1.5, color='h', label='Parasite mean')
#         ax.set_ylim([0,3])
#         ax.set_yticks([0,0.5,1.])
#         ax.set_xlim([0,8])
#         if osc:
#             ost = 'intrinsic'
#         else:
#             ost = 'just-in-time'
#         ax.text(0, 1.3, model+": "+signal+"-entrained, "+ost+"\nCondition: "+geno+", "+lcyc+", "+fcyc+" feeding", fontsize=7)
#         plt.legend()
#         #plt.savefig('results/many/'+model+"/"+case+".png")
#         plt.close(fig)