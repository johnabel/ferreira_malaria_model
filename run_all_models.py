# common imports
from __future__ import division

# python packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs

# local imports
from local_imports import LimitCycle as lc
from local_imports import PlotOptions as plo
from local_imports import Utilities as uts
from local_models import malaria_model as mm
reload(mm)
from local_models.malaria_model import param, y0in, malaria_model

def plot_L_F(ts, L, F, ax):
    """ plots bars for light (black-white) and feeding (geen-white)
    cycles at the top """

    ld = L(ts) # lgiht: 1 when light, 0 when dark
    ff = F(ts) # feeding: 1 when fed, 0 when fasted
    lighton = np.ma.masked_where(ld == 0, np.zeros(len(ts)))
    lightoff = np.ma.masked_where(ld > 0.005, np.zeros(len(ts)))
    feedon = np.ma.masked_where(ff > 0.005, np.zeros(len(ts)))
    feedoff = np.ma.masked_where(ff == 0, np.zeros(len(ts)))

    f1 = ax.fill_between(ts/24, lightoff+1.98, lightoff+1.85, color='black',
                    label='Dark')
    f2 = ax.fill_between(ts/24, lighton+1.98, lighton+1.85, color='white',
                    label='Light')
    f4 = ax.fill_between(ts/24, feedon+1.945, feedon+1.885, color='red',
                    label='Feeding')
    for fi in [f1, f2]:
        fi.set_edgecolor('k')


# key: here are the four models we are testing

# model type          signal   malaria intrinsic
models = {"Model 1": ['food', False],
          "Model 2": ['activity', False],
          "Model 3": ['food', True],
          "Model 4": ['activity', True]}

# experiments            geno  light feeding
experiments = {"Case1": ['WT', 'LD', 'ad-lib fed'],
               "Case2": ['WT', 'DD', 'ad-lib fed'],
               "Case3": ['WT', 'LD', 'spread-out fed'],
               "Case4": ['FB', 'DD', 'ad-lib fed'],
               "Case5": ['YY', 'DD', 'ad-lib fed']
               }

# run all models
for model in models.keys():
    for case in ['Case1', 'Case2', 'Case3', 'Case4', 'Case5']:
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
        fig = plt.figure()
        ax = plt.subplot()
        ax.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
        ax.plot(ts/24, states[:,5]/0.28, label='Malaria rhythm')
        plot_L_F(ts, L, F, ax)
        ax.set_ylim([0,2.8])
        ax.set_yticks([0,0.5,1.])
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Concentration (AU)'+
                      '                                               ')
        ax.set_xlim([0,8])
        if osc:
            ost = 'intrinsic'
        else:
            ost = 'just-in-time'
        ax.text(0.1, 2.05, model+": "+ost+", "+signal+"-entrained\nCondition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
        plt.legend()
        plt.tight_layout(**plo.layout_pad)

        plt.savefig('results/single/'+model+"/"+case+".pdf")
        plt.close(fig)



# single-figure: just-in-time
fig = plt.figure(figsize=(4.5,8.5))
gs = gridspec.GridSpec(5,2, height_ratios=(1.9,1,1,1,1))
for mi, model in enumerate(['Model 1', 'Model 2']):
    a2 = mi
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
        plot_L_F(ts, L, F, ax)
        ax.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
        ax.plot(ts/24, states[:,5]/0.28, label='Malaria rhythm')
        ax.set_ylim([0,2.3])
        ax.set_yticks([0,0.5,1.])
        ax.set_xlim([0,8])
        if osc:
            ost = 'intrinsic'
        else:
            ost = 'just-in-time'
        ax.text(0.1, 2.05, "Condition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
        if ci ==0:
            ax.set_ylim(0,4.1)
            plt.legend(ncol=2)
    ax.set_xlabel('Time (days)')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/single/just_in_time.pdf')



# single-figure: endogenous oscillator
fig = plt.figure(figsize=(4.5,8.5))
gs = gridspec.GridSpec(5,2, height_ratios=(1.9,1,1,1,1))
for mi, model in enumerate(['Model 3', 'Model 4']):
    a2 = mi
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
        plot_L_F(ts, L, F, ax)
        ax.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
        ax.plot(ts/24, states[:,5]/0.28, label='Malaria rhythm')
        ax.set_ylim([0,2.3])
        ax.set_yticks([0,0.5,1.])
        ax.set_xlim([0,8])
        if osc:
            ost = 'intrinsic'
        else:
            ost = 'just-in-time'
        ax.text(0.1, 2.05, "Condition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
        if ci ==0:
            ax.set_ylim(0,4.1)
            plt.legend(ncol=2)
    ax.set_xlabel('Time (days)')

plt.tight_layout(**plo.layout_pad)
plt.savefig('results/single/endogenous.pdf')










