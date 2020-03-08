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
models = {
          "Model 4": ['activity', True]}

# experiments            geno  light feeding
experiments = {
               "Case5": ['WT', 'DD', 'ad-lib fed']
               }

# run all models
model = "Model 4"
case = "Case5"

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
fig = plt.figure(figsize=(7.0, 4.0))
gs = gridspec.GridSpec(1, 3)

ax = plt.subplot(gs[0,0])
ax.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
ax.plot(ts/24, states[:,5]/0.28, label='Single-parasite rhythm')
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


bx = plt.subplot(gs[0,1])
bx.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
bx.plot(ts/24, states[:,5]/0.28*np.exp(0.4*ts/24), label='Growing population rhythm')
plot_L_F(ts, L, F, bx)
bx.set_ylim([0,10+2.8])
bx.set_yticks([0,0.5,1.])
bx.set_xlabel('Time (days)')
bx.set_ylabel('Concentration (AU)'+
                '                                               ')
bx.set_xlim([0,8])
if osc:
    ost = 'intrinsic'
else:
    ost = 'just-in-time'
bx.text(0.1, 2.05, model+": "+ost+", "+signal+"-entrained\nCondition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
plt.legend()


cx = plt.subplot(gs[0,2])
cx.plot(ts/24, states[:,0]/0.28, label='Mouse rhythm')
cx.plot(ts/24, states[:,5]/0.28, label='RPKM rhythm')
plot_L_F(ts, L, F, cx)
cx.set_ylim([0,2.8])
cx.set_yticks([0,0.5,1.])
cx.set_xlabel('Time (days)')
cx.set_ylabel('Concentration (AU)'+
                '                                               ')
cx.set_xlim([0,8])
if osc:
    ost = 'intrinsic'
else:
    ost = 'just-in-time'
cx.text(0.1, 2.05, model+": "+ost+", "+signal+"-entrained\nCondition: "+geno+", "+lcyc+", "+fcyc, fontsize=7)
plt.legend()

plt.tight_layout(**plo.layout_pad)

