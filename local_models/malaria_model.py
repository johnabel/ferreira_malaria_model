"""
Created on Tue Jan 13 13:01:35 2014
Revised Fri Jan 18 2019

@author: John H. Abel

Demo some ideas for malaria
Spontaneous Synchronization of Coupled Circadian Oscillators

Model formulation:
 External feeding signal given by F.

# mouse brain info
Mouse brain oscillator is X1-X3. The mouse brain oscillator is a single Gonze 2005 oscillator, parameterized as in the original model. The brain sends signal B1, like the intercellular coupling in the original model. To recreate the FBXL3-mutant mouse, I'm changing a parameter of the mouse oscillator to make the period 7.5% longer in the mouse brain.

# malaria transcription info
The malaria transcription may be made oscillatory by changing parameter nm from 1 (non-oscillatory, JIT) to 4 (self-sustained) as it crosses a Hopf bifurcation. I'm putting the food and brain input into the same term to keep things consistent.

# light info
There are two LD mice and three DD mice. The LD mouse has a square wave with period half of 24h, 

# feeding info
There are only two conditions here: activity, or day. If the mouse feeds when active, then we have a feeding cycle that is a square wave with a period that of the mouse oscillator itself. If the food is restricted to day, the feeding cycle is a square wave with the period 24h. Most importantly, feeding is 0 when mouse is being fed, feeding is 0.01 when it is not. Thus the feeding and the light drive entrainment in the same way.

# coupling
We removed the self-coupling within the brain--instead making the brain oscillator coupling only to the malaria one. Thus, both brain and malaria oscillator need same n (cooperativity) to oscillate autonomously.

This modeling was done in python 2.7 with cassadi 2.3.0


TODO:
- find period without coupling term
- edit to norm WT and FB correctly
"""

# common imports
from __future__ import division

# python packages
import numpy as np
import casadi as cs

modelversion = 'malaria_model'

# constants and equations setup, trying a new method
EqCount = 9
ParamCount  = 20

param = [  0.7,    1,    0.35,    1,  0.7, 0.35,   
             1,  0.7, 0.35,    1, 0.35,    1,    1,
           0.4,    1,  0.5]
y0in = np.array([ 0.05069219,  0.10174506,  2.28099242, 0.01522458,
                  0.01522458, 
                  0.05069219,  0.10174506,  2.28099242, 0.01522458])

# periods so as to get the time right
gonze_period = 30.27
malaria_period = 24.2
WT_period = 23.7
FB_period = 25.5


def malaria_model(light_schedule, mouse_signal, mouse_feeding, mouse_genotype,              malaria_intrinsic):
    """
    Malaria model of mouse-parasite circadian interation.
    light_schedule = ('DD', 'LD')
    mouse_signal = ('food', 'activity')
    mouse_feeding = ('AdLib', 'Ultradian')
    mouse_genotype = ('WT', 'FB', 'YY')
    malaria_intrinsic = (True, False)

    The setup of the experiment is handled within this model.
    """

    # set up signaling in model
    if mouse_signal=='food':
        feed_signal=1
        brain_signal = bs =0
    elif mouse_signal=='activity':
        brain_signal = bs = 1
        feed_signal = 0

    # set up oscillator in malaria
    if malaria_intrinsic==True:
        nm = 4
    elif malaria_intrinsic==False:
        nm = 2

    # set up mouse genotype
    if mouse_genotype=='WT':
        n = 4
        cryko = 1.
        mouse_period = WT_period
    elif mouse_genotype=='FB':
        n = 4
        cryko = 1.
        mouse_period = FB_period
    elif mouse_genotype=='YY':
        n = 4
        cryko = 0.
        bs = 0 # since we are averaging we don't want to average in the brain signal if there is none!
        mouse_period = WT_period

    # Time
    t = cs.SX.sym('t')
    
    # set up light schedule - 10 days
    if light_schedule=='DD':
        L = 0
    elif light_schedule=='LD':
        L = 0.01*((cs.heaviside(t) - cs.heaviside(t-12) + \
            cs.heaviside(t-24) - cs.heaviside(t-12-24) + \
            cs.heaviside(t-48) - cs.heaviside(t-12-48) + \
            cs.heaviside(t-72) - cs.heaviside(t-12-72) + \
            cs.heaviside(t-96) - cs.heaviside(t-12-96) + \
            cs.heaviside(t-120) - cs.heaviside(t-12-120) + \
            cs.heaviside(t-144) - cs.heaviside(t-12-144) + \
            cs.heaviside(t-168) - cs.heaviside(t-12-168) + \
            cs.heaviside(t-192) - cs.heaviside(t-12-192) + \
            cs.heaviside(t-216) - cs.heaviside(t-12-216)))

    # set up light schedule - 10 days
    if mouse_feeding=='ad-lib fed':
        # if dd, mouse feeds on its own period
        if light_schedule=='DD':
            if mouse_genotype=="YY":
                F = 0.005
            else:
                t1 = mouse_period/2
                t2 = mouse_period
                F = 0.01*((cs.heaviside(t) - cs.heaviside(t-t1) + \
                    cs.heaviside(t-t2) - cs.heaviside(t-t1-t2) + \
                    cs.heaviside(t-t2*2) - cs.heaviside(t-t1-t2*2) + \
                    cs.heaviside(t-t2*3) - cs.heaviside(t-t1-t2*3) + \
                    cs.heaviside(t-t2*4) - cs.heaviside(t-t1-t2*4) + \
                    cs.heaviside(t-t2*5) - cs.heaviside(t-t1-t2*5) + \
                    cs.heaviside(t-t2*6) - cs.heaviside(t-t1-t2*6) + \
                    cs.heaviside(t-t2*7) - cs.heaviside(t-t1-t2*7) + \
                    cs.heaviside(t-t2*8) - cs.heaviside(t-t1-t2*8) + \
                    cs.heaviside(t-t2*9) - cs.heaviside(t-t1-t2*9)))
        # if ld, mouse feeds on light-dark period
        elif light_schedule=='LD':
            t1 = 24/2
            t2 = 24
            F = 0.01*(cs.heaviside(t) - cs.heaviside(t-t1) + \
                cs.heaviside(t-t2) - cs.heaviside(t-t1-t2) + \
                cs.heaviside(t-t2*2) - cs.heaviside(t-t1-t2*2) + \
                cs.heaviside(t-t2*3) - cs.heaviside(t-t1-t2*3) + \
                cs.heaviside(t-t2*4) - cs.heaviside(t-t1-t2*4) + \
                cs.heaviside(t-t2*5) - cs.heaviside(t-t1-t2*5) + \
                cs.heaviside(t-t2*6) - cs.heaviside(t-t1-t2*6) + \
                cs.heaviside(t-t2*7) - cs.heaviside(t-t1-t2*7) + \
                cs.heaviside(t-t2*8) - cs.heaviside(t-t1-t2*8) + \
                cs.heaviside(t-t2*9) - cs.heaviside(t-t1-t2*9))

    elif mouse_feeding=='spread-out fed':
        assert light_schedule=='LD', "Light schedule must be LD for ultradian feeding."
        F = 0.005


    #############################################################
    #   Now, construct the model
    #############################################################

    # for mouse
    X1 = cs.SX.sym('X1')
    X2 = cs.SX.sym('X2')
    X3 = cs.SX.sym('X3')
    X4 = cs.SX.sym('X4')

    # for brain to malaria signal
    B1 = cs.SX.sym('B1')

    # for malaria
    M1 = cs.SX.sym('M1')
    M2 = cs.SX.sym('M2')
    M3 = cs.SX.sym('M3')
    M4 = cs.SX.sym('M4')
    
    state_set = cs.vertcat([X1, X2, X3, X4, B1, M1, M2, M3, M4])

    # Parameter Assignments
    v1  = cs.SX.sym('v1')
    K1  = cs.SX.sym('K1')
    v2  = cs.SX.sym('v2')
    K2  = cs.SX.sym('K2')
    k3  = cs.SX.sym('k3')
    v4  = cs.SX.sym('v4')
    K4  = cs.SX.sym('K4')
    k5  = cs.SX.sym('k5')
    v6  = cs.SX.sym('v6')
    K6  = cs.SX.sym('K6')
    k7  = cs.SX.sym('k7')
    v8  = cs.SX.sym('v8')
    K8  = cs.SX.sym('K8')
    vc  = cs.SX.sym('vc')
    Kc  = cs.SX.sym('Kc')
    K   = cs.SX.sym('K')

    param_set = cs.vertcat([v1,K1,v2,K2,k3,v4,K4,k5,v6,K6,k7,v8,K8,vc,Kc,K])

    # oscillators
    ode = [[]]*EqCount
    
    # mouse mRNA, protein, TF, internal signal
    ode[0] = gonze_period/mouse_period*(cryko*v1*K1**n/(K1**n + X3**n) \
             - v2*(X1)/(K2+X1) +vc*K*((X4))/(Kc +K*(X4))) \
              + L
    ode[1] = gonze_period/mouse_period*(k3*(X1) - v4*X2/(K4+X2))
    ode[2] = gonze_period/mouse_period*(k5*X2 - v6*X3/(K6+X3))
    ode[3] = gonze_period/mouse_period*(k7*(X1) - v8*B1/(K8+X4))
    
    # mouse signal from brain to parasite
    ode[4] = gonze_period/mouse_period*(k7*(X1) - v8*B1/(K8+B1))

    # parasite states 1, 2, 3, 4 (same as mouse 1-4)
    ode[5] = gonze_period/malaria_period*(v1*K1**nm/(K1**nm + M3**nm) \
             - v2*(M1)/(K2+M1) + (1/(1+bs))*vc*K*((M4))/(Kc +K*(M4)) \
             + (bs/(1+bs))*vc*K*((B1))/(Kc +K*(B1)) ) \
             + feed_signal*F
    ode[6] = gonze_period/malaria_period*(k3*(M1) - v4*M2/(K4+M2))
    ode[7] = gonze_period/malaria_period*(k5*M2 - v6*M3/(K6+M3))
    ode[8] = gonze_period/malaria_period*(k7*(M1) - v8*M4/(K8+M4))


    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t,x=state_set,p=param_set), 
            cs.daeOut(ode=ode))

    fn.setOption("name","malaria_model")

    return fn, siso_cs_to_np(t, L), siso_cs_to_np(t, F)

def siso_cs_to_np(cs_in, cs_out):
    """
    Takes SISO casadi SXFunction and makes a function out of it that works like a numpy function. Input must be SX('t')
    """

    def npfunction(npinput):
        """ autogenerated version of csfunction """
        csf = cs.SXFunction([cs_in], [cs_out])
        csf.init()
        out = []
        for inp in npinput:
            csf.setInput(inp)
            csf.evaluate()
            out.append(csf.getOutput().toArray().flatten())
        return np.array(out).flatten()

    return npfunction

