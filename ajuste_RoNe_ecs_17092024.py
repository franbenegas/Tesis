#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:27:52 2024

@author: facu
"""

import os
import numpy as np
import matplotlib.pyplot as plt

#RK4  
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

#%%
data = np.loadtxt(r"C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-30-night\Aviones\Aviones y pajaros/average RoNe rate", delimiter=',')
time_data = data[0]
rate = data[1]
error = data[2]

# data_sound = np.loadtxt(r"C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-30-night\Aviones\Aviones y pajaros/average RoNe sonido", delimiter=',')
time_sound = data_sound[0]
sound = data_sound[1]
error_s = data_sound[2]

#%%

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_RoNe = np.loadtxt('average RoNe sonido 300', delimiter=',')
Pressure_RoNe = np.loadtxt('average RoNe pressure', delimiter=',')
Rate_RoNe = np.loadtxt('average RoNe rate', delimiter=',')

time_sound_RoNe = Sound_RoNe[0]
sound_RoNe = Sound_RoNe[1]
error_s_RoNe = Sound_RoNe[2]


time_pressure_RoNe = Pressure_RoNe[0]
pressure_RoNe = Pressure_RoNe[1]
error_pressure_RoNe = Pressure_RoNe[2]

time_rate_RoNe = Rate_RoNe[0]
rate_RoNe = Rate_RoNe[1]
error_rate_RoNe = Rate_RoNe[2]

#%%
def f1(x, t, pars):
    x0 , tau, a = [par for par in pars]
    
    dxdt =  a*( x + x0) / tau
    
    return [dxdt]


dt = 0.01
time = np.arange(-40,60, dt)

x = np.zeros_like(time)

#initial condition
x_0 = 1.24
x[0] = x_0

#parameters
pars = [ 0,  4.5, 0.6 ]
pars2 = [ -1.5, -12, 1  ]   


for ix, tt in enumerate(time[:-1]):
    if tt<0:
        x[ix+1] = x_0
    elif tt < 5.57:
        x[ix+1] = rk4(f1, [x[ix]], tt, dt, pars)
    else:
        x[ix+1] = rk4(f1, [x[ix]], tt, dt, pars2)
    
plt.figure(figsize=(25,5))
plt.errorbar(time_rate_RoNe, rate_RoNe, yerr=error_rate_RoNe)
plt.plot(time, x)
# plt.ylim([1 , 3])