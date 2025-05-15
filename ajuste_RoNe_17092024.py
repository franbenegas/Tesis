#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:42:08 2024

@author: facu
"""
import os 
import numpy as np
import matplotlib.pyplot as plt


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

# data = np.loadtxt("/home/facu/Downloads/average RoNe rate.txt", delimiter=',')
# time_data = data[0]
# rate = data[1]
# error = data[2]

# data_sound = np.loadtxt("/home/facu/Downloads/average RoNe sonido.txt", delimiter=',')
# time_sound = data_sound[0][::100]
# sound = data_sound[1][::100]
# error_s = data_sound[2][::100]
#%%
fig = plt.figure(figsize=(15,9))
plt.errorbar(time_rate_RoNe, rate_RoNe, yerr=error_rate_RoNe)
fig.tight_layout()
#%%

def f(t, a, b, t0, c):
    return a * np.exp( b * (t-t0)) + c

fig = plt.figure(figsize=(15,9))
plt.errorbar(time_rate_RoNe, rate_RoNe, yerr=error_rate_RoNe)
plt.plot(time_rate_RoNe, f(time_rate_RoNe, 1, .75, 3, 1.25))
plt.plot(time_rate_RoNe, f(time_rate_RoNe, 1, -.05, 3, 1.5))
plt.xlabel("time (s)")
plt.ylabel("rate (Hz)")
plt.ylim([1,3])
fig.tight_layout()

#%% Hacia ecuaciones dinamicas
def gauss(t, mu, sigma, c, D):
    return c  + D * np.exp(- (t-mu)**2 / sigma)


synth_sound = gauss(time_sound, 8.4, 25, .8, 4)


fig, ax = plt.subplots(2, 1, figsize=(14,5), sharex=True)
ax[0].errorbar(time_sound, sound, yerr=error_s)
ax[0].plot(time_sound, synth_sound)
ax[0].plot(time_sound[:-1], np.diff(synth_sound)*10)
ax[0].plot(time_sound[:-2], np.diff(np.diff(synth_sound)*10)*10)



ax[1].errorbar(time_data, rate, yerr=error)
ax[1].plot(time_data, f(time_data, 1, .75, 3, 1.25))
ax[1].plot(time_data, f(time_data, 1, -.05, 3, 1.5))
ax[1].set_ylim([1,3])
#%%

def sigm(x):
    return 1 / ( 1 + np.exp(-10*x) ) - 0.5

sound_t = np.concatenate( (np.diff(synth_sound), [0]))
sound_tt = np.concatenate( (np.diff(sound_t), [0]))

fig, ax = plt.subplots(2, 1, figsize=(14,5), sharex=True)
ax[0].errorbar(time_sound, sound, yerr=error_s)
ax[0].plot(time_sound, synth_sound)
ax[0].plot(time_sound, sound_t*10)
ax[0].plot(time_sound, sound_tt*100)
ax[0].plot(time_sound, (np.sign(sound_t) + 1) * ( np.sign(sound_tt) + 1 )* 5)



ax[1].errorbar(time_data, rate, yerr=error)
ax[1].plot(time_data, f(time_data, 1, .75, 3, 1.25))
ax[1].plot(time_data, f(time_data, 1, -.05, 3, 1.5))
ax[1].set_ylim([1,3])

#%%

fig, ax = plt.subplots(2, 1, figsize=(14,5), sharex=True)
ax[0].errorbar(time_sound, sound, yerr=error_s)
ax[0].plot(time_sound, synth_sound)
ax[0].plot(time_sound, sound_t*10)
ax[0].plot(time_sound, sound_tt*100)
ax[0].plot(time_sound, ( (sound_t)>0) * (  sound_tt > 0  ) )


