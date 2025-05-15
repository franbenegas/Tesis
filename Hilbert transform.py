# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:26:23 2025

@author: beneg
"""
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import os 
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis'
os.chdir(directory)  # Change the working directory to the specified path
pressure = "pressure_CaFF028-RoNe_2023_01_30-22.18.17.wav"


fs, pressure = wavfile.read(pressure)

time = np.linspace(0, len(pressure) / fs, len(pressure))

# Normalize the audio and pressure data
pressure = pressure - np.mean(pressure)  # Remove mean
pressure_norm = pressure / np.max(pressure)  # Normalize by max value

# --- Normalizar en un intervalo [0, 1] segundos ---
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1



pressure_norm = norm11_interval(pressure_norm, 0, 1, fs)



plt.figure()
plt.plot(pressure_norm)


#%%


from scipy.signal import hilbert

analytic_signal = hilbert(pressure_norm)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * fs

fig, ax = plt.subplots(2,1,figsize = (15,9))

ax[0]. plot(time,pressure_norm)
ax[0].plot(time,amplitude_envelope)
ax[1].plot(time[1:],instantaneous_frequency)


plt.tight_layout()


#%%


analytic_signal = hilbert(pressure_norm)

amplitude = np.real(analytic_signal)
phase = np.imag(analytic_signal)

fig, ax = plt.subplots(2,1,figsize = (15,9),sharex=True)
ax[0]. plot(time,pressure_norm)
ax[0].plot(time,amplitude)
ax[1].plot(time,phase)
plt.tight_layout()