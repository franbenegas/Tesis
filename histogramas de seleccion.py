# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:08:26 2025

@author: beneg
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from scipy.signal import find_peaks

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
# os.chdir(directory)


data = np.load('cocientes.npy')



fig, ax = plt.subplots(figsize=(14,7))
ax.hist(data,100)
ax.axvline(1.59, c='r', zorder=10)
ax.set_xlabel('Cociente')
ax.set_label('Count')

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-30-night\Aviones\Aviones y pajaros'
os.chdir(directory)
sound = 'sound_CaFF028-RoNe_2023_01_30-22.18.17.wav'


fs, audio = wavfile.read(sound)
time = np.linspace(0, len(audio) / fs, len(audio)) 

audio = audio - np.mean(audio)  # Remove mean
audio_norm = audio / np.max(audio)  # Normalize by max value

# Function to normalize the data to [-1, 1] in a given time interval
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

audio_norm = norm11_interval(audio_norm, 0, 5, fs)

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(time,audio_norm)

#%%

# Function to find the longest interval of consecutive time values that are within a gap threshold
def find_longest_interval(times, max_gap=1):
    longest_interval = []
    current_interval = [times[0]]

    # Iterate through time values and check the gap between consecutive times
    for i in range(1, len(times)):
        if times[i] - times[i-1] <= max_gap:
            current_interval.append(times[i])
        else:
            # If current interval is longer than the previous longest, update it
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]

    # Final check for the last interval
    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval

    return longest_interval


from scipy import signal

f, t, Sxx = signal.spectrogram(audio_norm, fs)
frequency_cutoff = 1000  # Frequency cutoff in Hz
# Find the index of the closest frequency
freq_idx = np.argmin(np.abs(f - frequency_cutoff))

# Extract the log values for the selected frequency
selected_ln_values = np.log(Sxx[freq_idx, :])

Sxx_dB = np.log(Sxx)
threshold = -10# Find where frequencies exceed the threshold
freq_indices = np.where(f > frequency_cutoff)[0]
time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
time_above_threshold = t[time_indices]
longest_interval = find_longest_interval(time_above_threshold)



plt.figure()
plt.title('Distribucion de intervalos temporales')
plt.hist(time_above_threshold,bins=50, color='skyblue', edgecolor='black')

# Plot histogram
plt.figure(figsize=(10, 6))
plt.title('Decibeles en la frecuencia de corte')
plt.hist(selected_ln_values, bins=100, color='skyblue', edgecolor='black')
plt.axvline(threshold, c='r', zorder=10)


plt.figure()
plt.plot(time,audio_norm)
plt.axvline(longest_interval[0],color='k')
plt.axvline(longest_interval[-1],color='k')

#%%
plt.figure()
plt.hist(audio_norm,100)
#%%


peaks_sonido, _ = find_peaks(audio_norm, height=0, distance=int(fs * 0.1), prominence=.001)
time_sonido = time[peaks_sonido]
audio = audio_norm[peaks_sonido]
# Find the split index based on longest_interval[0]
split_idx = np.argmin(np.abs(time_sonido - longest_interval[0]))
# stop = np.argmin(np.abs(timesonido - 30))
# Split the audio signal
audio_before = audio[:split_idx]  
audio_after = audio[split_idx:]   

# Plot histograms
plt.figure(figsize=(10, 5))
plt.title("Histogramas de maximos")
plt.hist(abs(1- audio_before), bins=100, color='blue', alpha=0.7, edgecolor='black',label='Before')
plt.hist(abs(1- audio_after), bins=100, color='red', alpha=0.3, edgecolor='black',label='After')
plt.legend(fancybox=True, shadow=True)

