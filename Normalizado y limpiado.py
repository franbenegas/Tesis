# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:33:52 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks
import pickle


get_ipython().run_line_magic('matplotlib', 'qt5')


directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia\CaFF028-RoNe\2023-02-01-day\Aviones y pajaros'
os.chdir(directory)
#%%
lugar = os.listdir(directory)
Presiones = []
Sonidos = []

for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0:3]=='pre':
        Presiones.append(file)
#%%

def datos_normalizados_2(indice, ti, tf):
    
    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]
    
    fs,audio = wavfile.read(Sound)
    fs,pressure = wavfile.read(Pressure)
    
    pressure = pressure-np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)
    
    #funcion que normaliza al [-1, 1]
    def norm11_interval(x, ti, tf, fs):
      x_int = x[int(ti*fs):int(tf*fs)]
      return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1
        
    audio = audio-np.mean(audio)
    audio_norm = audio / np.max(audio)
    
    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs


# Find the longest continuous interval
def find_longest_interval(times, max_gap=1):
    longest_interval = []
    current_interval = [times[0]]

    for i in range(1, len(times)):
        if times[i] - times[i-1] <= max_gap:
            current_interval.append(times[i])
        else:
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]

    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval

    return longest_interval
      
        
#%%
columnas = ['Nombre','Tiempo inicial normalizacion','Tiempo final normalizacion','Tiempo inicial avion','Tiempo final avion']
Datos = pd.DataFrame(columns=columnas)

#%%
indice = 0
ti,tf = 2,5
print(indice+1,len(Sonidos))
audio, pressure, name, fs = datos_normalizados_2(indice, ti, tf)

#defino el tiempo
time = np.linspace(0, len(pressure)/fs, len(pressure))


f, t, Sxx = signal.spectrogram(audio, fs)
# Define the frequency cutoff
frequency_cutoff = 1000  # Adjust as needed #1000 funciona piola
threshold = -10  # dB threshold for detection   -10 se acerca mas

# Convert the spectrogram to dB scale
Sxx_dB =np.log(Sxx)

# Find the indices where the frequency is above the cutoff
freq_indices = np.where(f > frequency_cutoff)[0]

# Identify times where the spectrogram surpasses the threshold
time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
time_above_threshold = t[time_indices]


longest_interval = find_longest_interval(time_above_threshold)

plt.close('all')
plt.figure('Sonido vs presion',figsize=(14,7))
plt.suptitle(f'{name}')
plt.subplot(211)
plt.plot(time, audio)
plt.axvline(x=longest_interval[0], color='k', linestyle='-')
plt.axvline(x=longest_interval[-1], color='k', linestyle='-', label='Avion')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=-1, color='k', linestyle='-')
plt.ylabel("Audio (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.subplot(212)
# plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud')
# plt.colorbar(label='Intensity [dB]')
# plt.axhline(y=frequency_cutoff,color='k', linestyle='--', label='cutoff')
plt.plot(time,pressure)
plt.axvline(x=longest_interval[0], color='k', linestyle='-')
plt.axvline(x=longest_interval[-1], color='k', linestyle='-', label='Avion')
plt.axhline(y=1, color='k', linestyle='-')
plt.axhline(y=-1, color='k', linestyle='-')
plt.ylabel("Pressure (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.xlabel("Time (sec)")
plt.tight_layout()

#%% Los agrego al dataframe
temp_df = pd.DataFrame([[name,ti,tf,longest_interval[0],longest_interval[-1]]], columns=columnas)


Datos = pd.concat([Datos, temp_df], ignore_index=True)

#%% Calculo de maximos 

peaks_maximos,properties_maximos = find_peaks(pressure,prominence=1,height=0, distance=int(fs*0.1))
plt.close('Maximos')
plt.figure('Maximos',figsize=(14,7))
plt.plot(time,pressure)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
plt.axhline(y=0, color='k', linestyle='-')
for i in range(len(peaks_maximos)):
    plt.text( time[peaks_maximos[i]],pressure[peaks_maximos[i]], str(i) )

plt.ylabel("pressure (arb. u.)")
plt.xlabel("Time (sec)")
plt.legend(fancybox=True, shadow=True)
plt.tight_layout()

#%% Aca saco los que me molestan
sacar = [28,33]
picos_limpios = []

for i in range(len(peaks_maximos)):
    if i in sacar:
        None
    else:
        picos_limpios.append(peaks_maximos[i])

#%% los limpio y los enumero
plt.close('Picos limpios')
plt.figure('Picos limpios',figsize=(14,7))
# plt.subplot(211)
plt.plot(time,pressure)
plt.plot(time[picos_limpios],pressure[picos_limpios],'.C1',label='Maximos', ms=10)

for i in range(len(picos_limpios)):
    plt.text(time[picos_limpios[i]],pressure[picos_limpios[i]], str(i) )

plt.ylabel("pressure (arb. u.)")
plt.grid(linestyle='dashed')
plt.legend(fancybox=True, shadow=True)
np.savetxt(f'{name}_maximos.txt', picos_limpios, delimiter=',',newline='\n', fmt='%i',header='Indice Maximos')
#%% Guardo el dataframe

with open(f'Datos2 {name[:-9]}.pkl', 'wb') as file:
    pickle.dump(Datos, file)
#%%
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle(f'{name}')
ax[0].plot(time, audio)
ax[1].plot(time,pressure)
ax[1].plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
ax[2].plot(time[peaks_maximos][1::],1/np.diff(time[peaks_maximos]),'.C0',linestyle='-')