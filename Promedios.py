# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:36:48 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
import seaborn as sns
from plyer import notification
get_ipython().run_line_magic('matplotlib', 'qt5')
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\2023-01-15-night\Aviones\Aviones y pajaros'
os.chdir(directory)

#%%

lugar = os.listdir(directory)
Presiones = []
Sonidos = []

for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
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


audio, pressure, name, fs = datos_normalizados_2(0, 0, 5)
time = np.linspace(0, len(pressure)/fs, len(pressure))

#%% Importo los datos importantes

with open('Datos CaFF073-RoVio_2023_01_18.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos = pickle.load(f)

#%%
time_series_list = []


for indice in range(Datos.shape[0]):
    
    ti,tf = Datos.loc[indice,'Tiempo inicial normalizacion'],Datos.loc[indice,'Tiempo final normalizacion']
    tiempo_inicial = Datos.loc[indice,'Tiempo inicial avion']
    
    audio, pressure, name, fs = datos_normalizados_2(indice, ti, tf)
    
    
    peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), 
                          prominence=.001)
    spline_amplitude_sound = UnivariateSpline(time[peaks_sonido], audio[peaks_sonido], s=0, k=3)
    
    interpolado = spline_amplitude_sound(time)

    
    lugar_maximos = []
    maximos = np.loadtxt(f'{name}_maximos.txt')
    for i in maximos:
        lugar_maximos.append(int(i))
    # lugar_mininmos = []
    # for i in
    
    # lugar_maximos,_ = find_peaks(pressure,prominence=1,height=0, distance=int(fs*0.1))
    
    periodo = np.diff(time[lugar_maximos])
    tiempo = time-tiempo_inicial
    time_series_list.append({'time':tiempo,'interpolado':interpolado,
                             'time maximos':tiempo[lugar_maximos],
                             'presion':pressure[lugar_maximos],
                             'time periodo':tiempo[lugar_maximos][1::],'periodo':1/periodo})

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)    

#%%
#Sonido
start_time = min(ts["time"][0] for ts in time_series_list)
end_time = max(ts["time"][-1] for ts in time_series_list)
common_time_base_sonido = np.linspace(start_time, end_time, fs)

interpolated_data = []
for ts in time_series_list:
    interp_func = interp1d(ts["time"], ts["interpolado"], bounds_error=False, fill_value=np.nan)
    interpolated_data.append(interp_func(common_time_base_sonido))

# Convert the list of arrays to a 2D numpy array
interpolated_data = np.array(interpolated_data)

# Compute the average, ignoring NaNs
average_data_sonido = np.nanmean(interpolated_data, axis=0)
std_data_sonido = np.nanstd(interpolated_data,axis=0)/np.sqrt(Datos.shape[0])

#Maximos
start_time = min(ts["time maximos"][0] for ts in time_series_list)
end_time = max(ts["time maximos"][-1] for ts in time_series_list)
common_time_base_maximos = np.linspace(start_time, end_time, 300)

interpolated_data = []
for ts in time_series_list:
    interp_func = interp1d(ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
    interpolated_data.append(interp_func(common_time_base_maximos))

# Convert the list of arrays to a 2D numpy array
interpolated_data = np.array(interpolated_data)

# Compute the average, ignoring NaNs
average_data_maximos = np.nanmean(interpolated_data, axis=0)
std_data_maximos = np.nanstd(interpolated_data,axis=0)/np.sqrt(Datos.shape[0])

#Periodo
start_time = min(ts["time periodo"][0] for ts in time_series_list)
end_time = max(ts["time periodo"][-1] for ts in time_series_list)
common_time_base_periodo = np.linspace(start_time, end_time, 300)

interpolated_data = []
for ts in time_series_list:
    interp_func = interp1d(ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
    interpolated_data.append(interp_func(common_time_base_periodo))

# Convert the list of arrays to a 2D numpy array
interpolated_data = np.array(interpolated_data)

# Compute the average, ignoring NaNs
average_data_periodo = np.nanmean(interpolated_data, axis=0)
std_data_periodo = np.nanstd(interpolated_data,axis=0)/np.sqrt(Datos.shape[0])


plt.close('Promedio')
plt.figure('Promedio',figsize=(14, 7))
for ts in time_series_list:
    plt.subplot(311)
    plt.plot(ts["time"], ts["interpolado"],color='k', alpha=0.1,solid_capstyle='projecting')
    plt.subplot(312)
    plt.plot(ts["time maximos"], ts["presion"],color='k', alpha=0.1,solid_capstyle='projecting')
    plt.subplot(313)
    plt.plot(ts["time periodo"], ts["periodo"],color='k', alpha=0.1,solid_capstyle='projecting')
plt.subplot(311)
plt.errorbar(common_time_base_sonido, average_data_sonido, std_data_sonido, label='Average', color='r', linewidth=2)
plt.legend(fancybox=True,shadow=True)
plt.ylabel("Audio (arb. u.)")
plt.subplot(312)
plt.errorbar(common_time_base_maximos, average_data_maximos, std_data_maximos, label='Average', color='r', linewidth=2)
plt.legend(fancybox=True,shadow=True)
plt.ylabel("Pressure (arb. u.)")
plt.subplot(313)
plt.errorbar(common_time_base_periodo, average_data_periodo, std_data_periodo, label='Average', color='r', linewidth=2)
plt.legend(fancybox=True,shadow=True)
plt.ylabel('Rate (sec)')
plt.xlabel("Time (sec)")        
plt.tight_layout()


notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)    

#%%

promedios = {'tiempo sonido':common_time_base_sonido,'promedio sonido':average_data_sonido,'errores sonido': std_data_sonido,
             'tiempo maximos':common_time_base_maximos,'promedio maximos':average_data_maximos,'errores maximos': std_data_maximos,
             'tiempo rate':common_time_base_periodo,'promedio rate':average_data_periodo,'errores periodo': std_data_periodo}
#%%

with open('Interpolado, maximos y rate de CaFF028-RoNe_2023_01_31.pkl', 'wb') as file:
    pickle.dump(time_series_list, file)
with open('promedios de CaFF028-RoNe_2023_01_31.pkl', 'wb') as file:
    pickle.dump(promedios, file)
    
#%% Grafico que puedo abrir ya teniendo los datos
#%%
with open('Interpolado, maximos y rate de CaFF073-RoVio_2023_01_15.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    time_series_list = pickle.load(f)
    
with open('promedios de CaFF073-RoVio_2023_01_15.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    promedios = pickle.load(f)


#%%
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

for ts in time_series_list:
    ax[0].plot(ts["time"], ts["interpolado"], color='k', alpha=0.1, solid_capstyle='projecting')
    ax[1].plot(ts["time maximos"], ts["presion"], color='k', alpha=0.1, solid_capstyle='projecting')
    ax[2].plot(ts["time periodo"], ts["periodo"], color='k', alpha=0.1, solid_capstyle='projecting')

ax[0].errorbar(promedios['tiempo sonido'], promedios['promedio sonido'], promedios['errores sonido']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
ax[0].legend(fancybox=True, shadow=True)
ax[0].set_ylabel("Audio (arb. u.)")

ax[1].errorbar(promedios['tiempo maximos'], promedios['promedio maximos'], promedios['errores maximos']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
ax[1].legend(fancybox=True, shadow=True)
ax[1].set_ylabel("Pressure (arb. u.)")

ax[2].errorbar(promedios['tiempo rate'], promedios['promedio rate'], promedios['errores periodo']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
ax[2].legend(fancybox=True, shadow=True)
ax[2].set_ylabel('Rate (sec)')
ax[2].set_xlabel("Time (sec)")

plt.tight_layout()

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)    