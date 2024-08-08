# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:45:33 2024

@author: beneg
"""




import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
from plyer import notification
get_ipython().run_line_magic('matplotlib', 'qt5')
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia\CaFF028-RoNe'
os.chdir(directory)

carpetas = os.listdir(directory)
pajaritos = '\Aviones y pajaros'
noches_1 = directory + '/' + carpetas[0] + pajaritos
noches_2 = directory + '/' + carpetas[2] + pajaritos
noches_3 = directory + '/' + carpetas[3] + pajaritos
noches_4 = directory + '/' + carpetas[4] + pajaritos

#%%

def datos_normalizados_2(Sonidos,Presiones,indice, ti, tf):
    
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

time = np.linspace(0, 1324503/44150, 1324503)

def process_night_data(directories):
    """
    Process data from multiple directories and return a combined time series list.

    Args:
        directories (list of str): List of directory paths.

    Returns:
        list of dict: Combined time series data from all directories.
    """
    combined_time_series_list = []

    for directory in directories:
        os.chdir(directory)
        files = os.listdir(directory)
        
        presiones = []
        sonidos = []
        datos = []
        
        for file in files:
            if file[0] == 's':
                sonidos.append(file)
            elif file[0] == 'p':
                presiones.append(file)
            elif file[0] == 'D':
                datos.append(file)

        with open(datos[0], 'rb') as f:
            datos = pickle.load(f)
        
        for indice in range(datos.shape[0]):
            ti, tf = datos.loc[indice, 'Tiempo inicial normalizacion'], datos.loc[indice, 'Tiempo final normalizacion']
            tiempo_inicial = datos.loc[indice, 'Tiempo inicial avion']
            
            audio, pressure, name, fs = datos_normalizados_2(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure)/fs, len(pressure))
            
            peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), prominence=.001)
            spline_amplitude_sound = UnivariateSpline(time[peaks_sonido], audio[peaks_sonido], s=0, k=3)
            
            interpolado = spline_amplitude_sound(time)
            
            lugar_maximos = []
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))

            periodo = np.diff(time[lugar_maximos])
            tiempo = time - tiempo_inicial
            combined_time_series_list.append({
                'time': tiempo,
                'interpolado': interpolado,
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time periodo': tiempo[lugar_maximos][1:],
                'periodo': 1/periodo
            })

    return combined_time_series_list

def interpolate_time_series_data(time_series_list, common_time_length_sound=44150, common_time_length_maximos=300):
    # Sound interpolation
    start_time = min(ts["time"][0] for ts in time_series_list)
    end_time = max(ts["time"][-1] for ts in time_series_list)
    common_time_base_sound = np.linspace(start_time, end_time, common_time_length_sound)
    
    interpolated_data_sound = []
    for ts in time_series_list:
        interp_func = interp1d(ts["time"], ts["interpolado"], bounds_error=False, fill_value=np.nan)
        interpolated_data_sound.append(interp_func(common_time_base_sound))
    
    interpolated_data_sound = np.array(interpolated_data_sound)
    
    # Maximums interpolation
    start_time = min(ts["time maximos"][0] for ts in time_series_list)
    end_time = max(ts["time maximos"][-1] for ts in time_series_list)
    common_time_base_maximos = np.linspace(start_time, end_time, common_time_length_maximos)
    
    interpolated_data_maximos = []
    for ts in time_series_list:
        interp_func = interp1d(ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
        interpolated_data_maximos.append(interp_func(common_time_base_maximos))
    
    interpolated_data_maximos = np.array(interpolated_data_maximos)
    
    # Period interpolation
    start_time = min(ts["time periodo"][0] for ts in time_series_list)
    end_time = max(ts["time periodo"][-1] for ts in time_series_list)
    common_time_base_periodo = np.linspace(start_time, end_time, common_time_length_maximos)
    
    interpolated_data_periodo = []
    for ts in time_series_list:
        interp_func = interp1d(ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
        interpolated_data_periodo.append(interp_func(common_time_base_periodo))
    
    interpolated_data_periodo = np.array(interpolated_data_periodo)
    
    # Compute averages and standard errors, considering NaNs
    def compute_average_and_std(data):
        average = np.nanmean(data, axis=0)
        count_non_nan = np.sum(~np.isnan(data), axis=0)
        std_error = np.nanstd(data, axis=0) / np.sqrt(count_non_nan)
        return average, std_error, count_non_nan
    
    average_data_sound, std_data_sound, non_nan_sound = compute_average_and_std(interpolated_data_sound)
    average_data_maximos, std_data_maximos, non_nan_maximos = compute_average_and_std(interpolated_data_maximos)
    average_data_periodo, std_data_periodo, non_nan_periodo = compute_average_and_std(interpolated_data_periodo)
    
    return {
        'common_time_base_sound': common_time_base_sound,
        'common_time_base_maximos': common_time_base_maximos,
        'common_time_base_periodo': common_time_base_periodo,
        'average_data_sound': average_data_sound,
        'std_data_sound': std_data_sound,
        'non_nan_sound' : non_nan_sound,
        'average_data_maximos': average_data_maximos,
        'std_data_maximos': std_data_maximos,
        'non_nan_maximos' : non_nan_maximos,
        'average_data_periodo': average_data_periodo,
        'std_data_periodo': std_data_periodo,
        'non_nan_periodo' : non_nan_periodo
    }

#%%

time_series_list_noche_1 = process_night_data([noches_1])
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
time_series_list_noche_2 = process_night_data([noches_2])
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
time_series_list_noche_3 = process_night_data([noches_3])
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
time_series_list_noche_4 = process_night_data([noches_4])
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)


#%%


# Example usage
interpolated_data_noche_1 = interpolate_time_series_data(time_series_list_noche_1, 44150, 300)


interpolated_data_noche_2 = interpolate_time_series_data(time_series_list_noche_2, 44150, 300)


interpolated_data_noche_3 = interpolate_time_series_data(time_series_list_noche_3, 44150, 300)


interpolated_data_noche_4 = interpolate_time_series_data(time_series_list_noche_4, 44150, 300)


#%%



fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe dia')

for ts in time_series_list_noche_1:
    ax[0].plot(ts["time"], ts["interpolado"], color='k', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[0]}')
    ax[1].plot(ts["time maximos"], ts["presion"], color='k', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[0]}')
    ax[2].plot(ts["time periodo"], ts["periodo"], color='k', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[0]}')
   
for ts in time_series_list_noche_2:
    ax[0].plot(ts["time"], ts["interpolado"], color='C0', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[2]}')
    ax[1].plot(ts["time maximos"], ts["presion"], color='C0', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[2]}')
    ax[2].plot(ts["time periodo"], ts["periodo"], color='C0', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[2]}')
   
for ts in time_series_list_noche_3:
    ax[0].plot(ts["time"], ts["interpolado"], color='C1', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[3]}')
    ax[1].plot(ts["time maximos"], ts["presion"], color='C1', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[3]}')
    ax[2].plot(ts["time periodo"], ts["periodo"], color='C1', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[3]}')
plt.legend(fancybox=True,shadow=True)  
for ts in time_series_list_noche_4:
    ax[0].plot(ts["time"], ts["interpolado"], color='C2', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[4]}')
    ax[1].plot(ts["time maximos"], ts["presion"], color='C2', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[4]}')
    ax[2].plot(ts["time periodo"], ts["periodo"], color='C2', alpha=0.1, solid_capstyle='projecting',label=f'{carpetas[4]}')


legend_handles = [
    mpatches.Patch(color='k', label=carpetas[0]),
    mpatches.Patch(color='C0', label=carpetas[2]),
    mpatches.Patch(color='C1', label=carpetas[3]),
    mpatches.Patch(color='C2', label=carpetas[4])
]

# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)
plt.tight_layout()

#%% Junto las de 2022 y 2023 por separado y las comparo

directories = [noches_2,noches_3,noches_4]
time_series_list_2023 = process_night_data(directories)
time_seires_list_2022 = process_night_data([noches_1])

interpolated_data_2023 = interpolate_time_series_data(time_series_list_2023, 44150, 300)
interpolated_data_2022 = interpolate_time_series_data(time_seires_list_2022, 44150, 300)

#%%

average_sonido_noche_1 = interpolated_data_2023['average_data_sound']
std_sonido_noche_1 = interpolated_data_2023['std_data_sound']
average_maximos_noche_1 = interpolated_data_2023['average_data_maximos']
std_maximos_noche_1 = interpolated_data_2023['std_data_maximos']
average_rate_noche_1 = interpolated_data_2023['average_data_periodo']
std_rate_noche_1 = interpolated_data_2023['std_data_periodo']

average_sonido_noche_2 = interpolated_data_2022['average_data_sound']
std_sonido_noche_2 = interpolated_data_2022['std_data_sound']
average_maximos_noche_2 = interpolated_data_2022['average_data_maximos']
std_maximos_noche_2 = interpolated_data_2022['std_data_maximos']
average_rate_noche_2 = interpolated_data_2022['average_data_periodo']
std_rate_noche_2 = interpolated_data_2022['std_data_periodo']


fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe dia')

# Set custom colors
colors = ['k', 'C0']

# Night 1
ax[0].errorbar(interpolated_data_2023['common_time_base_sound'], average_sonido_noche_1, std_sonido_noche_1, label=carpetas[0], color=colors[0])
ax[1].errorbar(interpolated_data_2023['common_time_base_maximos'], average_maximos_noche_1, std_maximos_noche_1, label=carpetas[0], color=colors[0])
ax[2].errorbar(interpolated_data_2023['common_time_base_periodo'], average_rate_noche_1, std_rate_noche_1, label=carpetas[0], color=colors[0])

# Night 2
ax[0].errorbar(interpolated_data_2022['common_time_base_sound'], average_sonido_noche_2, std_sonido_noche_2, label=carpetas[2], color=colors[1])
ax[1].errorbar(interpolated_data_2022['common_time_base_maximos'], average_maximos_noche_2, std_maximos_noche_2, label=carpetas[2], color=colors[1])
ax[2].errorbar(interpolated_data_2022['common_time_base_periodo'], average_rate_noche_2, std_rate_noche_2, label=carpetas[2], color=colors[1])


legend_handles = [
    mpatches.Patch(color='k', label='2023'),
    mpatches.Patch(color='C0', label='2022'),
]

# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)
plt.tight_layout()

#%% Aca hago todos los datos de RoNe juntos

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia\CaFF028-RoNe'
os.chdir(directory)

carpetas = os.listdir(directory)
pajaritos = '\Aviones y pajaros'
noches_1 = directory + '/' + carpetas[0] + pajaritos
noches_2 = directory + '/' + carpetas[2] + pajaritos
noches_3 = directory + '/' + carpetas[3] + pajaritos
noches_4 = directory + '/' + carpetas[4] + pajaritos

directories = [noches_1,noches_2,noches_3,noches_4]
time_series_list_total = process_night_data(directories)


interpolated_data_total = interpolate_time_series_data(time_series_list_total, 44150, 300)

average_sonido_noche_1 = interpolated_data_total['average_data_sound']
std_sonido_noche_1 = interpolated_data_total['std_data_sound']
average_maximos_noche_1 = interpolated_data_total['average_data_maximos']
std_maximos_noche_1 = interpolated_data_total['std_data_maximos']
average_rate_noche_1 = interpolated_data_total['average_data_periodo']
std_rate_noche_1 = interpolated_data_total['std_data_periodo']

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe dia')

ax[0].errorbar(interpolated_data_total['common_time_base_sound'], average_sonido_noche_1, std_sonido_noche_1, color=colors[0])
ax[1].errorbar(interpolated_data_total['common_time_base_maximos'], average_maximos_noche_1, std_maximos_noche_1, color=colors[0])
ax[2].errorbar(interpolated_data_total['common_time_base_periodo'], average_rate_noche_1, std_rate_noche_1, color=colors[0])


legend_handles = [
    mpatches.Patch(color='k', label='Dia'),
    mpatches.Patch(color='C0', label='Noche'),
]

# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)
plt.tight_layout()

#%% Aca comparo noche y dia
## Dia
colors = ['k', 'C0']

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia\CaFF028-RoNe'
os.chdir(directory)

carpetas = os.listdir(directory)
pajaritos = '\Aviones y pajaros'
noches_1 = directory + '/' + carpetas[0] + pajaritos
noches_2 = directory + '/' + carpetas[2] + pajaritos
noches_3 = directory + '/' + carpetas[3] + pajaritos
noches_4 = directory + '/' + carpetas[4] + pajaritos
# time = np.linspace(0, 1324503/44150, 1324503)

directories = [noches_1,noches_2,noches_3,noches_4]
time_series_list_total = process_night_data(directories)


interpolated_data_total = interpolate_time_series_data(time_series_list_total, 44150, 300)

average_sonido_dia = interpolated_data_total['average_data_sound']
std_sonido_dia = interpolated_data_total['std_data_sound']
average_maximos_dia = interpolated_data_total['average_data_maximos']
std_maximos_dia = interpolated_data_total['std_data_maximos']
average_rate_dia = interpolated_data_total['average_data_periodo']
std_rate_dia = interpolated_data_total['std_data_periodo']

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe dia')

ax[0].errorbar(interpolated_data_total['common_time_base_sound'], average_sonido_dia, std_sonido_dia, color=colors[0])
ax[1].errorbar(interpolated_data_total['common_time_base_maximos'], average_maximos_dia, std_maximos_dia, color=colors[0])
ax[2].errorbar(interpolated_data_total['common_time_base_periodo'], average_rate_dia, std_rate_dia, color=colors[0])



## Noche
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe'
os.chdir(directory)

carpetas = os.listdir(directory)
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = directory + '/' + carpetas[0] + pajaritos
noches_2 = directory + '/' + carpetas[1] + pajaritos
noches_3 = directory + '/' + carpetas[2] + pajaritos
# time = np.linspace(0, 2649006/44150, 2649006)
directories = [noches_1,noches_2,noches_3]
time_series_list_noche = process_night_data(directories)

interpolated_data_noche = interpolate_time_series_data(time_series_list_noche, 44150, 300)

average_sonido_noche = interpolated_data_noche['average_data_sound']
std_sonido_noche = interpolated_data_noche['std_data_sound']
average_maximos_noche = interpolated_data_noche['average_data_maximos']
std_maximos_noche = interpolated_data_noche['std_data_maximos']
average_rate_noche = interpolated_data_noche['average_data_periodo']
std_rate_noche = interpolated_data_noche['std_data_periodo']

ax[0].errorbar(interpolated_data_noche['common_time_base_sound'], average_sonido_noche, std_sonido_noche, color='C0')
ax[1].errorbar(interpolated_data_noche['common_time_base_maximos'], average_maximos_noche, std_maximos_noche, color='C0')
ax[2].errorbar(interpolated_data_noche['common_time_base_periodo'], average_rate_noche, std_rate_noche, color='C0')


legend_handles = [
    mpatches.Patch(color='k', label='Dia'),
    mpatches.Patch(color='C0', label='Noche'),
]

# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)
plt.tight_layout()


#%%
# Find the minimum and maximum times in 'time' across all time series
min_time = min(ts['time'][0] for ts in time_series_list_total)
max_time = max(ts['time'][-1] for ts in time_series_list_total)

# Find the index where common_time_base_sound is closest to min_time
index_min_sound = np.argmin(np.abs(interpolated_data_noche['common_time_base_sound'] - min_time))
# Find the index where common_time_base_sound is closest to max_time
index_max_sound = np.argmin(np.abs(interpolated_data_noche['common_time_base_sound'] - max_time))

# Slicing the sound data based on the indices
common_time_base_sound_restricted = interpolated_data_noche['common_time_base_sound'][index_min_sound:index_max_sound+1]
average_sonido_noche_restricted = average_sonido_noche[index_min_sound:index_max_sound+1]
std_sonido_noche_restricted = std_sonido_noche[index_min_sound:index_max_sound+1]
non_nan_sound_noche_restricted = interpolated_data_noche['non_nan_sound'][index_min_sound:index_max_sound+1]

# Find the minimum and maximum times in 'time maximos' across all time series
min_time_maximos = min(ts['time maximos'][0] for ts in time_series_list_total)
max_time_maximos = max(ts['time maximos'][-1] for ts in time_series_list_total)

# Find the index where common_time_base_maximos is closest to min_time_maximos
index_min_maximos = np.argmin(np.abs(interpolated_data_noche['common_time_base_maximos'] - min_time_maximos))
# Find the index where common_time_base_maximos is closest to max_time_maximos
index_max_maximos = np.argmin(np.abs(interpolated_data_noche['common_time_base_maximos'] - max_time_maximos))

# Slicing the maximos data based on the indices
common_time_base_maximos_restricted = interpolated_data_noche['common_time_base_maximos'][index_min_maximos:index_max_maximos+1]
average_maximos_noche_restricted = average_maximos_noche[index_min_maximos:index_max_maximos+1]
std_maximos_noche_restricted = std_maximos_noche[index_min_maximos:index_max_maximos+1]
non_nan_maximos_noche_restricted = interpolated_data_noche['non_nan_maximos'][index_min_maximos:index_max_maximos+1]

# Find the minimum and maximum times in 'time periodo' across all time series
min_time_periodo = min(ts['time periodo'][0] for ts in time_series_list_total)
max_time_periodo = max(ts['time periodo'][-1] for ts in time_series_list_total)

# Find the index where common_time_base_periodo is closest to min_time_periodo
index_min_periodo = np.argmin(np.abs(interpolated_data_noche['common_time_base_periodo'] - min_time_periodo))
# Find the index where common_time_base_periodo is closest to max_time_periodo
index_max_periodo = np.argmin(np.abs(interpolated_data_noche['common_time_base_periodo'] - max_time_periodo))

# Slicing the periodo data based on the indices
common_time_base_periodo_restricted = interpolated_data_noche['common_time_base_periodo'][index_min_periodo:index_max_periodo+1]
average_rate_noche_restricted = average_rate_noche[index_min_periodo:index_max_periodo+1]
std_rate_noche_restricted = std_rate_noche[index_min_periodo:index_max_periodo+1]
non_nan_rate_noche_restricted = interpolated_data_noche['non_nan_periodo'][index_min_periodo:index_max_periodo+1]

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe')

ax[0].errorbar(interpolated_data_total['common_time_base_sound'], average_sonido_dia, std_sonido_dia, color=colors[0])
ax[1].errorbar(interpolated_data_total['common_time_base_maximos'], average_maximos_dia, std_maximos_dia, color=colors[0])
ax[2].errorbar(interpolated_data_total['common_time_base_periodo'], average_rate_dia, std_rate_dia, color=colors[0])


# Sound plot
ax[0].errorbar(common_time_base_sound_restricted, average_sonido_noche_restricted, std_sonido_noche_restricted, color='C0')

# Maximos plot
ax[1].errorbar(common_time_base_maximos_restricted, average_maximos_noche_restricted, std_maximos_noche_restricted, color='C0')

# Periodo plot
ax[2].errorbar(common_time_base_periodo_restricted, average_rate_noche_restricted, std_rate_noche_restricted, color='C0')


legend_handles = [
    mpatches.Patch(color='k', label='Dia'),
    mpatches.Patch(color='C0', label='Noche'),
]
ax[0].set_ylabel('Sound (arb. u.)')
ax[1].set_ylabel('Pressure (arb. u.)')
ax[2].set_ylabel('Rate (Hz)')
ax[2].set_xlabel('Time (s)')
# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)
plt.tight_layout()



#%%
def welch_statistic(prom1, prom2, std1, std2, N1, N2):
    return (prom1 - prom2) / np.sqrt((std1**2) / N1 + (std2**2) / N2)

def degrees_freedom(std1, std2, N1, N2):
    # Calculate the numerator: (std1^2 / N1 + std2^2 / N2)^2
    numerator = ( (std1**2 / N1) + (std2**2 / N2) ) ** 2
    
    # Calculate the denominator for each part
    denom_part1 = (std1**2 / N1) ** 2 / (N1 - 1)
    denom_part2 = (std2**2 / N2) ** 2 / (N2 - 1)
    
    # Sum of the denominator parts
    denominator = denom_part1 + denom_part2
    
    # Degrees of freedom formula
    degrees_of_freedom = numerator / denominator
    
    return degrees_of_freedom


def calculate_welch_metric(time_base_dia, avg_dia, std_dia, N_dia, time_base_noche, avg_noche, std_noche, N_noche):
    welch_metrics = []
    degrees_of_freed = []
    for i in range(len(time_base_noche)):
        time_noche = time_base_noche[i]
        closest_idx_dia = np.argmin(np.abs(time_base_dia - time_noche))
        
        prom_dia = avg_dia[closest_idx_dia]
        var_dia = std_dia[closest_idx_dia]
        non_nan_dia = N_dia[closest_idx_dia]
        prom_noche = avg_noche[i]
        var_noche = std_noche[i]
        non_nan_noche = N_noche[i]
        
        welch_metric = welch_statistic(prom_noche, prom_dia, var_noche, var_dia, non_nan_noche, non_nan_dia)
        welch_metrics.append(welch_metric)
        degree_of_freedom = degrees_freedom(var_noche, var_dia, non_nan_noche, non_nan_dia)
        degrees_of_freed.append(degree_of_freedom)
        
    return welch_metrics, degrees_of_freed

# Calculate Welch metrics for sound, maximos, and periodo
welch_metrics_sound, degrees_of_freedon_sound = calculate_welch_metric(interpolated_data_total['common_time_base_sound'], average_sonido_dia, std_sonido_dia, 
                                             interpolated_data_total['non_nan_sound'],
                                             common_time_base_sound_restricted, average_sonido_noche_restricted, std_sonido_noche_restricted, 
                                             non_nan_sound_noche_restricted)

welch_metrics_maximos, degrees_of_freedon_maximos = calculate_welch_metric(interpolated_data_total['common_time_base_maximos'], average_maximos_dia, std_maximos_dia, 
                                               interpolated_data_total['non_nan_maximos'],
                                               common_time_base_maximos_restricted, average_maximos_noche_restricted, std_maximos_noche_restricted, 
                                               non_nan_maximos_noche_restricted)

welch_metrics_periodo, degrees_of_freedon_periodo = calculate_welch_metric(interpolated_data_total['common_time_base_periodo'], 
                                               average_rate_dia / np.mean(average_rate_dia[0:20]), std_rate_dia, 
                                               interpolated_data_total['non_nan_periodo'],
                                               common_time_base_periodo_restricted, average_rate_noche_restricted, std_rate_noche_restricted,
                                               non_nan_rate_noche_restricted)

#%%

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoNe')

# Original errorbar plots
ax[0].errorbar(interpolated_data_total['common_time_base_sound'], average_sonido_dia, std_sonido_dia, color='k', alpha=0.5)
ax[1].errorbar(interpolated_data_total['common_time_base_maximos'], average_maximos_dia, std_maximos_dia, color='k', alpha=0.5)
ax[2].errorbar(interpolated_data_total['common_time_base_periodo'][:-1], average_rate_dia[:-1]/np.mean(average_rate_dia[0:20]), std_rate_dia[:-1], color='k', alpha=0.5)

# Sound plot
ax[0].errorbar(common_time_base_sound_restricted, average_sonido_noche_restricted, std_sonido_noche_restricted, color='C0', alpha=0.5)

# Maximos plot
ax[1].errorbar(common_time_base_maximos_restricted, average_maximos_noche_restricted, std_maximos_noche_restricted, color='C0', alpha=0.5)

# Periodo plot
ax[2].errorbar(common_time_base_periodo_restricted, average_rate_noche_restricted, std_rate_noche_restricted, color='C0', alpha=0.5)

# Adding Welch metric plots with twin axes
ax0_twin = ax[0].twinx()
ax1_twin = ax[1].twinx()
ax2_twin = ax[2].twinx()

ax0_twin.plot(common_time_base_sound_restricted, welch_metrics_sound, color='C2')
ax1_twin.plot(common_time_base_maximos_restricted, welch_metrics_maximos, color='C2')
ax2_twin.plot(common_time_base_periodo_restricted[:-1], welch_metrics_periodo[:-1], color='C2')

# Setting labels and legends
legend_handles = [
    mpatches.Patch(color='k', label='Dia'),
    mpatches.Patch(color='C0', label='Noche'),
    mpatches.Patch(color='C2', label='Welch Metric')
]

ax[0].set_ylabel('Sound (arb. u.)')
ax[1].set_ylabel('Pressure (arb. u.)')
ax[2].set_ylabel('Rate (Hz)')
ax[2].set_xlabel('Time (s)')
ax1_twin.set_ylabel('Welch Metric')
# Create the legend with custom handles
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True)

plt.tight_layout()


#%%

plt.figure()
plt.plot(degrees_of_freedon_maximos)
plt.plot(degrees_of_freedon_periodo)