# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:32:01 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
from plyer import notification
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches

#%%

def datos_normalizados_2(Sonidos,Presiones,indice, ti, tf):
    
    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]
    
    fs,audio = wavfile.read(Sound)
    fs,pressure = wavfile.read(Pressure)
   
    audio = audio-np.mean(audio)
    audio_norm = audio / np.max(audio)
    
    pressure = pressure-np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)
    
    #funcion que normaliza al [-1, 1]
    def norm11_interval(x, ti, tf, fs):
      x_int = x[int(ti*fs):int(tf*fs)]
      return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1
        
    
    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs

# time = np.linspace(0, 1324503/44150, 1324503)

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

# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)
carpetas = os.listdir(directory)


## RoNe
pajaro = carpetas[0]


subdirectory = os.path.join(directory, pajaro)


dias = os.listdir(subdirectory)
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos
noches_2 = subdirectory + '/' + dias[1] + pajaritos
noches_3 = subdirectory + '/' + dias[2] + pajaritos

directories = [noches_1,noches_2,noches_3]
time_series_list_noche = process_night_data(directories)

interpolated_data_noche_RoNe = interpolate_time_series_data(time_series_list_noche, 44150, 300)

time_sonido_RoNe = interpolated_data_noche_RoNe['common_time_base_sound']
time_maximos_RoNe = interpolated_data_noche_RoNe['common_time_base_maximos']
time_rate_RoNe = interpolated_data_noche_RoNe['common_time_base_periodo']
average_sonido_noche_RoNe = interpolated_data_noche_RoNe['average_data_sound']
std_sonido_noche_RoNe = interpolated_data_noche_RoNe['std_data_sound']
average_maximos_noche_RoNe = interpolated_data_noche_RoNe['average_data_maximos']
std_maximos_noche_RoNe = interpolated_data_noche_RoNe['std_data_maximos']
average_rate_noche_RoNe = interpolated_data_noche_RoNe['average_data_periodo']
std_rate_noche_RoNe = interpolated_data_noche_RoNe['std_data_periodo']


## RoVio
pajaro = carpetas[1]


subdirectory = os.path.join(directory, pajaro)


dias = os.listdir(subdirectory)
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos
noches_2 = subdirectory + '/' + dias[1] + pajaritos
noches_3 = subdirectory + '/' + dias[2] + pajaritos

directories = [noches_1,noches_2,noches_3]
time_series_list_noche = process_night_data(directories)

interpolated_data_noche_RoVio = interpolate_time_series_data(time_series_list_noche, 44150, 300)

time_sonido_RoVio = interpolated_data_noche_RoVio['common_time_base_sound']
time_maximos_RoVio = interpolated_data_noche_RoVio['common_time_base_maximos']
time_rate_RoVio = interpolated_data_noche_RoVio['common_time_base_periodo']
average_sonido_noche_RoVio = interpolated_data_noche_RoVio['average_data_sound']
std_sonido_noche_RoVio = interpolated_data_noche_RoVio['std_data_sound']
average_maximos_noche_RoVio = interpolated_data_noche_RoVio['average_data_maximos']
std_maximos_noche_RoVio = interpolated_data_noche_RoVio['std_data_maximos']
average_rate_noche_RoVio = interpolated_data_noche_RoVio['average_data_periodo']
std_rate_noche_RoVio = interpolated_data_noche_RoVio['std_data_periodo']


## NaRo
pajaro = carpetas[2]


subdirectory = os.path.join(directory, pajaro)


dias = os.listdir(subdirectory)
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos
noches_2 = subdirectory + '/' + dias[1] + pajaritos
noches_3 = subdirectory + '/' + dias[2] + pajaritos

directories = [noches_1,noches_2,noches_3]
time_series_list_noche = process_night_data(directories)

interpolated_data_noche_NaRo = interpolate_time_series_data(time_series_list_noche, 44150, 300)

time_sonido_NaRo = interpolated_data_noche_NaRo['common_time_base_sound']
time_maximos_NaRo = interpolated_data_noche_NaRo['common_time_base_maximos']
time_rate_NaRo = interpolated_data_noche_NaRo['common_time_base_periodo']
average_sonido_noche_NaRo = interpolated_data_noche_NaRo['average_data_sound']
std_sonido_noche_NaRo = interpolated_data_noche_NaRo['std_data_sound']
average_maximos_noche_NaRo = interpolated_data_noche_NaRo['average_data_maximos']
std_maximos_noche_NaRo = interpolated_data_noche_NaRo['std_data_maximos']
average_rate_noche_NaRo = interpolated_data_noche_NaRo['average_data_periodo']
std_rate_noche_NaRo = interpolated_data_noche_NaRo['std_data_periodo']

#%%
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

ax[0].set_ylabel("Audio (arb. u.)")
ax[1].set_ylabel("Pressure (arb. u.)")
ax[2].set_ylabel("Rate (Hz)")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()


ax[0].errorbar(time_sonido_RoNe, average_sonido_noche_RoNe, std_sonido_noche_RoNe, color='C0')
ax[0].errorbar(time_sonido_RoVio, average_sonido_noche_RoVio, std_sonido_noche_RoVio, color='C1')
ax[0].errorbar(time_sonido_NaRo, average_sonido_noche_NaRo, std_sonido_noche_NaRo, color='C2')

ax[1].errorbar(time_maximos_RoNe, average_maximos_noche_RoNe, std_maximos_noche_RoNe, color='C0')
ax[1].errorbar(time_maximos_RoVio, average_maximos_noche_RoVio, std_maximos_noche_RoVio, color='C1')
ax[1].errorbar(time_maximos_NaRo, average_maximos_noche_NaRo, std_maximos_noche_NaRo, color='C2')

ax[2].errorbar(time_rate_RoNe, average_rate_noche_RoNe/np.mean(average_rate_noche_RoNe[0:100]), std_rate_noche_RoNe, color='C0')
ax[2].errorbar(time_rate_RoVio, average_rate_noche_RoVio/np.mean(average_rate_noche_RoVio[0:100]), std_rate_noche_RoVio, color='C1')
ax[2].errorbar(time_rate_NaRo, average_rate_noche_NaRo/np.mean(average_rate_noche_NaRo[0:100]), std_rate_noche_NaRo, color='C2')

legend_handles = [
    mpatches.Patch(color='C0', label='RoNe'),
    mpatches.Patch(color='C1', label='RoVio'),
    mpatches.Patch(color='C2', label='NaRo'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')


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
#%%
# Calculate Welch metrics for sound, maximos, and periodo

#RoNe-RoVio
welch_metrics_sound_RoNe_RoVio, degrees_of_freedom_sound_RoNe_RoVio = calculate_welch_metric(time_sonido_RoNe, average_sonido_noche_RoNe, std_sonido_noche_RoNe, 
                                             interpolated_data_noche_RoNe['non_nan_sound'],
                                             time_sonido_RoVio, average_sonido_noche_RoVio, std_sonido_noche_RoVio, 
                                             interpolated_data_noche_RoVio['non_nan_sound'])

welch_metrics_maximos_RoNe_RoVio, degrees_of_freedom_maximos_RoNe_RoVio = calculate_welch_metric(time_maximos_RoNe, average_maximos_noche_RoNe, std_maximos_noche_RoNe, 
                                               interpolated_data_noche_RoNe['non_nan_maximos'],
                                               time_maximos_RoVio, average_maximos_noche_RoVio, std_maximos_noche_RoVio, 
                                               interpolated_data_noche_RoVio['non_nan_maximos'])

welch_metrics_periodo_RoNe_RoVio, degrees_of_freedom_periodo_RoNe_RoVio = calculate_welch_metric(time_rate_RoNe, average_rate_noche_RoNe/np.mean(average_rate_noche_RoNe[0:100]), std_rate_noche_RoNe, 
                                               interpolated_data_noche_RoNe['non_nan_periodo'],
                                               time_rate_RoVio, average_rate_noche_RoVio/np.mean(average_rate_noche_RoVio[0:100]), std_rate_noche_RoVio,
                                               interpolated_data_noche_RoVio['non_nan_periodo'])

#RoNe-NaRo
welch_metrics_sound_RoNe_NaRo, degrees_of_freedom_sound_RoNe_NaRo = calculate_welch_metric(
    time_sonido_RoNe, average_sonido_noche_RoNe, std_sonido_noche_RoNe, 
    interpolated_data_noche_RoNe['non_nan_sound'],
    time_sonido_NaRo, average_sonido_noche_NaRo, std_sonido_noche_NaRo, 
    interpolated_data_noche_NaRo['non_nan_sound']
)

welch_metrics_maximos_RoNe_NaRo, degrees_of_freedom_maximos_RoNe_NaRo = calculate_welch_metric(
    time_maximos_RoNe, average_maximos_noche_RoNe, std_maximos_noche_RoNe, 
    interpolated_data_noche_RoNe['non_nan_maximos'],
    time_maximos_NaRo, average_maximos_noche_NaRo, std_maximos_noche_NaRo, 
    interpolated_data_noche_NaRo['non_nan_maximos']
)

welch_metrics_periodo_RoNe_NaRo, degrees_of_freedom_periodo_RoNe_NaRo = calculate_welch_metric(
    time_rate_RoNe, average_rate_noche_RoNe/np.mean(average_rate_noche_RoNe[0:100]), std_rate_noche_RoNe, 
    interpolated_data_noche_RoNe['non_nan_periodo'],
    time_rate_NaRo, average_rate_noche_NaRo/np.mean(average_rate_noche_NaRo[0:100]), std_rate_noche_NaRo,
    interpolated_data_noche_NaRo['non_nan_periodo']
)

#RoVio-NaRo
welch_metrics_sound_RoVio_NaRo, degrees_of_freedom_sound_RoVio_NaRo = calculate_welch_metric(
    time_sonido_RoVio, average_sonido_noche_RoVio, std_sonido_noche_RoVio, 
    interpolated_data_noche_RoVio['non_nan_sound'],
    time_sonido_NaRo, average_sonido_noche_NaRo, std_sonido_noche_NaRo, 
    interpolated_data_noche_NaRo['non_nan_sound']
)

welch_metrics_maximos_RoVio_NaRo, degrees_of_freedom_maximos_RoVio_NaRo = calculate_welch_metric(
    time_maximos_RoVio, average_maximos_noche_RoVio, std_maximos_noche_RoVio, 
    interpolated_data_noche_RoVio['non_nan_maximos'],
    time_maximos_NaRo, average_maximos_noche_NaRo, std_maximos_noche_NaRo, 
    interpolated_data_noche_NaRo['non_nan_maximos']
)

welch_metrics_periodo_RoVio_NaRo, degrees_of_freedom_periodo_RoVio_NaRo = calculate_welch_metric(
    time_rate_RoVio, average_rate_noche_RoVio/np.mean(average_rate_noche_RoVio[0:100]), std_rate_noche_RoVio, 
    interpolated_data_noche_RoVio['non_nan_periodo'],
    time_rate_NaRo, average_rate_noche_NaRo/np.mean(average_rate_noche_NaRo[0:100]), std_rate_noche_NaRo,
    interpolated_data_noche_NaRo['non_nan_periodo']
)

#%%

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)

ax[0].set_ylabel("Welch Audio (arb. u.)")
ax[1].set_ylabel("Welch Pressure (arb. u.)")
ax[2].set_ylabel("Welch Rate (Hz)")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()

ax[0].plot(time_sonido_RoNe[10:-10],welch_metrics_sound_RoNe_RoVio[10:-10],color='C3')
ax[0].plot(time_sonido_RoNe[10:-10],welch_metrics_sound_RoNe_NaRo[10:-10],color='C4')
ax[0].plot(time_sonido_RoVio[10:-100],welch_metrics_sound_RoVio_NaRo[10:-100],color='C5')

ax[1].plot(time_maximos_RoNe,welch_metrics_maximos_RoNe_RoVio,color='C3')
ax[1].plot(time_maximos_RoNe,welch_metrics_maximos_RoNe_NaRo,color='C4')
ax[1].plot(time_maximos_RoVio,welch_metrics_maximos_RoVio_NaRo,color='C5')

ax[2].plot(time_rate_RoNe,welch_metrics_periodo_RoNe_RoVio,color='C3')
ax[2].plot(time_rate_RoNe,welch_metrics_periodo_RoNe_NaRo,color='C4')
ax[2].plot(time_rate_RoVio[:-1],welch_metrics_periodo_RoVio_NaRo[:-1],color='C5')

legend_handles = [
    mpatches.Patch(color='C3', label='RoNe-RoVio'),
    mpatches.Patch(color='C4', label='RoNe-NaRo'),
    mpatches.Patch(color='C5', label='RoVio-NaRo'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')

# Degrees of freedom
fig,ax=plt.subplots(3,1,figsize=(14,7),sharex=True)

ax[0].set_ylabel("D.F Audio (arb. u.)")
ax[1].set_ylabel("D.F Pressure (arb. u.)")
ax[2].set_ylabel("D.F Rate (Hz)")
ax[2].set_xlabel("Time (s)")
plt.tight_layout()


ax[0].plot(time_sonido_RoNe[10:-10],degrees_of_freedom_sound_RoNe_RoVio[10:-10],color='C3')
ax[0].plot(time_sonido_RoNe[10:-10],degrees_of_freedom_sound_RoNe_NaRo[10:-10],color='C4')
ax[0].plot(time_sonido_RoVio[10:-100],degrees_of_freedom_sound_RoVio_NaRo[10:-100],color='C5')

ax[1].plot(time_maximos_RoNe,degrees_of_freedom_maximos_RoNe_RoVio,color='C3')
ax[1].plot(time_maximos_RoNe,degrees_of_freedom_maximos_RoNe_NaRo,color='C4')
ax[1].plot(time_maximos_RoVio,degrees_of_freedom_maximos_RoVio_NaRo,color='C5')

ax[2].plot(time_rate_RoNe,degrees_of_freedom_periodo_RoNe_RoVio,color='C3')
ax[2].plot(time_rate_RoNe,degrees_of_freedom_periodo_RoNe_NaRo,color='C4')
ax[2].plot(time_rate_RoVio[:-1],degrees_of_freedom_periodo_RoVio_NaRo[:-1],color='C5')

legend_handles = [
    mpatches.Patch(color='C3', label='RoNe-RoVio'),
    mpatches.Patch(color='C4', label='RoNe-NaRo'),
    mpatches.Patch(color='C5', label='RoVio-NaRo'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True)
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')

#%% Sliders para ajustar el sonido y la presion

from matplotlib.widgets import Slider
from scipy.stats import norm
from scipy.signal import find_peaks
from numba import njit

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax[0].set_ylabel("Audio (arb. u.)")
ax[1].set_ylabel("Pressure (arb. u.)")
ax[1].set_xlabel("Time (sec)")
plt.tight_layout()

ax[0].errorbar(time_sonido_RoNe, average_sonido_noche_RoNe, std_sonido_noche_RoNe, color='C0')

ax[1].errorbar(time_maximos_RoNe, average_maximos_noche_RoNe, std_maximos_noche_RoNe, color='C0')
plt.subplots_adjust(bottom=0.35)

@njit
def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def f(v, t, pars):
    cr, w0, ci, As, tau, A_s0 = pars
    r, theta, mu = v
    
    # Equations
    drdt = mu * r - cr * r**3
    dthetadt = w0 - ci * r**2
    dmudt = -mu / tau + A_s0 * As
    
    return np.array([drdt, dthetadt, dmudt])

@njit
def integrate_system(time, dt, pars, As_values):
    # Initial conditions
    r = np.zeros_like(time)
    theta = np.zeros_like(time)
    mu = np.zeros_like(time)
    r[0], theta[0], mu[0] = 0.25, 0, 0

    for ix in range(len(time) - 1):
        pars[3] = As_values[ix]  # Modify As within the loop
        r[ix + 1], theta[ix + 1], mu[ix + 1] = rk4(f, np.array([r[ix], theta[ix], mu[ix]]), time[ix], dt, pars)
    
    return r, theta, mu

    
time = time_sonido_RoNe
dt = np.mean(np.diff(time))

s, =ax[0].plot([],[],'r')
p, = ax[1].plot([], [], 'r')
# r_plot, = ax[2].plot([], [],'r')


# Slider positions
slider_params = [
    ("cr", 0.1, 0.25, 0.3, 0.03, 0.0, 10.0, 0.5),
    ("w0", 0.1, 0.20, 0.3, 0.03, 0.0, 10.0, 7.28),
    ("ci", 0.1, 0.15, 0.3, 0.03, -5, 0.0, -1),
    ("tau", 0.1, 0.1, 0.3, 0.03, 0.0, 1, 0.05),
    ("A_s0", 0.65, 0.25, 0.3, 0.03, 0.0, 5.0, 1),
]

sliders = {}
for name, left, bottom, width, height, min_val, max_val, init_val in slider_params:
    sliders[name] = Slider(plt.axes([left, bottom, width, height]), name, min_val, max_val, valinit=init_val)


sliders = {}
for name, left, bottom, width, height, min_val, max_val, init_val in slider_params:
    sliders[name] = Slider(plt.axes([left, bottom, width, height]), name, min_val, max_val, valinit=init_val)

# Function to update plots
def update(val):
    # Convert all slider values to float32 to ensure consistency
    pars = np.array([
        sliders['cr'].val, sliders['w0'].val, sliders['ci'].val, 1.0,
        sliders['tau'].val, sliders['A_s0'].val
    ], dtype=np.float64)
    
    r, theta, mu = integrate_system(time, dt, pars, average_sonido_noche_RoNe)

    peaks, _ = find_peaks(r * np.cos(theta), height=0, distance=int(0.1 / dt))
    
    # Update plots
    p.set_data(time[peaks], (r * np.cos(theta))[peaks])
    s.set_data(time,mu)
    # r_plot.set_data(time, w)
    ax[1].relim()
    ax[1].autoscale_view()
    ax[0].relim()
    ax[0].autoscale_view()
    fig.canvas.draw_idle()

# Connect sliders to the update function
for slider in sliders.values():
    slider.on_changed(update)

# Initial call to update the plot
update(None)
plt.show()

#%% Sliders para ajustar el rate
from scipy.signal import butter, sosfiltfilt

@njit
def sigm(x):
    return 1 / (1 + np.exp(-10 * x))

@njit
def f(v, t, pars):
    w0, As, A_s0, t_up, t_down, Asp, alpha = pars
    w = v

    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) * sigm(Asp - alpha)) * (A_s0 * As + w0 - w)
    
    return np.array([dwdt])


@njit
def integrate_system(time, dt, pars, As_values, Asp_values):
    # Initial conditions
    w = np.zeros_like(time)
    w[0] = pars[0]

    for ix in range(len(time) - 1):
        pars[1] = As_values[ix]  # Modify As within the loop
        pars[-2] = Asp_values[ix]  # Modify Asp within the loop
        w[ix + 1] = rk4(f, np.array([w[ix]]), time[ix], dt, pars)
    
    return w

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

# Apply the band-pass filter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = sosfiltfilt(sos, data)
    return filtered_signal


fig, ax = plt.subplots(figsize=(14, 7), sharex=True)


ax.set_ylabel("Rate (Hz)")
ax.set_xlabel("Time (sec)")
plt.tight_layout()


ax.errorbar(time_rate_RoNe, average_rate_noche_RoNe/np.mean(average_rate_noche_RoNe[0:100]), std_rate_noche_RoNe, color='C0')

lowcut = 0.01  # Low cutoff frequency in Hz
highcut = 20  # High cutoff frequency in Hz
fs = 44150  # Sampling frequency in Hz
order = 6  # Filter order

# Apply the band-pass filter
filtered_signal = butter_bandpass_filter(
    average_sonido_noche_RoNe, lowcut, highcut, fs, order=order)

time = time_sonido_RoNe
dt = np.mean(np.diff(time))
Asp = np.concatenate(([0], np.diff(filtered_signal)))


plt.subplots_adjust(bottom=0.35)


r_plot, = ax.plot([], [],'r')

# Slider positions
slider_params = [
    ("w0", 0.1, 0.20, 0.3, 0.03, 0.0, 10.0, 5),
    ("A_s0", 0.65, 0.25, 0.3, 0.03, 0.0, 5.0, 1),
    ("t_up", 0.65, 0.20, 0.3, 0.03, 0.0, 5.0, 1),
    ("t_down", 0.65, 0.15, 0.3, 0.03, 0.0, 5.0, 1),
    ("alpha", 0.65, 0.1, 0.3, 0.03, 0.0, 5.0, 1),
]

sliders = {}
for name, left, bottom, width, height, min_val, max_val, init_val in slider_params:
    sliders[name] = Slider(plt.axes([left, bottom, width, height]), name, min_val, max_val, valinit=init_val)

# Function to update plots
def update(val):
    # Convert all slider values to float32 to ensure consistency
    pars = np.array([
        sliders['w0'].val, 1.0,
        sliders['A_s0'].val, sliders['t_up'].val, sliders['t_down'].val, 1.0,
        sliders['alpha'].val
    ], dtype=np.float64)
    
    w = integrate_system(time, dt, pars, average_sonido_noche_RoNe, Asp)
    
    # Update plots
    r_plot.set_data(time, w)
    ax.relim()
    ax.autoscale_view()

    fig.canvas.draw_idle()

# Connect sliders to the update function
for slider in sliders.values():
    slider.on_changed(update)

# Initial call to update the plot
update(None)
plt.show()


#%%

fig, ax = plt.subplots(figsize=(14, 7), sharex=True)


ax.set_ylabel("Rate (Hz)")
ax.set_xlabel("Time (s)")
plt.tight_layout()


ax.errorbar(time_rate_RoNe, average_rate_noche_RoNe/np.mean(average_rate_noche_RoNe[0:100]), std_rate_noche_RoNe, color='C0')
r_plot, = ax.plot([], [],'r')

lowcut = 0.01  # Low cutoff frequency in Hz
highcut = 20  # High cutoff frequency in Hz
fs = 44150  # Sampling frequency in Hz
order = 6  # Filter order

# Apply the band-pass filter
filtered_signal = butter_bandpass_filter(
    average_sonido_noche_RoNe, lowcut, highcut, fs, order=order)

time = time_sonido_RoNe
dt = np.mean(np.diff(time))
Asp = np.concatenate(([0], np.diff(filtered_signal)))

A_s0 =0.31 #1.75 / 4
t_up =0.05
t_down = 27
w0 = 1.2

#      w0, As, A_s0, t_up, t_down, Asp
pars = [w0, 1, A_s0, t_up, t_down, 1]

w = integrate_system(time, dt, pars, average_sonido_noche_RoNe, Asp)
r_plot.set_data(time, w)