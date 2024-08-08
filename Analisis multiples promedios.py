# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:47:48 2024

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

#%%

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe'
os.chdir(directory)

carpetas = os.listdir(directory)
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = directory + '/' + carpetas[0] + pajaritos
noches_2 = directory + '/' + carpetas[1] + pajaritos
noches_3 = directory + '/' + carpetas[2] + pajaritos

noche_1 = os.listdir(noches_1)
noche_2 = os.listdir(noches_2)
noche_3 = os.listdir(noches_3)

#%%

Presiones_noche_1 = []
Sonidos_noche_1 = []
Datos_noche_1 = []
for file in noche_1:
    if file[0]=='s':
        Sonidos_noche_1.append(file)
    elif file[0]=='p':
        Presiones_noche_1.append(file)
    elif file[0]=='D':
        Datos_noche_1.append(file)
        
Presiones_noche_2 = []
Sonidos_noche_2 = []
Datos_noche_2 = []

for file in noche_2:
    if file[0]=='s':
        Sonidos_noche_2.append(file)
    elif file[0]=='p':
        Presiones_noche_2.append(file)
    elif file[0]=='D':
        Datos_noche_2.append(file)
        
Presiones_noche_3 = []
Sonidos_noche_3 = []
Datos_noche_3 = []

for file in noche_3:
    if file[0]=='s':
        Sonidos_noche_3.append(file)
    elif file[0]=='p':
        Presiones_noche_3.append(file)
    elif file[0]=='D':
        Datos_noche_3.append(file)
        
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

time = np.linspace(0, 2649006/44150, 2649006)


    
#%% Hago la serie temporal de todas las noches
os.chdir(noches_1)
with open(Datos_noche_1[0], 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos_noche_1 = pickle.load(f)

time_series_list_noche_1 = []


for indice in range(Datos_noche_1.shape[0]):
    
    ti,tf = Datos_noche_1.loc[indice,'Tiempo inicial normalizacion'],Datos_noche_1.loc[indice,'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_1.loc[indice,'Tiempo inicial avion']
    
    audio, pressure, name, fs = datos_normalizados_2(Sonidos_noche_1,Presiones_noche_1,indice, ti, tf)
    
    
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
    time_series_list_noche_1.append({'time':tiempo,'interpolado':interpolado,
                             'time maximos':tiempo[lugar_maximos],
                             'presion':pressure[lugar_maximos],
                             'time periodo':tiempo[lugar_maximos][1::],'periodo':1/periodo})

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)    

os.chdir(noches_2)
with open(Datos_noche_2[0], 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos_noche_2 = pickle.load(f)

time_series_list_noche_2 = []


for indice in range(Datos_noche_2.shape[0]):
    
    ti,tf = Datos_noche_2.loc[indice,'Tiempo inicial normalizacion'],Datos_noche_2.loc[indice,'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_2.loc[indice,'Tiempo inicial avion']
    
    audio, pressure, name, fs = datos_normalizados_2(Sonidos_noche_2,Presiones_noche_2,indice, ti, tf)
    
    
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
    time_series_list_noche_2.append({'time':tiempo,'interpolado':interpolado,
                             'time maximos':tiempo[lugar_maximos],
                             'presion':pressure[lugar_maximos],
                             'time periodo':tiempo[lugar_maximos][1::],'periodo':1/periodo})

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)    

os.chdir(noches_3)
with open(Datos_noche_3[0], 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos_noche_3 = pickle.load(f)

time_series_list_noche_3 = []


for indice in range(Datos_noche_3.shape[0]):
    
    ti,tf = Datos_noche_3.loc[indice,'Tiempo inicial normalizacion'],Datos_noche_3.loc[indice,'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_3.loc[indice,'Tiempo inicial avion']
    
    audio, pressure, name, fs = datos_normalizados_2(Sonidos_noche_3,Presiones_noche_3,indice, ti, tf)
    
    
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
    time_series_list_noche_3.append({'time':tiempo,'interpolado':interpolado,
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

###################        Noche 1
#Sonido
start_time = min(ts["time"][0] for ts in time_series_list_noche_1)
end_time = max(ts["time"][-1] for ts in time_series_list_noche_1)
common_time_base_sonido_noche_1 = np.linspace(start_time, end_time, fs)

interpolated_data_noche_1 = []
for ts in time_series_list_noche_1:
    interp_func = interp1d(ts["time"], ts["interpolado"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(interp_func(common_time_base_sonido_noche_1))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_sound_noche_1 = np.array(interpolated_data_noche_1)
# Compute the average, ignoring NaNs
average_data_sonido_noche_1 = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_sonido_noche_1 = np.nanstd(interpolated_data_noche_1,axis=0)/np.sqrt(Datos_noche_1.shape[0])

#Maximos
start_time = min(ts["time maximos"][0] for ts in time_series_list_noche_1)
end_time = max(ts["time maximos"][-1] for ts in time_series_list_noche_1)
common_time_base_maximos_noche_1 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_1 = []
for ts in time_series_list_noche_1:
    interp_func = interp1d(ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(interp_func(common_time_base_maximos_noche_1))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_maximos_noche_1 = np.array(interpolated_data_noche_1)
# Compute the average, ignoring NaNs
average_data_maximos_noche_1 = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_maximos_noche_1 = np.nanstd(interpolated_data_noche_1,axis=0)/np.sqrt(Datos_noche_1.shape[0])

#Periodo
start_time = min(ts["time periodo"][0] for ts in time_series_list_noche_1)
end_time = max(ts["time periodo"][-1] for ts in time_series_list_noche_1)
common_time_base_periodo_noche_1 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_1 = []
for ts in time_series_list_noche_1:
    interp_func = interp1d(ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(interp_func(common_time_base_periodo_noche_1))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_periodo_noche_1 = np.array(interpolated_data_noche_1)

# Compute the average, ignoring NaNs
average_data_periodo_noche_1 = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_periodo_noche_1 = np.nanstd(interpolated_data_noche_1,axis=0)/np.sqrt(Datos_noche_1.shape[0])

###################        Noche 2
#Sonido
start_time = min(ts["time"][0] for ts in time_series_list_noche_2)
end_time = max(ts["time"][-1] for ts in time_series_list_noche_2)
common_time_base_sonido_noche_2 = np.linspace(start_time, end_time, fs)

interpolated_data_noche_2 = []
for ts in time_series_list_noche_2:
    interp_func = interp1d(ts["time"], ts["interpolado"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_2.append(interp_func(common_time_base_sonido_noche_2))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_2 = np.array(interpolated_data_noche_2)
interpolated_sound_noche_2 = np.array(interpolated_data_noche_2)

# Compute the average, ignoring NaNs
average_data_sonido_noche_2 = np.nanmean(interpolated_data_noche_2, axis=0)
std_data_sonido_noche_2 = np.nanstd(interpolated_data_noche_2, axis=0) / np.sqrt(Datos_noche_2.shape[0])

#Maximos
start_time = min(ts["time maximos"][0] for ts in time_series_list_noche_2)
end_time = max(ts["time maximos"][-1] for ts in time_series_list_noche_2)
common_time_base_maximos_noche_2 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_2 = []
for ts in time_series_list_noche_2:
    interp_func = interp1d(ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_2.append(interp_func(common_time_base_maximos_noche_2))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_2 = np.array(interpolated_data_noche_2)
interpolated_maximos_noche_2 = np.array(interpolated_data_noche_2)

# Compute the average, ignoring NaNs
average_data_maximos_noche_2 = np.nanmean(interpolated_data_noche_2, axis=0)
std_data_maximos_noche_2 = np.nanstd(interpolated_data_noche_2, axis=0) / np.sqrt(Datos_noche_2.shape[0])

#Periodo
start_time = min(ts["time periodo"][0] for ts in time_series_list_noche_2)
end_time = max(ts["time periodo"][-1] for ts in time_series_list_noche_2)
common_time_base_periodo_noche_2 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_2 = []
for ts in time_series_list_noche_2:
    interp_func = interp1d(ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_2.append(interp_func(common_time_base_periodo_noche_2))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_2 = np.array(interpolated_data_noche_2)
interpolated_periodo_noche_2 = np.array(interpolated_data_noche_2)

# Compute the average, ignoring NaNs
average_data_periodo_noche_2 = np.nanmean(interpolated_data_noche_2, axis=0)
std_data_periodo_noche_2 = np.nanstd(interpolated_data_noche_2, axis=0) / np.sqrt(Datos_noche_2.shape[0])



###################        Noche 3
#Sonido
start_time = min(ts["time"][0] for ts in time_series_list_noche_3)
end_time = max(ts["time"][-1] for ts in time_series_list_noche_3)
common_time_base_sonido_noche_3 = np.linspace(start_time, end_time, fs)

interpolated_data_noche_3 = []
for ts in time_series_list_noche_3:
    interp_func = interp1d(ts["time"], ts["interpolado"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_3.append(interp_func(common_time_base_sonido_noche_3))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_3 = np.array(interpolated_data_noche_3)
interpolated_sound_noche_3 = np.array(interpolated_data_noche_3)

# Compute the average, ignoring NaNs
average_data_sonido_noche_3 = np.nanmean(interpolated_data_noche_3, axis=0)
std_data_sonido_noche_3 = np.nanstd(interpolated_data_noche_3, axis=0) / np.sqrt(Datos_noche_3.shape[0])

#Maximos
start_time = min(ts["time maximos"][0] for ts in time_series_list_noche_3)
end_time = max(ts["time maximos"][-1] for ts in time_series_list_noche_3)
common_time_base_maximos_noche_3 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_3 = []
for ts in time_series_list_noche_3:
    interp_func = interp1d(ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_3.append(interp_func(common_time_base_maximos_noche_3))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_3 = np.array(interpolated_data_noche_3)
interpolated_maximos_noche_3 = np.array(interpolated_data_noche_3)

# Compute the average, ignoring NaNs
average_data_maximos_noche_3 = np.nanmean(interpolated_data_noche_3, axis=0)
std_data_maximos_noche_3 = np.nanstd(interpolated_data_noche_3, axis=0) / np.sqrt(Datos_noche_3.shape[0])

#Periodo
start_time = min(ts["time periodo"][0] for ts in time_series_list_noche_3)
end_time = max(ts["time periodo"][-1] for ts in time_series_list_noche_3)
common_time_base_periodo_noche_3 = np.linspace(start_time, end_time, 300)

interpolated_data_noche_3 = []
for ts in time_series_list_noche_3:
    interp_func = interp1d(ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_3.append(interp_func(common_time_base_periodo_noche_3))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_3 = np.array(interpolated_data_noche_3)
interpolated_periodo_noche_3 = np.array(interpolated_data_noche_3)

# Compute the average, ignoring NaNs
average_data_periodo_noche_3 = np.nanmean(interpolated_data_noche_3, axis=0)
std_data_periodo_noche_3 = np.nanstd(interpolated_data_noche_3, axis=0) / np.sqrt(Datos_noche_3.shape[0])


#%%
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('RoVio')
ax[0].errorbar(common_time_base_sonido_noche_1, average_data_sonido_noche_1, std_data_sonido_noche_1, label='Noche 1', color='r', linewidth=2)
ax[0].errorbar(common_time_base_sonido_noche_2, average_data_sonido_noche_2, std_data_sonido_noche_2, label='Noche 2', color='g', linewidth=2)
ax[0].errorbar(common_time_base_sonido_noche_3, average_data_sonido_noche_3, std_data_sonido_noche_3, label='Noche 3', color='b', linewidth=2)
ax[0].legend(fancybox=True, shadow=True)
ax[0].set_ylabel("Audio (arb. u.)")

ax[1].errorbar(common_time_base_maximos_noche_1, average_data_maximos_noche_1, std_data_maximos_noche_1, label='Noche 1', color='r', linewidth=2)
ax[1].errorbar(common_time_base_maximos_noche_2, average_data_maximos_noche_2, std_data_maximos_noche_2, label='Noche 2', color='g', linewidth=2)
ax[1].errorbar(common_time_base_maximos_noche_3, average_data_maximos_noche_3, std_data_maximos_noche_3, label='Noche 3', color='b', linewidth=2)
ax[1].legend(fancybox=True, shadow=True)
ax[1].set_ylabel("Pressure (arb. u.)")

ax[2].errorbar(common_time_base_periodo_noche_1, average_data_periodo_noche_1, std_data_periodo_noche_1, label='Noche 1', color='r', linewidth=2)
ax[2].errorbar(common_time_base_periodo_noche_2, average_data_periodo_noche_2, std_data_periodo_noche_2, label='Noche 2', color='g', linewidth=2)
ax[2].errorbar(common_time_base_periodo_noche_3, average_data_periodo_noche_3, std_data_periodo_noche_3, label='Noche 3', color='b', linewidth=2)
ax[2].legend(fancybox=True, shadow=True)
ax[2].set_ylabel('Rate (sec)')
ax[2].set_xlabel("Time (sec)")

plt.tight_layout()

#%%
# interpolated_data_noche_1 = pd.DataFrame(interpolated_data_noche_1)
# counts = interpolated_data_noche_1[0].value_counts(dropna=True)
# counts.plot.bar(edgecolor='k')

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('Distribucion de rates')
ax[0].set_title('Noche 1')
ax[1].set_title('Noche 2')
ax[2].set_title('Noche 3')
ax[2].set_xlabel("Rate (1/sec)")
for i in range(len(interpolated_periodo_noche_1)):
    ax[0].hist(interpolated_periodo_noche_1[i],color='k', alpha=0.5)
for i in range(len(interpolated_periodo_noche_2)):
    ax[1].hist(interpolated_periodo_noche_2[i],color='k', alpha=0.5)
for i in range(len(interpolated_periodo_noche_3)):
    ax[2].hist(interpolated_periodo_noche_3[i],color='k', alpha=0.5)

plt.tight_layout()

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('Distribucion de sonidos')
ax[0].set_title('Noche 1')
ax[1].set_title('Noche 2')
ax[2].set_title('Noche 3')
ax[2].set_xlabel("Sound (arb. u.)")
for i in range(len(interpolated_sound_noche_1)):
    ax[0].hist(interpolated_sound_noche_1[i],color='k', alpha=0.5)
for i in range(len(interpolated_sound_noche_2)):
    ax[1].hist(interpolated_sound_noche_2[i],color='k', alpha=0.5)
for i in range(len(interpolated_sound_noche_3)):
    ax[2].hist(interpolated_sound_noche_3[i],color='k', alpha=0.5)

plt.tight_layout()

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('Distribucion de presion')
ax[0].set_title('Noche 1')
ax[1].set_title('Noche 2')
ax[2].set_title('Noche 3')
ax[2].set_xlabel("Pressure (arb. u.)")
for i in range(len(interpolated_maximos_noche_1)):
    ax[0].hist(interpolated_maximos_noche_1[i],color='k', alpha=0.5)
for i in range(len(interpolated_maximos_noche_2)):
    ax[1].hist(interpolated_maximos_noche_2[i],color='k', alpha=0.5)
for i in range(len(interpolated_maximos_noche_3)):
    ax[2].hist(interpolated_maximos_noche_3[i],color='k', alpha=0.5)

plt.tight_layout()

#%%

interpolated_periodo_noche_1 = pd.DataFrame(interpolated_periodo_noche_1)
interpolated_periodo_noche_2 = pd.DataFrame(interpolated_periodo_noche_2)
interpolated_periodo_noche_3 = pd.DataFrame(interpolated_periodo_noche_3)

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle('Distribucion de Rate en dos tiempos')
ax[0].set_title('Noche 1')
ax[1].set_title('Noche 2')
ax[2].set_title('Noche 3')
ax[0].hist(interpolated_periodo_noche_1[50],edgecolor='k',label=round(common_time_base_periodo_noche_1[50],2))
ax[0].hist(interpolated_periodo_noche_1[140],alpha=0.5,edgecolor='k',label=round(common_time_base_periodo_noche_1[140],2))
ax[0].legend(fancybox=True, shadow=True)

ax[1].hist(interpolated_periodo_noche_2[50],edgecolor='k',label=round(common_time_base_periodo_noche_2[50],2))
ax[1].hist(interpolated_periodo_noche_2[140],alpha=0.5,edgecolor='k',label=round(common_time_base_periodo_noche_2[140],2))
ax[1].legend(fancybox=True, shadow=True)

ax[2].hist(interpolated_periodo_noche_3[50],edgecolor='k',label=round(common_time_base_periodo_noche_3[50],2))
ax[2].hist(interpolated_periodo_noche_3[140],alpha=0.5,edgecolor='k',label=round(common_time_base_periodo_noche_3[140],2))
ax[2].legend(fancybox=True, shadow=True)
#%% Hago pruebas con el modelo
def rk4(f, y0, t0, dt, params):

    k1 = np.array(f(*y0, t0, params)) * dt
    k2 = np.array(f(*(y0 + k1/2), t0 + dt/2, params)) * dt
    k3 = np.array(f(*(y0 + k2/2), t0 + dt/2, params)) * dt
    k4 = np.array(f(*(y0 + k3), t0 + dt, params)) * dt
    return y0 + (k1 + 2*k2 + 2*k3 + k4) / 6



def modelo(rho,phi,u_r, t, pars):
    c_r, c_i, w, tau, value = pars[0], pars[1], pars[2], pars[3], pars[4]
    drdt = u_r*rho - c_r*rho**3
    dphidt = w - c_i*rho**2
    du_rdt = (1/tau) *( -u_r + value)
    return drdt,dphidt,du_rdt
#%%
dt = np.mean(np.diff(common_time_base_sonido_noche_1))
respuesta = np.zeros((len(common_time_base_sonido_noche_1), 3))
respuesta[0] = [0.1, 2*np.pi, average_data_sonido_noche_1[0]]  


c_r, c_i, w, tau = 0.75, -1, 2*np.pi + 1, 1/20

# time = np.linspace(-60,60,len(pressure))

# dt = np.mean(np.diff(time))


with tqdm(total=len(common_time_base_sonido_noche_1)) as pbar_h:
    for ix, tt in enumerate(common_time_base_sonido_noche_1[:-1]):
        respuesta[ix+1] = rk4(modelo, respuesta[ix], tt, dt, [c_r, c_i, w, tau, average_data_sonido_noche_1[ix]])
        pbar_h.update(1)
        
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)      


#%%
r_cos = respuesta[:,0]*np.cos(respuesta[:,1])
picos_respuesta,_  = find_peaks(r_cos,height=0)#, distance=int(fs*0.1))
periodo = np.diff(common_time_base_sonido_noche_1[picos_respuesta])

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
plt.suptitle(noches_1[58:81])
ax[0].errorbar(common_time_base_sonido_noche_1, average_data_sonido_noche_1, std_data_sonido_noche_1, label='Noche 1', color='r', linewidth=2,alpha=0.5)
ax[0].plot(common_time_base_sonido_noche_1,respuesta[:, 2],color='k',label='Modelo')
ax[0].set_ylabel("Audio (arb. u.)")
ax[0].legend(fancybox=True, shadow=True)

ax[1].errorbar(common_time_base_maximos_noche_1, average_data_maximos_noche_1, std_data_maximos_noche_1, label='Noche 1', color='r', linewidth=2,alpha=0.5)
ax[1].plot(common_time_base_sonido_noche_1[picos_respuesta],r_cos[picos_respuesta],color='k',label='Modelo')
ax[1].set_ylabel("Pressure (arb. u.)")
ax[1].legend(fancybox=True, shadow=True)

ax[2].errorbar(common_time_base_periodo_noche_1, average_data_periodo_noche_1, std_data_periodo_noche_1, label='Noche 1', color='r', linewidth=2,alpha=0.5)
ax[2].plot(common_time_base_sonido_noche_1[picos_respuesta][1::],1/periodo,color='k',label='Modelo')
ax[2].set_xlabel("Time (sec)")
ax[2].set_ylabel('Rate (1/sec)')
ax[2].legend(fancybox=True, shadow=True)
plt.tight_layout()






