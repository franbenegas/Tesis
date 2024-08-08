# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:11:19 2024

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

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
# os.chdir(directory)

# Datos = pd.read_csv('adq-log.txt', delimiter='\t') 

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
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\2023-01-15-night\Aviones'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = []
Sonidos = []

for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
        Presiones.append(file)

#%%
columnas = ['Nombre','Tiempo inicial normalizacion','Tiempo final normalizacion','Tiempo inicial avion','Tiempo final avion']
Datos = pd.DataFrame(columns=columnas)

tiempo_inicial = []

#%%
indice = 0
ti,tf = 3,5
print(indice)
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

plt.close('Sonido vs presion')
plt.figure('Sonido vs presion',figsize=(14,7))
plt.suptitle(f'{name}')
plt.subplot(211)
plt.plot(time, audio)
plt.axvline(x=longest_interval[0], color='k', linestyle='-')
plt.axvline(x=longest_interval[-1], color='k', linestyle='-', label='Avion')
# plt.axhline(y=1, color='k', linestyle='-')
# plt.axhline(y=-1, color='k', linestyle='-')
plt.ylabel("Audio (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.subplot(212)
# plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud')
# plt.colorbar(label='Intensity [dB]')
# plt.axhline(y=frequency_cutoff,color='k', linestyle='--', label='cutoff')
plt.plot(time,pressure)
plt.axvline(x=longest_interval[0], color='k', linestyle='-')
plt.axvline(x=longest_interval[-1], color='k', linestyle='-', label='Avion')
# plt.axhline(y=1, color='k', linestyle='-')
# plt.axhline(y=-1, color='k', linestyle='-')
plt.ylabel("Pressure (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.xlabel("Time (sec)")
plt.tight_layout()

#%% Los agrego al dataframe

tiempo_inicial.append(longest_interval[0])

temp_df = pd.DataFrame([[name,ti,tf,longest_interval[0],longest_interval[-1]]], columns=columnas)


Datos = pd.concat([Datos, temp_df], ignore_index=True)

#%% Guardo los datos

with open(f'Datos {name[:-9]}.pkl', 'wb') as file:
    pickle.dump(Datos, file)

#%% Calculo de maximos 

peaks_maximos,properties_maximos = find_peaks(pressure,prominence=1,height=0, distance=int(fs*0.1))
plt.close('Maximos')
plt.figure('Maximos',figsize=(14,7))
plt.plot(time,pressure)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)

for i in range(len(peaks_maximos)):
    plt.text( time[peaks_maximos[i]],pressure[peaks_maximos[i]], str(i) )

plt.ylabel("pressure (arb. u.)")
plt.xlabel("Time (sec)")
plt.legend(fancybox=True, shadow=True)
plt.tight_layout()

#%% Aca saco los que me molestan
sacar = [50,60,72,86,88]
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
np.savetxt(f'{name}_maximos.txt', picos_limpios, delimiter=',',newline='\n',header='Indice Maximos')
#%%
periodo_original = np.diff(time[picos_limpios])

plt.close('Perdiodos comparacion')
plt.figure('Perdiodos comparacion',figsize=(14,7))
plt.subplot(111)
plt.plot(periodo_original,label='Originales')

plt.legend(fancybox=True,shadow=True)

#%%
peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), 
                      prominence=.001)


spline_amplitude_sound = UnivariateSpline(time[peaks_sonido], audio[peaks_sonido], s=0, k=3)
plt.close('interpolado')
plt.figure('interpolado',figsize=(14,7))
plt.plot(time, audio)
plt.plot(time[peaks_sonido], audio[peaks_sonido], '.-')

plt.plot(time, spline_amplitude_sound(time))

interpolado = spline_amplitude_sound(time)

np.savetxt(f'{name}_interpolado.txt', picos_limpios, delimiter=',',newline='\n',header='Interopolado')
#%% Runge-kutta

dt = np.mean(np.diff(time))
respuesta = np.zeros((len(time), 3))
respuesta[0] = [0.1, 2*np.pi, audio[0]]  


c_r, c_i, w, tau = 0.75, -1, 2*np.pi + 1, 1/20

# time = np.linspace(-60,60,len(pressure))

dt = np.mean(np.diff(time))


with tqdm(total=len(time)) as pbar_h:
    for ix, tt in enumerate(time[:-1]):
        respuesta[ix+1] = rk4(modelo, respuesta[ix], tt, dt, [c_r, c_i, w, tau, interpolado[ix]])
        pbar_h.update(1)
        
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)      

#%% Graficos del runge-kutta
r_cos = respuesta[:,0]*np.cos(respuesta[:,1])
picos_respuesta,_  = find_peaks(r_cos,height=0, distance=int(fs*0.1))
periodo = np.diff(time[picos_respuesta])
periodo_datos = np.diff(time[peaks_maximos])

plt.close('Modelo')
plt.figure('Modelo',figsize=(16,8))
plt.suptitle(f'c_r, c_i, w, tau = {round(c_r,2)}, {round(c_i,2)}, {round(w,2)}, {tau}')
plt.subplot(311)
plt.plot(time,r_cos)
plt.plot(time[picos_respuesta],r_cos[picos_respuesta],'.C1',ms=10)
plt.ylabel('rcos(phi)')

plt.subplot(312)
plt.plot(time[picos_respuesta][1::],1/periodo,'C1',label='Modelo')
plt.ylabel('Rate')
plt.legend(loc='upper left')
ax1 = plt.gca().twinx()
ax1.plot(time[peaks_maximos][1::],1/periodo_datos,'C0',label='Presion')
plt.legend()


plt.subplot(313)
plt.plot(time, interpolado, label='Sonido', color='C0')
plt.xlabel('Time [sec]')
plt.ylabel('Sonido')
plt.legend(loc='upper right')
ax1 = plt.gca().twinx()
ax1.plot(time, respuesta[:, 2], 'C1', label='Modelo')
ax1.set_ylabel('Modelo Output')
ax1.legend(loc='upper left')
plt.tight_layout()


#%% Modelo vs real
theta = np.linspace( 0 , 2 * np.pi , 150 )
 
radius = 1
 
a = radius * np.cos( theta )
b = radius * np.sin( theta )

r_sen = respuesta[:,0]*np.sin(respuesta[:,1])
periodo = np.diff(time[picos_respuesta])
plt.close('espacio de fases')
plt.figure('espacio de fases',figsize=(16,8))
plt.plot(r_cos,respuesta[:,0]*np.sin(respuesta[:,1]))
plt.plot(r_cos[0],r_sen[0],'.C1',ms=10)
plt.plot(a,b,'C3')
# plt.plot(np.cos(tita),np.sin(tita),'C3')
plt.tight_layout()

plt.close('modelo vs real')
plt.figure('modelo vs real',figsize=(16,8))
plt.suptitle(f'{name}')
plt.plot(time,pressure,label='Presion')
plt.plot(time,r_cos,label='Modelo')
plt.legend(fancybox=True,shadow=True)


plt.tight_layout()

    
#%%

with open('Datos CaFF028-RoNe_2023_01_31.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos = pickle.load(f)
#%%
plt.close('Superoposicion de todo')
plt.figure('Superoposicion de todo',figsize=(14,7))

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
        

    periodo = np.diff(time[lugar_maximos])

    plt.subplot(311)
    plt.plot(time-tiempo_inicial,interpolado,'.',color='k',ms=0.5,alpha=0.1,solid_capstyle='projecting')
    plt.ylabel("Audio (arb. u.)")
    plt.subplot(312)        
    # plt.plot(time,r_cos,'C1')
    plt.plot((time-tiempo_inicial)[lugar_maximos],pressure[lugar_maximos],'.',linestyle='-',color='k',ms=0.5,alpha=0.1,solid_capstyle='projecting') 
    plt.ylabel("Pressure (arb. u.)")
    plt.subplot(313)
    plt.plot((time-tiempo_inicial)[lugar_maximos][1::],1/periodo,'.',linestyle='-',color='k',ms=0.5,alpha=0.1,solid_capstyle='projecting')
    plt.ylabel('Rate (sec)')
    plt.xlabel("Time (sec)")        
    plt.tight_layout()

#%%
from scipy.interpolate import interp1d
time_series_list = []
# plt.close('Superoposicion de todo')
# plt.figure('Superoposicion de todo',figsize=(14,7))

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
        

    periodo = np.diff(time[lugar_maximos])
    tiempo = time-tiempo_inicial
    time_series_list.append({'time':tiempo,'interpolado':interpolado,
                             'time maximos':tiempo[lugar_maximos],
                             'presion':pressure[lugar_maximos],
                             'time periodo':tiempo[lugar_maximos][1::],'periodo':1/periodo})

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
std_data_sonido = np.nanstd(interpolated_data,axis=0)

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
std_data_maximos = np.nanstd(interpolated_data,axis=0)

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
std_data_periodo = np.nanstd(interpolated_data,axis=0)


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
#%%
with open('Interpolado, maximos y rate de CaFF028-RoNe_2023_01_31.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    time_series_list = pickle.load(f)
    
with open('promedios de CaFF028-RoNe_2023_01_31.pkl', 'rb') as f:
    # Load the DataFrame from the pickle file
    promedios = pickle.load(f)
#%%


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
plt.errorbar(promedios['tiempo sonido'], promedios['promedio sonido'], promedios['errores sonido']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
plt.legend(fancybox=True,shadow=True)
plt.ylabel("Audio (arb. u.)")
plt.subplot(312)
plt.errorbar(promedios['tiempo maximos'], promedios['promedio maximos'], promedios['errores maximos']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
plt.legend(fancybox=True,shadow=True)
plt.ylabel("Pressure (arb. u.)")
plt.subplot(313)
plt.errorbar(promedios['tiempo rate'], promedios['promedio rate'], promedios['errores periodo']/np.sqrt(len(time_series_list)), label='Average', color='r', linewidth=2)
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
dt = np.mean(np.diff(promedios['tiempo sonido']))
respuesta = np.zeros((len(promedios['tiempo sonido']), 3))
respuesta[0] = [0.1, 2*np.pi, promedios['tiempo sonido'][0]]  


c_r, c_i, w, tau = 0.75, -1, 2*np.pi + 1, 1/20

# time = np.linspace(-60,60,len(pressure))

with tqdm(total=len(promedios['tiempo sonido'])) as pbar_h:
    for ix, tt in enumerate(promedios['tiempo sonido'][:-1]):
        respuesta[ix+1] = rk4(modelo, respuesta[ix], tt, dt, [c_r, c_i, w, tau, promedios['tiempo sonido'][ix]])
        pbar_h.update(1)
        
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)      


