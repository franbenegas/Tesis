# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:29:41 2024

@author: beneg
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks

get_ipython().run_line_magic('matplotlib', 'qt5')

# Specify the directory containing the files
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
os.chdir(directory)

Datos = pd.read_csv('adq-log.txt', delimiter='\t') 

#%%

def datos_normalizados(indice):
    
    #La funcion que normaliza. 
    def normalizar(x, mean, max):
      #x: el array con la señal que levantas en .wav
      #mean y max: el valor medio y el maximo de la señal. Esta info esta en el txt asociado a esa medicion 
      return (x / np.max(np.abs(x))) * max + mean
    
    Pressure = Datos.loc[indice,'pressure_fname']
    Sound = Datos.loc[indice,'sound_fname']

    fs,audio = wavfile.read(Sound)
    ps,pressure = wavfile.read(Pressure)
    
    #normalizamos aca
    #levantamos el archivo txt que tiene info de la normalizacion
    data_norm = np.loadtxt(Pressure[9:-4] + '.txt',delimiter=',',skiprows=1)
    name = Pressure[9:-4]
    #aca asignamos los valores medios y los maximos de sonido y presion (revisar si los indices estan bien puestos)
    mean_s, max_s = data_norm[0], data_norm[2]
    mean_p, max_p = data_norm[1], data_norm[3]

    #la info de los maximos y las medias se usa para normalizar:
    pressure_norm = normalizar(pressure, mean_p, max_p)
    audio_norm = normalizar(audio, mean_s, max_s)
    
    return audio_norm, pressure_norm, name, fs


#%%

indice = 268

_, pressure, name, ps = datos_normalizados(indice)

#defino el tiempo
time = np.linspace(0, len(pressure)/ps, len(pressure))

# time = time[int(ps*10):int(ps*20)]
# pressure = pressure[int(ps*10):int(ps*20)]


plt.figure(figsize=(14,7))
plt.plot(time,pressure)

#%%

# La forma de gabo
#integrador
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

def f_suav(x, t, pars):
    l, value = pars[0], pars[1]
    dxdt = - l * x + value
    return dxdt

#%%
p_suav = np.zeros_like(time)
dt = np.mean(np.diff(time))

#esto regula cuanto suaviza
l = 50

for ix, tt in enumerate(time[:-1]):
    p_suav[ix+1] = rk4(f_suav, [p_suav[ix]], tt, dt, [l, pressure[ix]])  #<- aca vamos poniendo que copie a la presion


plt.figure(figsize=(14,7))
plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
plt.subplot(211)
plt.plot(time, pressure)
plt.ylabel("pressure (arb. u.)")
plt.subplot(212)
plt.plot(time, p_suav)
plt.ylabel("Smoothed pressure (arb. u.)")
plt.xlabel("Time (s)")

#%%

peaks_maximos,properties_maximos = find_peaks(pressure,prominence=0.1,height=0)
peaks_suav, properties_suav = find_peaks(p_suav,prominence=0.002,height=0)
#%%

plt.close(fig="Maximos originales")
plt.figure("Maximos originales",figsize=(14,7))
plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
plt.subplot(211)
plt.plot(time,pressure)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
# plt.xlim(41.6,41.8)
plt.ylabel("pressure (arb. u.)")
plt.grid(linestyle='dashed')
plt.legend(fancybox=True, shadow=True)



plt.subplot(212)
plt.plot(time, p_suav)
plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
# plt.xlim(41.6,41.8)
plt.grid(linestyle='dashed')
plt.ylabel("Smoothed pressure (arb. u.)")
plt.xlabel('Time [sec]')
plt.legend(fancybox=True, shadow=True)