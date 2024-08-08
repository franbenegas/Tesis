# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:37:04 2024

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


directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-28-night\Aviones\Aviones y pajaros'
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

audio, pressure, name, fs = datos_normalizados_2(0, 0, 5)
time = np.linspace(0, len(pressure)/fs, len(pressure))
#%%


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

#%%

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