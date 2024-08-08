# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:47:29 2024

@author: beneg
"""

from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
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

# %%

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
# %%

Presiones_noche_1 = []
Sonidos_noche_1 = []
Datos_noche_1 = []
for file in noche_1:
    if file[0] == 's':
        Sonidos_noche_1.append(file)
    elif file[0] == 'p':
        Presiones_noche_1.append(file)
    elif file[0] == 'D':
        Datos_noche_1.append(file)

Presiones_noche_2 = []
Sonidos_noche_2 = []
Datos_noche_2 = []

for file in noche_2:
    if file[0] == 's':
        Sonidos_noche_2.append(file)
    elif file[0] == 'p':
        Presiones_noche_2.append(file)
    elif file[0] == 'D':
        Datos_noche_2.append(file)

Presiones_noche_3 = []
Sonidos_noche_3 = []
Datos_noche_3 = []

for file in noche_3:
    if file[0] == 's':
        Sonidos_noche_3.append(file)
    elif file[0] == 'p':
        Presiones_noche_3.append(file)
    elif file[0] == 'D':
        Datos_noche_3.append(file)

# %%


def datos_normalizados_2(Sonidos, Presiones, indice, ti, tf):

    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]

    fs, audio = wavfile.read(Sound)
    fs, pressure = wavfile.read(Pressure)

    pressure = pressure-np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)

    # funcion que normaliza al [-1, 1]
    def norm11_interval(x, ti, tf, fs):
        x_int = x[int(ti*fs):int(tf*fs)]
        return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1

    audio = audio-np.mean(audio)
    audio_norm = audio / np.max(audio)

    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs


time = np.linspace(0, 2649006/44150, 2649006)


# %%


time_series_list = []


os.chdir(noches_1)
with open(Datos_noche_1[0], 'rb') as f:
    # Load the DataFrame from the pickle file
    Datos_noche_1 = pickle.load(f)

# time_series_list_noche_1 = []


for indice in range(Datos_noche_1.shape[0]):

    ti, tf = Datos_noche_1.loc[indice,
                               'Tiempo inicial normalizacion'], Datos_noche_1.loc[indice, 'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_1.loc[indice, 'Tiempo inicial avion']

    audio, pressure, name, fs = datos_normalizados_2(
        Sonidos_noche_1, Presiones_noche_1, indice, ti, tf)

    peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1),
                                 prominence=.001)
    spline_amplitude_sound = UnivariateSpline(
        time[peaks_sonido], audio[peaks_sonido], s=0, k=3)

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
    time_series_list.append({'time': tiempo, 'interpolado': interpolado,
                             'time maximos': tiempo[lugar_maximos],
                             'presion': pressure[lugar_maximos],
                             'time periodo': tiempo[lugar_maximos][1::], 'periodo': 1/periodo})

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

# time_series_list_noche_2 = []


for indice in range(Datos_noche_2.shape[0]):

    ti, tf = Datos_noche_2.loc[indice,
                               'Tiempo inicial normalizacion'], Datos_noche_2.loc[indice, 'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_2.loc[indice, 'Tiempo inicial avion']

    audio, pressure, name, fs = datos_normalizados_2(
        Sonidos_noche_2, Presiones_noche_2, indice, ti, tf)

    peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1),
                                 prominence=.001)
    spline_amplitude_sound = UnivariateSpline(
        time[peaks_sonido], audio[peaks_sonido], s=0, k=3)

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
    time_series_list.append({'time': tiempo, 'interpolado': interpolado,
                             'time maximos': tiempo[lugar_maximos],
                             'presion': pressure[lugar_maximos],
                             'time periodo': tiempo[lugar_maximos][1::], 'periodo': 1/periodo})

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

# time_series_list_noche_3 = []


for indice in range(Datos_noche_3.shape[0]):

    ti, tf = Datos_noche_3.loc[indice,
                               'Tiempo inicial normalizacion'], Datos_noche_3.loc[indice, 'Tiempo final normalizacion']
    tiempo_inicial = Datos_noche_3.loc[indice, 'Tiempo inicial avion']

    audio, pressure, name, fs = datos_normalizados_2(
        Sonidos_noche_3, Presiones_noche_3, indice, ti, tf)

    peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1),
                                 prominence=.001)
    spline_amplitude_sound = UnivariateSpline(
        time[peaks_sonido], audio[peaks_sonido], s=0, k=3)

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
    time_series_list.append({'time': tiempo, 'interpolado': interpolado,
                             'time maximos': tiempo[lugar_maximos],
                             'presion': pressure[lugar_maximos],
                             'time periodo': tiempo[lugar_maximos][1::], 'periodo': 1/periodo})

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)

# %%

# Noche 1
# Sonido
start_time = min(ts["time"][0] for ts in time_series_list)
end_time = max(ts["time"][-1] for ts in time_series_list)
common_time_base_sonido_todos = np.linspace(start_time, end_time, fs)

interpolated_data_noche_1 = []
for ts in time_series_list:
    interp_func = interp1d(ts["time"], ts["interpolado"],
                           bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(
        interp_func(common_time_base_sonido_todos))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_sound_noche_1 = np.array(interpolated_data_noche_1)
# Compute the average, ignoring NaNs
average_data_sonido_todos = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_sonido_todos = np.nanstd(
    interpolated_data_noche_1, axis=0)/np.sqrt(Datos_noche_1.shape[0])

# Maximos
start_time = min(ts["time maximos"][0] for ts in time_series_list)
end_time = max(ts["time maximos"][-1] for ts in time_series_list)
common_time_base_maximos_todos = np.linspace(start_time, end_time, 300)

interpolated_data_noche_1 = []
for ts in time_series_list:
    interp_func = interp1d(
        ts["time maximos"], ts["presion"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(
        interp_func(common_time_base_maximos_todos))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_maximos_noche_1 = np.array(interpolated_data_noche_1)
# Compute the average, ignoring NaNs
average_data_maximos_todos = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_maximos_todos = np.nanstd(
    interpolated_data_noche_1, axis=0)/np.sqrt(Datos_noche_1.shape[0])

# Periodo
start_time = min(ts["time periodo"][0] for ts in time_series_list)
end_time = max(ts["time periodo"][-1] for ts in time_series_list)
common_time_base_periodo_todos = np.linspace(start_time, end_time, 300)

interpolated_data_noche_1 = []
for ts in time_series_list:
    interp_func = interp1d(
        ts["time periodo"], ts["periodo"], bounds_error=False, fill_value=np.nan)
    interpolated_data_noche_1.append(
        interp_func(common_time_base_periodo_todos))

# Convert the list of arrays to a 2D numpy array
interpolated_data_noche_1 = np.array(interpolated_data_noche_1)
interpolated_periodo_noche_1 = np.array(interpolated_data_noche_1)

# Compute the average, ignoring NaNs
average_data_periodo_todos = np.nanmean(interpolated_data_noche_1, axis=0)
std_data_periodo_todos = np.nanstd(
    interpolated_data_noche_1, axis=0)/np.sqrt(Datos_noche_1.shape[0])

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)

# %% Hago pruebas con el modelo


def rk4(f, y0, t0, dt, params):

    k1 = np.array(f(*y0, t0, params)) * dt
    k2 = np.array(f(*(y0 + k1/2), t0 + dt/2, params)) * dt
    k3 = np.array(f(*(y0 + k2/2), t0 + dt/2, params)) * dt
    k4 = np.array(f(*(y0 + k3), t0 + dt, params)) * dt
    return y0 + (k1 + 2*k2 + 2*k3 + k4) / 6


def modelo(rho, phi, u_r, t, pars):
    c_r, c_i, w, tau, value = pars[0], pars[1], pars[2], pars[3], pars[4]
    drdt = u_r*rho - c_r*rho**3
    dphidt = w - c_i*rho**2
    du_rdt = (1/tau) * (-u_r + value)
    return drdt, dphidt, du_rdt


# %%
dt = np.mean(np.diff(common_time_base_sonido_todos))
respuesta = np.zeros((len(common_time_base_sonido_todos), 3))
respuesta[0] = [0.1, 2*np.pi, average_data_sonido_todos[0]]


c_r, c_i, w, tau = 0.75, -1, 2*np.pi + 1, 1/20

with tqdm(total=len(common_time_base_sonido_todos)) as pbar_h:
    for ix, tt in enumerate(common_time_base_sonido_todos[:-1]):
        respuesta[ix+1] = rk4(modelo, respuesta[ix], tt, dt,
                              [c_r, c_i, w, tau, average_data_sonido_todos[ix]])
        pbar_h.update(1)

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)


# %%
r_cos = respuesta[:, 0]*np.cos(respuesta[:, 1])
picos_respuesta, _ = find_peaks(r_cos, height=0)  # , distance=int(fs*0.1))
periodo = np.diff(common_time_base_sonido_todos[picos_respuesta])
plt.close('all')
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# plt.suptitle(noches_1[58:81])
plt.suptitle(
    f'$c_r$, $c_i$, $\omega$, $\\tau$ = {round(c_r,2)}, {round(c_i,2)}, {round(w,2)}, {tau}')
ax[0].errorbar(common_time_base_sonido_todos, average_data_sonido_todos,
               std_data_sonido_todos, label='RoNe', color='r', linewidth=2, alpha=0.5)
ax[0].plot(common_time_base_sonido_todos,
           respuesta[:, 2], color='k', label='Modelo')
ax[0].set_ylabel("Audio (arb. u.)")
ax[0].legend(fancybox=True, shadow=True)

ax[1].errorbar(common_time_base_maximos_todos, average_data_maximos_todos,
               std_data_maximos_todos, label='RoNe', color='r', linewidth=2, alpha=0.5)
ax[1].plot(common_time_base_sonido_todos[picos_respuesta],
           r_cos[picos_respuesta], color='k', label='Modelo')
ax[1].set_ylabel("Pressure (arb. u.)")
ax[1].legend(fancybox=True, shadow=True)

ax[2].errorbar(common_time_base_periodo_todos, average_data_periodo_todos,
               std_data_periodo_todos, label='RoNe', color='r', linewidth=2, alpha=0.5)
ax[2].plot(common_time_base_sonido_todos[picos_respuesta]
           [1::], 1/periodo, color='k', label='Modelo')
ax[2].set_xlabel("Time (sec)")
ax[2].set_ylabel('Rate (1/sec)')
ax[2].legend(fancybox=True, shadow=True)
plt.tight_layout()
# %%

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


lowcut = 0.01  # Low cutoff frequency in Hz
highcut = 20  # High cutoff frequency in Hz
fs = fs  # Sampling frequency in Hz
order = 6  # Filter order

# Apply the band-pass filter
filtered_signal = butter_bandpass_filter(
    average_data_sonido_todos, lowcut, highcut, fs, order=order)

plt.figure()
plt.plot(common_time_base_sonido_todos,average_data_sonido_todos)
plt.plot(common_time_base_sonido_todos,filtered_signal)
# plt.plot(common_time_base_sonido_todos[1::], np.diff(filtered_signal))
# plt.plot(common_time_base_sonido_todos[1::],(np.sign( np.diff(filtered_signal))+1)/2)

# %% Como lo hizo facu

# RK4


def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6


def sigm(x):
    return 1 / (1 + np.exp(-10*x))


# Vector field
def f(v, t, pars):

    cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp = [par for par in pars]
    r, theta, mu, w = v[0], v[1], v[2], v[3]

    # Ecuaciones
    drdt = mu * r - cr * r**3

    dthetadt = w - ci * r**2

    dmudt = - mu / tau + A_s0 * As

    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) *
            sigm(Asp-.5)) * (A_s0 * As + w0/2*np.pi - w)

    return [drdt, dthetadt, dmudt, dwdt]


As = average_data_sonido_todos
Asp = np.concatenate(([0], np.diff(filtered_signal)))

# parameters
cr = 0.75
w0 = 2*np.pi+1
ci = - 1
tau = 1/20

A_s0 = 1

t_up = 0.5
t_down = 25

#      cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp
pars = [cr, w0, ci, 1, tau, A_s0, t_up, t_down, 1]

dt = np.mean(np.diff(common_time_base_sonido_todos))
r = np.zeros_like(common_time_base_sonido_todos)
theta = np.zeros_like(common_time_base_sonido_todos)
mu = np.zeros_like(common_time_base_sonido_todos)
w = np.zeros_like(common_time_base_sonido_todos)

# initial condition
r_0, theta_0, mu_0 = 0.1, 2*np.pi, average_data_sonido_todos[0]
r[0] = r_0
theta[0] = theta_0
mu[0] = mu_0
w[0] = 12

for ix, tt in enumerate(common_time_base_sonido_todos[:-1]):
    pars[3] = As[ix]
    pars[-1] = Asp[ix]
    r[ix+1], theta[ix+1], mu[ix+1], w[ix+1] = rk4(f, [r[ix], theta[ix], mu[ix],
                                                      w[ix]], tt, dt, pars)

# %%

plt.figure()
# plt.plot(w)

plt.plot(common_time_base_sonido_todos, average_data_sonido_todos-.8)
plt.axhline(np.mean(average_data_sonido_todos))
# %%
As = average_data_sonido_todos
Asp = np.concatenate(([0], np.diff(filtered_signal)))

plt.figure(figsize=(14,7))
plt.plot(common_time_base_sonido_todos,1_000*Asp,label = 'Derivada')
plt.plot(common_time_base_sonido_todos,sigm(1_000*Asp-0.5),label='Sigmoide')
# plt.plot(common_time_base_sonido_todos,(np.sign(1_000*Asp)+1)/2,label = 'Funcion signo')
plt.legend(fancybox=True, shadow=True)
plt.tight_layout()

#%%
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6


def sigm(x):
    return 1 / (1 + np.exp(-10*x))

# Vector field
def f(v, t, pars):

    w0, As, A_s0, t_up, t_down, Asp = [par for par in pars]
    w = v[0]
    #Sigmoide
    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1)
            * sigm(1_000*Asp-.5)) * (A_s0 * As + w0 - w)
    #Escalon
    # dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) * ((np.sign(1_000*Asp)+1)/2)) * (A_s0 * As + w0 - w)
    return [dwdt]

dt = np.mean(np.diff(common_time_base_sonido_todos))/5
As = average_data_sonido_todos
Asp = np.concatenate(([0], np.diff(filtered_signal)))

A_s0 =0.31 #1.75 / 4
t_up =0.05
t_down = 27
w0 = 1.2

#      w0, As, A_s0, t_up, t_down, Asp
pars = [w0, 1, A_s0, t_up, t_down, 1]

w = np.zeros_like(common_time_base_sonido_todos)

w[0] = 1.1
with tqdm(total=len(common_time_base_sonido_todos)) as pbar_h:
    for ix, tt in enumerate(common_time_base_sonido_todos[:-1]):
        pars[1] = As[ix] -.8
        pars[-1] = Asp[ix]
        w[ix+1] = rk4(f, [w[ix]], tt, dt, pars)
        pbar_h.update(1)
#%%
plt.close('prueba')
plt.figure('prueba',figsize=(14, 7))
plt.suptitle(f'$\\tau_c$,$\\tau_d$,$\omega_0$,$A_0$ = {t_up},{t_down},{w0},{round(A_s0,2)}')
# plt.errorbar(common_time_base_sonido_todos, average_data_sonido_todos, std_data_sonido_todos, label='RoNe', color='r', linewidth=2,alpha=0.1)
plt.errorbar(common_time_base_periodo_todos, average_data_periodo_todos,
             std_data_periodo_todos, label='RoNe rate', color='r', linewidth=2, alpha=0.5)
plt.plot(common_time_base_sonido_todos, w,label = 'Modelo nuevo')
# plt.plot(common_time_base_sonido_todos[picos_respuesta]
#         [1::], 1/periodo, color='g', label='Modelo hopf')
plt.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%%

plt.figure()
plt.plot(common_time_base_sonido_todos,average_data_sonido_todos)
plt.errorbar(common_time_base_periodo_todos, average_data_periodo_todos,
             std_data_periodo_todos, label='RoNe rate', color='r', linewidth=2, alpha=0.5)
    #%%
    
interpolado_periodo = UnivariateSpline(common_time_base_periodo_todos,average_data_periodo_todos, s=0, k=3)
interpolado = interpolado_periodo(common_time_base_sonido_todos)
print(sum((w - interpolado)**2))
# plt.axhline(np.mean((w - interpolado)**2))
#%% Barrido en los parametros
t_ups = np.linspace(0.05,3,10)
t_downs = np.linspace(3,30,10)
chis = []
with tqdm(total=10*10) as pbar_h:
    for t_up in t_ups:
        plt.figure(figsize=(14, 7))
        plt.suptitle(f'$\\tau_c$= {round(t_up,2)}')
        plt.errorbar(common_time_base_periodo_todos, average_data_periodo_todos,
                     std_data_periodo_todos, label='RoNe rate', color='r', linewidth=2, alpha=0.5)
        for t_down in t_downs:
            #      w0, As, A_s0, t_up, t_down, Asp
            pars = [w0, 1, A_s0, t_up, t_down, 1]
            
            w = np.zeros_like(common_time_base_sonido_todos)

            w[0] = 1.1

            for ix, tt in enumerate(common_time_base_sonido_todos[:-1]):
                pars[1] = As[ix] - 0.8
                pars[-1] = Asp[ix]
                w[ix+1] = rk4(f, [w[ix]], tt, dt, pars)
                
            pbar_h.update(1)
            
            plt.plot(common_time_base_sonido_todos, w,label = f'$\\tau_d={round(t_down,2)}$')
            plt.legend(fancybox=True, shadow=True)
            
            interpolado_periodo = UnivariateSpline(common_time_base_periodo_todos,average_data_periodo_todos, s=0, k=3)
            interpolado = interpolado_periodo(common_time_base_sonido_todos)
            
            chi_cuadrado = sum((w - interpolado)**2)
            chis.append(chi_cuadrado)

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
#%%
              
chis_2d = np.array(chis).reshape(len(t_ups), len(t_downs))
# Find the indices of the minimum value in chis_2d
min_index = np.unravel_index(np.argmin(chis_2d), chis_2d.shape)
min_t_up = t_ups[min_index[0]]
min_t_down = t_downs[min_index[1]]
print(f'minimo local = tau_d={round(min_t_down,2)}, tau_c={round(min_t_up,2)}')

# Create the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(chis_2d, aspect='auto', origin='lower', extent=[t_downs.min(), t_downs.max(), t_ups.min(), t_ups.max()])
plt.colorbar(label='$\chi^2$')
plt.scatter(min_t_down, min_t_up, color='red', s=100, edgecolor='black', label='Minimum $\\chi^2$')
plt.xlabel('$\\tau_d$')
plt.ylabel('$\\tau_c$')
plt.title('$\chi^2$ values for different $\\tau_c$ and $\\tau_d$')            
plt.legend(fancybox=True, shadow=True)                       
#%%

A_s0s = np.linspace(1,2,10)/4

with tqdm(total=10*10*10) as pbar_h:
    for A_s0 in A_s0s:
        
        chis = []
        for t_up in t_ups:
            # plt.figure(figsize=(14, 7))
            # plt.suptitle(f'$\\tau_c$= {round(t_up,2)}')
            # plt.errorbar(common_time_base_periodo_todos, average_data_periodo_todos,
            #              std_data_periodo_todos, label='RoNe rate', color='r', linewidth=2, alpha=0.5)
            for t_down in t_downs:
                #      w0, As, A_s0, t_up, t_down, Asp
                pars = [w0, 1, A_s0, t_up, t_down, 1]
                
                w = np.zeros_like(common_time_base_sonido_todos)
    
                w[0] = 1.1
    
                for ix, tt in enumerate(common_time_base_sonido_todos[:-1]):
                    pars[1] = As[ix] - 0.8
                    pars[-1] = Asp[ix]
                    w[ix+1] = rk4(f, [w[ix]], tt, dt, pars)
                    
                
                
                # plt.plot(common_time_base_sonido_todos, w,label = f'$\\tau_d={round(t_down,2)}$')
                # plt.legend(fancybox=True, shadow=True)
                
                interpolado_periodo = UnivariateSpline(common_time_base_periodo_todos,average_data_periodo_todos, s=0, k=3)
                interpolado = interpolado_periodo(common_time_base_sonido_todos)
                
                chi_cuadrado = sum((w - interpolado)**2)
                chis.append(chi_cuadrado)
                
                pbar_h.update(1)
        
        chis_2d = np.array(chis).reshape(len(t_ups), len(t_downs))
        # Find the indices of the minimum value in chis_2d
        min_index = np.unravel_index(np.argmin(chis_2d), chis_2d.shape)
        min_t_up = t_ups[min_index[0]]
        min_t_down = t_downs[min_index[1]]
        # print(f'minimo local = tau_d={round(min_t_down,2)}, tau_c={round(min_t_up,2)}')
    
        # Create the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(chis_2d, aspect='auto', origin='lower', extent=[t_downs.min(), t_downs.max(), t_ups.min(), t_ups.max()])
        plt.colorbar(label='$\chi^2$')
        plt.scatter(min_t_down, min_t_up, color='red', s=100, edgecolor='black', label=f'Minimum $\\chi^2$: $\\tau_d$={round(min_t_down,2)}, $\\tau_c$={round(min_t_up,2)}')
        plt.xlabel('$\\tau_d$')
        plt.ylabel('$\\tau_c$')
        plt.title(f'$A_0$ = {round(A_s0,2)}, $\\chi^2$:{round(min(chis),2)}')            
        plt.legend(fancybox=True, shadow=True)    

notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)



                
                
                