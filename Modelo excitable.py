# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:48:16 2024

@author: beneg
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import matplotlib.animation as animation


def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def funcion_patada(w,t,pars):
    w0, tau, A = pars
    dwdt = (-w + w0)/tau + A
    return dwdt


def integrate_system(time, dt, pars, indice_sound, indice_sound_max, amplitud_patada):
    # Initial conditions
    w = np.zeros_like(time)
    w[0] = pars[0]  # Set initial condition for P
    
    # Loop over time steps
    for ix in range(len(time) - 1):

        if indice_sound <= ix < indice_sound_max+1:
            pars[-1] = amplitud_patada
            w[ix + 1] = rk4(funcion_patada, w[ix], time[ix], dt, pars)
        else:
            pars[-1] = 0.0
            w[ix + 1] = rk4(funcion_patada, w[ix], time[ix], dt, pars)
    
    return w

#%% Importo datos de RoNe

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_RoNe = np.loadtxt('average RoNe sonido 300', delimiter=',')
Pressure_RoNe = np.loadtxt('average RoNe pressure', delimiter=',')
Rate_RoNe = np.loadtxt('average RoNe rate', delimiter=',')

time_sound_RoNe = Sound_RoNe[0]
sound_RoNe = Sound_RoNe[1]
error_s_RoNe = Sound_RoNe[2]


time_pressure_RoNe = Pressure_RoNe[0]
pressure_RoNe = Pressure_RoNe[1]
error_pressure_RoNe = Pressure_RoNe[2]

time_rate_RoNe = Rate_RoNe[0]
rate_RoNe = Rate_RoNe[1]
error_rate_RoNe = Rate_RoNe[2]

dt = np.mean(np.diff(time_sound_RoNe))
sound_derivative_RoNe = np.gradient(sound_RoNe,time_sound_RoNe)

#%% Rate RoNe
       #W0, tau, patada
pars = [1.25, 20, 0]
indice_rate = np.argmin(abs(time_rate_RoNe - 0))
indice_rate_maximo = np.argmin(abs(time_rate_RoNe - max(rate_RoNe)))
amplitud_patada = 0.3

rate_simulado = integrate_system(time_rate_RoNe, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate RoNe')
ax.errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe, label = 'Datos RoNe')
ax.plot(time_rate_RoNe,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_RoNe, sound_derivative_RoNe, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()
derivadamax_RoNe = time_sound_RoNe[np.argmax(sound_derivative_RoNe)]

def MSE(datos,simulacion,error_datos,weights=False):
    
    if weights == True:
        weights = 1 / (error_datos ** 2 + 1e-8)
        loss = np.mean(weights*(simulacion - datos) ** 2)
        
    else: 
        loss = np.mean((simulacion - datos) ** 2)
    return loss


weights = 1 / (error_pressure_RoNe ** 2 + 1e-8)
weighted_loss = np.mean( (rate_simulado - pressure_RoNe) ** 2)

print(weighted_loss)
#%% modelo presion excitable


pars = [0.95, 9, 0]
indice_rate = np.argmin(abs(time_pressure_RoNe - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_RoNe - max(pressure_RoNe)))
amplitud_patada = 0.6

presion_simulado = integrate_system(time_pressure_RoNe, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion RoNe')
ax.errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, label = 'Datos RoNe')
ax.plot(time_pressure_RoNe,presion_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_RoNe, sound_derivative_RoNe, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()

#%%

fig, ax = plt.subplots(figsize=(14,7))
ax.errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe, label = 'Datos RoNe')
ax.plot(time_rate_RoNe,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, label = 'Datos RoNe')
ax.plot(time_pressure_RoNe,presion_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_RoNe, sound_derivative_RoNe, label = 'Derivada sonido')
plt.tight_layout()

#%% Carg RoVio
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[1]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_RoVio = np.loadtxt('average RoVio sonido 300', delimiter=',')
Pressure_RoVio = np.loadtxt('average RoVio pressure', delimiter=',')
Rate_RoVio = np.loadtxt('average RoVio rate', delimiter=',')

time_sound_RoVio = Sound_RoVio[0]
sound_RoVio = Sound_RoVio[1]
error_s_RoVio = Sound_RoVio[2]

time_pressure_RoVio = Pressure_RoVio[0]
pressure_RoVio = Pressure_RoVio[1]
error_pressure_RoVio = Pressure_RoVio[2]

time_rate_RoVio = Rate_RoVio[0]
rate_RoVio = Rate_RoVio[1]
error_rate_RoVio = Rate_RoVio[2]

sound_derivative_RoVio = np.gradient(sound_RoVio,time_sound_RoVio)
#%% Modelo RoVio
       #W0, tau, patada
pars = [1.1, 15, 0]
indice_rate = np.argmin(abs(time_rate_RoVio - 0))
indice_rate_maximo = np.argmin(abs(time_rate_RoVio - max(rate_RoVio)))
amplitud_patada = 0.2

rate_simulado = integrate_system(time_rate_RoVio, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate RoVio')
ax.errorbar(time_rate_RoVio,rate_RoVio,error_rate_RoVio, label = 'Datos RoVio')
ax.plot(time_rate_RoVio,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_RoVio[np.argmax(sound_derivative_RoVio)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_RoVio, sound_derivative_RoVio, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()
derivadamax_RoVio = time_sound_RoVio[np.argmax(sound_derivative_RoVio)]
#%% Modelo presion excitable RoVio
pars = [0.9, 6, 0]
indice_rate = np.argmin(abs(time_pressure_RoVio - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_RoVio - max(pressure_RoVio)))
amplitud_patada = 0.5

presion_simulado = integrate_system(time_pressure_RoVio, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion RoVio')
ax.errorbar(time_pressure_RoVio, pressure_RoVio, error_pressure_RoVio, label = 'Datos RoVio')
ax.plot(time_pressure_RoVio,presion_simulado, linewidth=5, label = 'Modelo exitable')
ax.plot(time_pressure_RoVio[indice_rate_maximo+5],pressure_RoVio[indice_rate_maximo+5],'C2o', ms=11,label='maximo?')
ax.axvline(time_sound_RoVio[np.argmax(sound_derivative_RoVio)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_RoVio, sound_derivative_RoVio, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()

#%%
fig, ax = plt.subplots(figsize=(14,7))
ax.errorbar(time_rate_RoVio,rate_RoVio,error_rate_RoVio, label = 'Datos RoVio')
ax.plot(time_rate_RoVio,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.errorbar(time_pressure_RoVio, pressure_RoVio, error_pressure_RoVio, label = 'Datos RoVio')
ax.plot(time_pressure_RoVio,presion_simulado, linewidth=5, label = 'Modelo exitable')
plt.tight_layout()

#%% importo NaRo
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[2]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_NaRo = np.loadtxt('average NaRo sonido 300', delimiter=',')
Pressure_NaRo = np.loadtxt('average NaRo pressure', delimiter=',')
Rate_NaRo = np.loadtxt('average NaRo rate', delimiter=',')

time_sound_NaRo = Sound_NaRo[0]
sound_NaRo = Sound_NaRo[1]
error_s_NaRo = Sound_NaRo[2]

time_pressure_NaRo = Pressure_NaRo[0]
pressure_NaRo = Pressure_NaRo[1]
error_pressure_NaRo = Pressure_NaRo[2]

time_rate_NaRo = Rate_NaRo[0]
rate_NaRo = Rate_NaRo[1]
error_rate_NaRo = Rate_NaRo[2]

sound_derivative_NaRo = np.gradient(sound_NaRo,time_sound_NaRo)
#%% Modelo NaRo
       #W0, tau, patada
pars = [0.9, 15, 0]
indice_rate = np.argmin(abs(time_rate_NaRo - 0))
indice_rate_maximo = np.argmin(abs(time_rate_NaRo - max(rate_NaRo)))
amplitud_patada = 0.2

rate_simulado = integrate_system(time_rate_NaRo, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate NaRo')
ax.errorbar(time_rate_NaRo,rate_NaRo,error_rate_NaRo, label = 'Datos NaRo')
ax.plot(time_rate_NaRo,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_NaRo[np.argmax(sound_derivative_NaRo)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_NaRo, sound_derivative_NaRo, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()
derivadamax_NaRo = time_sound_NaRo[np.argmax(sound_derivative_NaRo)]
#%% Modelo excitable presion NaRo
pars = [0.95, 7, 0]
indice_rate = np.argmin(abs(time_pressure_NaRo - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_NaRo - max(pressure_NaRo)))
amplitud_patada = 0.6

rate_simulado = integrate_system(time_pressure_NaRo, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion NaRo')
ax.errorbar(time_pressure_NaRo, pressure_NaRo, error_pressure_NaRo, label = 'Datos NaRo')
ax.plot(time_pressure_NaRo,rate_simulado, linewidth=5, label = 'Modelo exitable')
ax.axvline(time_sound_NaRo[np.argmax(sound_derivative_NaRo)], color='k', linestyle='-',label='Maximo de la derivada')
ax.plot(time_sound_NaRo, sound_derivative_NaRo, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()

#%%
# hola = (np.sign(sound_derivative_RoNe-0.1)+1)*0.5

# def funcion(P, t, pars):
       
#     A_0, P0, t_up, t_down, As, Asp = pars
#     dPdt = (t_down**-1 - (t_down**-1 - t_up**-1)*Asp)* (A_0 * As + P0 - P)
#     return dPdt

# # Integration function for the system using PyTorch
# def integrate_system(time, dt, pars, sound, sound_derivative):
#     # Initial conditions
#     P = np.zeros_like(time)
#     P[0] = 1.0  # Set initial condition for P

#     # Loop over time steps
#     for ix in range(len(time) - 1):
#         pars[-2] = sound[ix]  # Get As value at current time step
#         pars[-1] = sound_derivative[ix]
#         P[ix + 1] = rk4(funcion, P[ix], time[ix], dt, pars)
    
#     return P
#         #A_0, P0, t_up, t_down, sound, derivative 
# pars = [0.7, 1, 0.25, 2, 1, 1]
# P_final = integrate_system(time_sound_RoNe, dt, pars, sound_RoNe-0.6,hola)

# # plt.close(fig)
# fig, ax = plt.subplots(2,1,figsize=(14,7))
# ax[0].plot(time_sound_RoNe,sound_RoNe-0.6)
# ax[0].plot(time_sound_RoNe,hola)
# ax[0].errorbar(time_pressure_RoNe, pressure_RoNe-1, yerr= error_pressure_RoNe, label='Pressure_values (target)', linestyle='--')
# ax[1].plot(time_sound_RoNe, P_final, label='P_result')
# ax[1].errorbar(time_pressure_RoNe, pressure_RoNe, yerr= error_pressure_RoNe, label='Pressure_values (target)', linestyle='--')
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Pressure')
# ax[0].set_ylabel('Sound')
# ax[1].legend()
#%%

# def funcion_seguidor(P, t, pars):
#     A_0, tau, As = pars
#     dPdt = (-P + A_0 * As + 1) / tau
#     return dPdt

# # Integration function for the system using PyTorch
# def integrate_system_seguidor(time, dt, pars, As_values):
#     # Initial conditions
#     P = np.zeros_like(time)
#     P[0] = 1.0  # Set initial condition for P

#     # Loop over time steps
#     for ix in range(len(time) - 1):
#         pars[-1] = As_values[ix]  # Get As value at current time step
#         P[ix + 1] = rk4(funcion_seguidor, P[ix], time[ix], dt, pars)
    
#     return P
#        #A_0,tau,sonido
# pars = [0.7,0.25,1]
# P_final_2 = integrate_system_seguidor(time_sound_RoNe, dt, pars, sound_RoNe-0.6)
# plt.close(fig)
# fig, ax = plt.subplots(2,1,figsize=(14,7))
# ax[0].plot(time_sound_RoNe,sound_RoNe-0.6)
# ax[0].plot(time_sound_RoNe,hola)
# ax[0].errorbar(time_pressure_RoNe, pressure_RoNe-1, yerr= error_pressure_RoNe, label='Pressure_values (target)', linestyle='--')
# ax[1].plot(time_sound_RoNe, P_final_2, label='P_result')
# ax[1].errorbar(time_pressure_RoNe, pressure_RoNe, yerr= error_pressure_RoNe, label='Pressure_values (target)', linestyle='--')
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Pressure')
# ax[0].set_ylabel('Sound')
# ax[1].legend()


#%% RoNe dia


directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_RoNe_day = np.loadtxt('average RoNe sonido 300 day', delimiter=',')
Pressure_RoNe_day = np.loadtxt('average RoNe pressure day', delimiter=',')
Rate_RoNe_day = np.loadtxt('average RoNe rate day', delimiter=',')

time_sound_RoNe_day = Sound_RoNe_day[0]
sound_RoNe_day = Sound_RoNe_day[1]
error_s_RoNe_day = Sound_RoNe_day[2]


time_pressure_RoNe_day = Pressure_RoNe_day[0]
pressure_RoNe_day = Pressure_RoNe_day[1]
error_pressure_RoNe_day = Pressure_RoNe_day[2]

time_rate_RoNe_day = Rate_RoNe_day[0]
rate_RoNe_day = Rate_RoNe_day[1]
error_rate_RoNe_day = Rate_RoNe_day[2]

dt_day = np.mean(np.diff(time_sound_RoNe_day))
sound_derivative_RoNe_day = np.gradient(sound_RoNe_day,time_sound_RoNe_day)


#%% Rate RoNe dia
       #W0, tau, patada
pars = [1.95, 5, 0]
indice_rate = np.argmin(abs(time_rate_RoNe_day - 0))
# indice_rate_maximo = np.argmin(abs(time_rate_RoNe_day - max(rate_RoNe_day)))
indice_rate_maximo = 150
amplitud_patada = 0.25

rate_simulado = integrate_system(time_rate_RoNe_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate RoNe Day')
ax.errorbar(time_rate_RoNe_day, rate_RoNe_day, error_rate_RoNe_day, label='Datos RoNe Day')
ax.plot(time_rate_RoNe_day, rate_simulado, linewidth=5, label='Modelo exitable')
ax.axvline(time_sound_RoNe_day[np.argmax(sound_derivative_RoNe_day)], color='k', linestyle='-', label='Maximo de la derivada')
# ax.plot(time_sound_RoNe_day, sound_derivative_RoNe_day, label='Derivada sonido')
# ax.plot(time_sound_RoNe_day,sound_RoNe_day)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()
#%%

fig, ax = plt.subplots(3,1,figsize=(14,7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()
ax[0].errorbar(time_sound_RoNe_day,sound_RoNe_day,error_s_RoNe_day)
ax[1].errorbar(time_pressure_RoNe_day,pressure_RoNe_day,error_pressure_RoNe_day)
ax[2].errorbar(time_rate_RoNe_day, rate_RoNe_day, error_rate_RoNe_day, label='Datos RoNe Day')

#%% Modelo presion excitable dia

pars = [0.95, 9, 0]
indice_rate = np.argmin(abs(time_pressure_RoNe_day - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_RoNe_day - max(pressure_RoNe_day)))
amplitud_patada = 0.6

rate_simulado = integrate_system(time_pressure_RoNe_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion RoNe Day')
ax.errorbar(time_pressure_RoNe_day, pressure_RoNe_day, error_pressure_RoNe_day, label='Datos RoNe Day')
ax.errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, label = 'Datos RoNe')
# ax.plot(time_pressure_RoNe_day, rate_simulado, linewidth=5, label='Modelo exitable')
# ax.axvline(time_sound_RoNe_day[np.argmax(sound_derivative_RoNe_day)], color='k', linestyle='-', label='Maximo de la derivada')
# ax.plot(time_sound_RoNe_day, sound_derivative_RoNe_day, label='Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%% RoVio dia

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoVio
pajaro = carpetas[1]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_RoVio_day = np.loadtxt('average RoVio sonido 300 day', delimiter=',')
Pressure_RoVio_day = np.loadtxt('average RoVio pressure day', delimiter=',')
Rate_RoVio_day = np.loadtxt('average RoVio rate day', delimiter=',')

time_sound_RoVio_day = Sound_RoVio_day[0]
sound_RoVio_day = Sound_RoVio_day[1]
error_s_RoVio_day = Sound_RoVio_day[2]

time_pressure_RoVio_day = Pressure_RoVio_day[0]
pressure_RoVio_day = Pressure_RoVio_day[1]
error_pressure_RoVio_day = Pressure_RoVio_day[2]

time_rate_RoVio_day = Rate_RoVio_day[0]
rate_RoVio_day = Rate_RoVio_day[1]
error_rate_RoVio_day = Rate_RoVio_day[2]

dt_day_RoVio = np.mean(np.diff(time_sound_RoVio_day))
sound_derivative_RoVio_day = np.gradient(sound_RoVio_day, time_sound_RoVio_day)

#%% Rate RoVio dia
       #W0, tau, patada
pars = [1.25, 20, 0]
indice_rate = np.argmin(abs(time_rate_RoVio_day - 0))
indice_rate_maximo = np.argmin(abs(time_rate_RoVio_day - max(rate_RoVio_day)))
amplitud_patada = 0.3

rate_simulado = integrate_system(time_rate_RoVio_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate RoVio Day')
ax.errorbar(time_rate_RoVio_day, rate_RoVio_day, error_rate_RoVio_day, label='Datos RoVio Day')
ax.plot(time_rate_RoVio_day, rate_simulado, linewidth=5, label='Modelo exitable')
ax.axvline(time_sound_RoVio_day[np.argmax(sound_derivative_RoVio_day)], color='k', linestyle='-', label='Maximo de la derivada')
ax.plot(time_sound_RoVio_day, sound_derivative_RoVio_day, label='Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%% Modelo presion excitable dia

pars = [0.95, 9, 0]
indice_rate = np.argmin(abs(time_pressure_RoVio_day - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_RoVio_day - max(pressure_RoVio_day)))
amplitud_patada = 0.6

rate_simulado = integrate_system(time_pressure_RoVio_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion RoVio Day')
ax.errorbar(time_pressure_RoVio_day, pressure_RoVio_day, error_pressure_RoVio_day, label='Datos RoVio Day')
ax.plot(time_pressure_RoVio_day, rate_simulado, linewidth=5, label='Modelo exitable')
ax.axvline(time_sound_RoVio_day[np.argmax(sound_derivative_RoVio_day)], color='k', linestyle='-', label='Maximo de la derivada')
ax.plot(time_sound_RoVio_day, sound_derivative_RoVio_day, label='Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%% NaRo dia

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## NaRo
pajaro = carpetas[2]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound_NaRo_day = np.loadtxt('average NaRo sonido 300 day', delimiter=',')
Pressure_NaRo_day = np.loadtxt('average NaRo pressure day', delimiter=',')
Rate_NaRo_day = np.loadtxt('average NaRo rate day', delimiter=',')

time_sound_NaRo_day = Sound_NaRo_day[0]
sound_NaRo_day = Sound_NaRo_day[1]
error_s_NaRo_day = Sound_NaRo_day[2]

time_pressure_NaRo_day = Pressure_NaRo_day[0]
pressure_NaRo_day = Pressure_NaRo_day[1]
error_pressure_NaRo_day = Pressure_NaRo_day[2]

time_rate_NaRo_day = Rate_NaRo_day[0]
rate_NaRo_day = Rate_NaRo_day[1]
error_rate_NaRo_day = Rate_NaRo_day[2]

dt_day_NaRo = np.mean(np.diff(time_sound_NaRo_day))
sound_derivative_NaRo_day = np.gradient(sound_NaRo_day, time_sound_NaRo_day)


#%% Rate NaRo dia
       #W0, tau, patada
pars = [1.25, 20, 0]
indice_rate = np.argmin(abs(time_rate_NaRo_day - 0))
indice_rate_maximo = np.argmin(abs(time_rate_NaRo_day - max(rate_NaRo_day)))
amplitud_patada = 0.3

rate_simulado = integrate_system(time_rate_NaRo_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate NaRo Day')
ax.errorbar(time_rate_NaRo_day, rate_NaRo_day, error_rate_NaRo_day, label='Datos NaRo Day')
ax.plot(time_rate_NaRo_day, rate_simulado, linewidth=5, label='Modelo exitable')
ax.axvline(time_sound_NaRo_day[np.argmax(sound_derivative_NaRo_day)], color='k', linestyle='-', label='Maximo de la derivada')
ax.plot(time_sound_NaRo_day, sound_derivative_NaRo_day, label='Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%% Modelo presion excitable dia

pars = [0.95, 9, 0]
indice_rate = np.argmin(abs(time_pressure_NaRo_day - 0))
indice_rate_maximo = np.argmin(abs(time_pressure_NaRo_day - max(pressure_NaRo_day)))
amplitud_patada = 0.6

rate_simulado = integrate_system(time_pressure_NaRo_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Presion NaRo Day')
ax.errorbar(time_pressure_NaRo_day, pressure_NaRo_day, error_pressure_NaRo_day, label='Datos NaRo Day')
ax.plot(time_pressure_NaRo_day, rate_simulado, linewidth=5, label='Modelo exitable')
ax.axvline(time_sound_NaRo_day[np.argmax(sound_derivative_NaRo_day)], color='k', linestyle='-', label='Maximo de la derivada')
ax.plot(time_sound_NaRo_day, sound_derivative_NaRo_day, label='Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True, shadow=True)
plt.tight_layout()


#%%

print(derivadamax_RoNe,derivadamax_RoVio,derivadamax_NaRo)



#%%

