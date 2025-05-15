# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:32:20 2025

@author: beneg
"""

import os
# import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import matplotlib.animation as animation
from scipy.signal import savgol_filter

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

def MSE(datos,simulacion,error_datos,weights=False):
    
    if weights == True:
        weights = 1 / (error_datos ** 2 + 1e-8)
        loss = np.mean(weights*(simulacion - datos) ** 2)
        
    else: 
        loss = np.mean((simulacion - datos) ** 2)
    return loss


def create_centered_list(number):
    length = 11
    center_index = length // 2  # Center index is 5 for a list of length 11
    
    if isinstance(number, int):
        # Create a list of consecutive integers
        start = number - center_index
        centered_list = [start + i for i in range(length)]
    else:
        # Create a list with values separated by 0.05
        start = number - center_index * 0.05
        centered_list = [start + i * 0.05 for i in range(length)]
    
    return centered_list

def optimizadorV2(time:list, datos:list, error_datos:list, taus:list, amplitudes_patada:list, indices_maximos:list):
    
    dt = np.mean(np.diff(time))
    
    indice_0 = np.argmin(abs(time - 0))
    
    w0 = np.mean(datos[:indice_0])
    
    errores = []
    parametros = []

    best_error = float('inf')
    best_params = None

    for tau in taus:
        for patada in amplitudes_patada:
            for indice_max in indices_maximos:
                pars = [w0, tau, 0]
                amplitud_patada = patada
                
                rate_simulado = integrate_system(time, dt, pars, indice_0, indice_max, amplitud_patada)
                error_ajuste = MSE(datos, rate_simulado, error_datos)
                
                errores.append(error_ajuste)
                parametros.append((tau, indice_max, patada))

                if error_ajuste < best_error:
                    best_error = error_ajuste
                    best_params = (tau, indice_max, patada)

    # Print best combination
    print(f"Best parameters: tau={best_params[0]}, t_max={best_params[1]}, amplitud_patada={best_params[2]}")
    print(f"Minimum error: {best_error}")
    
    return best_params

def optimizadorV3(time:list, datos:list, error_datos:list, tau, amplitud_patada, indice_maximo):
    
    dt = np.mean(np.diff(time))
    
    indice_0 = np.argmin(abs(time - 0))
    
    w0 = np.mean(datos[:indice_0])
    
    taus = create_centered_list(tau)
    amplitudes_patada = create_centered_list(amplitud_patada)
    indices_maximos = create_centered_list(int(indice_maximo))
    
    errores = []
    parametros = []

    best_error = float('inf')
    best_params = None

    for tau in taus:
        for patada in amplitudes_patada:
            for indice_max in indices_maximos:
                pars = [w0, tau, 0]
                amplitud_patada = patada
                
                rate_simulado = integrate_system(time, dt, pars, indice_0, indice_max, amplitud_patada)
                error_ajuste = MSE(datos, rate_simulado, error_datos)
                
                errores.append(error_ajuste)
                parametros.append((tau, indice_max, patada))

                if error_ajuste < best_error:
                    best_error = error_ajuste
                    best_params = (tau, indice_max, patada)
                    
    error_tau = np.mean(np.diff(taus))
    error_amplitud = np.mean(np.diff(amplitudes_patada))
    error_indices = time[best_params[1]+1] - time[best_params[1]]
    # Print best combination
    print(f"Best parameters: tau={best_params[0]}+/- {error_tau}, t_max={time[best_params[1]+ 1]} +/- {error_indices}, amplitud_patada={best_params[2]} +/- {error_amplitud}")
    print(f"Minimum error: {best_error}")
    
    return best_params

def simulacion_opt(time,datos,pars_opt):
    dt = np.mean(np.diff(time))
    
    indice_0 = np.argmin(abs(time - 0))
    
    w0 = np.mean(datos[:indice_0])
    
    pars = [w0, pars_opt[0],0]
    indice_rate = np.argmin(abs(time - 0))
    amplitud_patada = pars_opt[2]


    simulacion = integrate_system(time, dt, pars, indice_rate, pars_opt[1], amplitud_patada)

    return simulacion

def plot_simulacion(time,datos,error,simulacion,ylabel,title):
    fig, ax = plt.subplots(figsize=(14,7))
    fig.suptitle(title,fontsize=20)
    ax.set_ylabel(ylabel,fontsize=14)
    ax.set_xlabel('Time (s)',fontsize=14)
    ax.errorbar(time,datos,error,label='Data')
    ax.plot(time,simulacion,linewidth=5,label='Model')
    
    ax.legend(fancybox=True, shadow=True)
    
    plt.tight_layout()
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
derivadamax_RoNe = time_sound_RoNe[np.argmax(sound_derivative_RoNe)]
# #%% Rate RoNe
#        #W0, tau, patada
# pars = [1.25, 20, 0]
# indice_rate = np.argmin(abs(time_rate_RoNe - 0))
# indice_rate_maximo = np.argmin(abs(time_rate_RoNe - max(rate_RoNe)))
# amplitud_patada = 0.3

# rate_simulado = integrate_system(time_rate_RoNe, dt, pars, indice_rate, indice_rate_maximo+5, amplitud_patada)
  
# fig, ax = plt.subplots(figsize=(14,7))
# ax.set_title('Rate RoNe')
# ax.errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe, label = 'Datos RoNe')
# ax.plot(time_rate_RoNe,rate_simulado, linewidth=5, label = 'Modelo exitable')
# ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada')
# ax.plot(time_sound_RoNe, sound_derivative_RoNe, label = 'Derivada sonido')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Rate (Hz)')
# ax.legend(fancybox=True,shadow=True)
# plt.tight_layout()

            
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

indice_rate_maximo = np.argmin(abs(time_rate_RoVio - max(rate_RoVio)))

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


#%% Optimizaciones

#Optimizacion de RoNe
indice_maximos_RoNe = np.argmax(sound_derivative_RoNe)

params_opt_RoNe_rate_V3 = optimizadorV3(time_rate_RoNe, rate_RoNe, error_rate_RoNe, 20, 0.3, indice_maximos_RoNe)

params_opt_RoNe_presion_V3 = optimizadorV3(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, 9, 0.6, indice_maximos_RoNe)



# Optimizacion RoVio
indice_maximo_RoVio = np.argmax(sound_derivative_RoVio)

params_opt_RoVio_rate_V3 = optimizadorV3(time_rate_RoVio, rate_RoVio, error_rate_RoVio, 15, 0.2, indice_maximo_RoVio)

params_opt_RoVio_presion_V3 = optimizadorV3(time_pressure_RoVio, pressure_RoVio, error_pressure_RoVio, 6, 0.5, indice_maximo_RoVio)



# optimizacion de NaRo
indice_maximo_NaRo = np.argmax(sound_derivative_NaRo)

params_opt_NaRo_rate_V3 = optimizadorV3(time_rate_NaRo, rate_NaRo, error_rate_NaRo, 15, 0.2, indice_maximo_NaRo)

params_opt_NaRo_presion_V3 = optimizadorV3(time_pressure_NaRo, pressure_NaRo, error_pressure_NaRo, 7, 0.6, indice_maximo_NaRo)

#%%

pars = [1.25, params_opt_RoNe_rate_V3[0], 0]
indice_rate = np.argmin(abs(time_rate_RoNe - 0))
amplitud_patada = params_opt_RoNe_rate_V3[2]

rate_simulado = integrate_system(time_rate_RoNe, dt, pars, indice_rate, params_opt_RoNe_rate_V3[1], amplitud_patada)
  


pars = [0.95, params_opt_RoNe_presion_V3[0],0]
indice_rate = np.argmin(abs(time_pressure_RoNe - 0))
amplitud_patada = params_opt_RoNe_presion_V3[2]


presion_simulado = integrate_system(time_pressure_RoNe, dt, pars, indice_rate, params_opt_RoNe_presion_V3[1], amplitud_patada)

#%% Grafico de las dos RoNe

fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)

ax[0].set_ylabel("Presion (u. a.)", fontsize=20)
ax[1].set_ylabel("Rate (Hz)", fontsize=20)
ax[1].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)


ax[0].errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe,color='royalblue', label = 'Datos RoNe')
ax[0].plot(time_pressure_RoNe,presion_simulado, linewidth=5,color='#890304', label = 'Modelo',zorder=50)
# ax[0].axvspan(time_pressure_RoNe[params_opt_RoNe_presion_V3[1]], time_pressure_RoNe[params_opt_RoNe_presion_V3[1]+2], facecolor='g', alpha=0.5, edgecolor='k', linestyle='--',label='Tiempo optimo')
# ax[0].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--',label='Maximo de la derivada')

ax[1].errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe,color='royalblue')
ax[1].plot(time_rate_RoNe,rate_simulado, linewidth=5,color='#890304',zorder=50)
# ax[1].axvspan(time_rate_RoNe[params_opt_RoNe_rate_V3[1]], time_rate_RoNe[params_opt_RoNe_rate_V3[1]+2], facecolor='g', alpha=0.5, edgecolor='k', linestyle='--')
# ax[1].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--')

# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.951), bbox_transform=fig.transFigure, ncol=4, fontsize=12)
# fig.legend(loc='outside upper center', ncol=4, fontsize=12,mode='expand',prop={'size': 15})
ax[0].legend(fancybox=True,shadow=True, loc='upper left', fontsize=12,prop={'size': 20})
# plt.tight_layout(h_pad=5, rect=[0, 0, 1, 0.96]) 
plt.tight_layout()
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('rungekutta RoNe.pdf')


#%%
import matplotlib.animation as animation


# Create figure and axes
fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

# Set labels
ax[0].set_ylabel("Presion (u. a.)", fontsize=20)
ax[1].set_ylabel("Rate (Hz)", fontsize=20)
ax[1].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)

# Set up line objects (empty at first)
line_presion, = ax[0].plot([], [], color='#890304', linewidth=5, label='Modelo', zorder=50)
line_presion_data = ax[0].errorbar([], [], yerr=[], color='royalblue', label='Datos RoNe')[0]

line_rate, = ax[1].plot([], [], color='#890304', linewidth=5, zorder=50)
line_rate_data = ax[1].errorbar([], [], yerr=[], color='royalblue')[0]

# Set axes limits (optional: adjust to your data)
ax[0].set_xlim(time_pressure_RoNe[0], time_pressure_RoNe[-1])
ax[0].set_ylim(min(pressure_RoNe)-0.1, max(pressure_RoNe)+0.1)
ax[1].set_ylim(min(rate_RoNe)-0.1, max(rate_RoNe)+0.1)

# Add legend
ax[0].legend(fancybox=True, shadow=True, loc='upper left', fontsize=12, prop={'size': 20})

# Animation function
def update(frame):
    i = frame
    line_presion.set_data(time_pressure_RoNe[:i], presion_simulado[:i])
    line_presion_data.set_data(time_pressure_RoNe[:i], pressure_RoNe[:i])
    line_rate.set_data(time_rate_RoNe[:i], rate_simulado[:i])
    line_rate_data.set_data(time_rate_RoNe[:i], rate_RoNe[:i])
    return line_presion, line_rate, line_presion_data, line_rate_data

# Number of frames
frames = min(len(time_pressure_RoNe), len(time_rate_RoNe))

# Create animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

plt.tight_layout()
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Graficos defensa'
# os.chdir(directory)  # Change the working directory to the specified path
# writer = animation.PillowWriter(fps=15,
#                                 metadata=dict(artist='Me'),
#                                 bitrate=1800)
# ani.save("animation.gif", writer=writer)
#%%
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(-2, 6)
ax.set_ylim(-1, 1)
ax.axis('off')

# Create base elements
line, = ax.plot([-2, 5], [0, 0], color='black', lw=1)

arrow1 = ax.arrow(-1.5, 0, 0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
arrow2 = ax.arrow(1.5, 0, -0.5, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Points
dot_A0, = ax.plot(0, 0, 'ko', ms=8)
dot_C, = ax.plot([], [], 'ko', ms=8)

# Labels
label_A0 = ax.text(0, -0.3, r"$A_0$", fontsize=16, ha='center')
label_C = ax.text(0, -0.3, "", fontsize=16, ha='center')

# Arrows for perturbed system
arrow3 = None
arrow4 = None

def init():
    dot_C.set_data([], [])
    label_C.set_text("")
    return dot_A0, dot_C, label_C

def update(frame):
    if frame < 30:
        # Show original system
        dot_A0.set_data(0, 0)
        label_A0.set_position((0, -0.3))
        dot_C.set_data([], [])
        label_C.set_text("")
    else:
        # Move A0 to left, add C to the right
        dot_A0.set_data(-1.5, 0)
        label_A0.set_position((-1.5, -0.3))
        dot_C.set_data(1.5, 0)
        label_C.set_text(r"$C$")
        label_C.set_position((1.5, -0.3))
    return dot_A0, dot_C, label_A0, label_C

ani = animation.FuncAnimation(fig, update, frames=60, init_func=init, interval=100, blit=True)

#%% Grafico comparando con las derivada del sonido
from scipy.signal import savgol_filter


yhat = savgol_filter(sound_derivative_RoNe, 11, 3)

fig, ax = plt.subplots(4,1,figsize=(15,9),sharex=True)
ax[0].set_ylabel("Sonido (u. a.)", fontsize=20)
ax[1].set_ylabel("Derivada sonido (u. a.)", fontsize=20)
ax[2].set_ylabel("Presion (u. a.)", fontsize=20)
ax[3].set_ylabel("Rate (Hz)", fontsize=20)
ax[3].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[2].tick_params(axis='both', labelsize=10)
ax[3].tick_params(axis='both', labelsize=10)

ax[0].errorbar(time_sound_RoNe,sound_RoNe,error_s_RoNe,color='royalblue')
ax[1].plot(time_sound_RoNe,yhat,color='royalblue')
ax[2].errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe,color='royalblue', label = 'Datos RoNe')
ax[2].plot(time_pressure_RoNe,presion_simulado, linewidth=5,color='#890304', label = 'Modelo')
ax[3].errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe,color='royalblue')
ax[3].plot(time_rate_RoNe,rate_simulado, linewidth=5,color='#890304')

ax[0].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--',label='Maximo de la derivada')
ax[1].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--')
ax[2].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--')
ax[3].axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--')
ax[0].set_xlim([-50,58])
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.951), bbox_transform=fig.transFigure, ncol=3, fontsize=12,prop={'size': 15})
fig.legend(loc='outside upper center', ncol=3, fontsize=12,mode='expand',prop={'size': 15})
# ax[0].legend(fancybox=True,shadow=True, loc='upper left', fontsize=12,prop={'size': 20})
plt.tight_layout(h_pad=5, rect=[0, 0, 1, 0.96]) 

# plt.savefig('comparacion derivada ajuste.pdf')
#%%
print("RoNe")
print(f"tiempo rate: {time_rate_RoNe[params_opt_RoNe_rate_V3[1]+1]}")
print(f"tiempo presion: {time_pressure_RoNe[params_opt_RoNe_presion_V3[1]+1]}")

print("RoVio")
print(f"tiempo rate: {time_rate_RoVio[params_opt_RoVio_rate_V3[1]+1]}")
print(f"tiempo presion: {time_pressure_RoVio[params_opt_RoVio_presion_V3[1]+1]}")

print("NaRo")
print(f"tiempo rate: {time_rate_NaRo[params_opt_NaRo_rate_V3[1]+1]}")
print(f"tiempo presion: {time_pressure_NaRo[params_opt_NaRo_presion_V3[1]+1]}")

#%% Grafico NaRo

pars = [0.9, params_opt_NaRo_rate_V3[0], 0]
indice_rate = np.argmin(abs(time_rate_NaRo - 0))
amplitud_patada = params_opt_NaRo_rate_V3[2]

rate_simulado = integrate_system(time_rate_NaRo, dt, pars, indice_rate, params_opt_NaRo_rate_V3[1], amplitud_patada)
yhat = savgol_filter(sound_derivative_NaRo, 21, 3)
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Rate NaRo')
ax.errorbar(time_rate_NaRo,rate_NaRo,error_rate_NaRo, label = 'Datos NaRo')
ax.plot(time_rate_NaRo,rate_simulado, linewidth=5, label = 'Modelo')
ax.axvline(time_sound_NaRo[np.argmax(sound_derivative_NaRo)], color='k', linestyle='-',label='Maximo de la derivada')
ax.axvline(time_rate_NaRo[params_opt_NaRo_rate_V3[1]+1], color='r', linestyle='-',label='Tiempo optimo')
# ax.plot(time_sound_RoNe, yhat, label = 'Derivada sonido')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()
#%%

fig, ax = plt.subplots(figsize=(14,7))
ax.errorbar(time_sound_NaRo,sound_NaRo,error_s_NaRo)
ax.errorbar(time_rate_NaRo,rate_NaRo,error_rate_NaRo, label = 'Datos NaRo')
ax.plot(time_sound_NaRo,yhat)
ax.plot(time_sound_NaRo,sound_derivative_NaRo)
ax.axvline(time_sound_NaRo[np.argmax(yhat)], color='k', linestyle='-',label='Maximo de la derivada')
ax.axvline(time_sound_NaRo[np.argmax(sound_derivative_NaRo)], color='b', linestyle='-',label='Maximo de la derivada')

# pars = [0.95, params_opt_NaRo_presion_V3[0],0]
# indice_rate = np.argmin(abs(time_pressure_RoNe - 0))
# amplitud_patada = params_opt_RoNe_presion_V3[2]


# presion_simulado = integrate_system(time_pressure_RoNe, dt, pars, indice_rate, params_opt_RoNe_presion_V3[1], amplitud_patada)


# fig, ax = plt.subplots(figsize=(14,7))
# ax.set_title('Presion NaRo')
# ax.errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, label = 'Datos RoNe')
# ax.plot(time_pressure_RoNe,presion_simulado, linewidth=5, label = 'Modelo')
# ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada')
# ax.axvline(time_pressure_RoNe[params_opt_RoNe_presion_V3[1]], color='r', linestyle='-',label='Tiempo optimo')
# # ax.plot(time_sound_RoNe, sound_derivative_RoNe, label = 'Derivada sonido')
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Rate (Hz)')
# ax.legend(fancybox=True,shadow=True)
# plt.tight_layout()


#%% suavizado de la derivada

from scipy.signal import savgol_filter


yhat = savgol_filter(sound_derivative_RoNe, 11, 3)
# yhat2 = savgol_filter(sound_derivative_NaRo, 21, 3)
fig, ax = plt.subplots(figsize=(14,7))
# ax.plot(sound_RoNe)
ax.plot(time_sound_RoNe,sound_derivative_RoNe)
ax.plot(time_sound_RoNe,yhat)
# ax.plot(time_sound_NaRo,yhat2)
ax.axvline(time_sound_RoNe[np.argmax(sound_derivative_RoNe)], color='k', linestyle='-',label='Maximo de la derivada1')
ax.axvline(time_sound_RoNe[np.argmax(yhat)], color='r', linestyle='-',label='Maximo de la derivada2')
# ax.axvline(time_sound_RoNe[np.argmax(yhat2)], color='b', linestyle='-',label='Maximo de la derivada3')
ax.legend()

# soundhat = savgol_filter(sound_RoNe, 21, 3)
# fig, ax = plt.subplots(figsize=(14,7))
# ax.plot(time_sound_RoNe,sound_RoNe)
# ax.plot(time_sound_RoNe,soundhat)
# ax.plot(time_sound_RoNe,np.gradient(soundhat,time_sound_RoNe))
# ax.plot(time_sound_RoNe,sound_derivative_RoNe)
# # ax.plot(time_sound_RoNe,yhat)
#%%
derivada = np.gradient(soundhat,time_sound_RoNe)
derivada_seghat = savgol_filter(np.gradient(derivada,time_sound_RoNe), 21, 3)

fig, ax = plt.subplots(figsize=(14,7))

ax.plot(time_sound_RoNe,derivada)
ax.plot(time_sound_RoNe,derivada*derivada_seghat)

#%%

    
plot_simulacion(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe, presion_simulado, ylabel= 'Pressure (arb. u.)',title='RoNe')
plot_simulacion(time_rate_RoNe,rate_RoNe,error_rate_RoNe, rate_simulado, ylabel = 'Rate (Hz)',title='RoNe') 
    
    
#%% simulacion RoVio

    

rate_RoNe_sim = simulacion_opt(time_rate_RoNe,rate_RoNe,params_opt_RoNe_rate_V3)

plot_simulacion(time_rate_RoNe,rate_RoNe,error_rate_RoNe, rate_RoNe_sim, ylabel= 'Pressure (arb. u.)',title='RoNe')
    


rate_RoVio_sim = simulacion_opt(time_rate_RoVio, rate_RoVio, params_opt_RoVio_rate_V3)
plot_simulacion(time_rate_RoVio,rate_RoVio,error_rate_RoVio, rate_RoVio_sim, ylabel= 'Pressure (arb. u.)',title='RoVio')
    
#%%

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




#%%

indice_maximos_RoNe_day = np.argmax(sound_derivative_RoNe_day)


params_opt_RoNe_rate_V3_day = optimizadorV3(time_rate_RoNe_day, rate_RoNe_day, error_rate_RoNe_day, 8, 0.25, 150)

params_opt_RoNe_presion_V3_day = optimizadorV3(time_pressure_RoNe_day, pressure_RoNe_day, error_pressure_RoNe_day, 5, 0.6, indice_maximos_RoNe_day)

#%%
pars = [1.95, params_opt_RoNe_rate_V3_day[0], 0]
indice_rate = np.argmin(abs(time_rate_RoNe_day - 0))
amplitud_patada = params_opt_RoNe_rate_V3_day[2]

rate_simulado = integrate_system(time_rate_RoNe_day, dt_day, pars, indice_rate, params_opt_RoNe_rate_V3_day[1], amplitud_patada)


plot_simulacion(time_rate_RoNe_day, rate_RoNe_day, error_rate_RoNe_day, rate_simulado, ylabel='Rate', title='Rate dia')
#%%


pars = [1.95, 5, 0]
indice_rate = np.argmin(abs(time_rate_RoNe_day - 0))
# indice_rate_maximo = np.argmin(abs(time_rate_RoNe_day - max(rate_RoNe_day)))
indice_rate_maximo = 150
amplitud_patada = 0.25

rate_simulado = integrate_system(time_rate_RoNe_day, dt_day, pars, indice_rate, indice_rate_maximo + 5, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(15,9))
ax.set_ylabel("Rate (Hz)", fontsize=20)
ax.set_xlabel("Tiempo (s)", fontsize=20)
ax.tick_params(axis='both', labelsize=10)

# ax.set_title('Rate RoNe Day')
ax.errorbar(time_rate_RoNe_day[:-1], rate_RoNe_day[:-1], error_rate_RoNe_day[:-1],color='royalblue', label='Datos RoNe dia')
ax.plot(time_rate_RoNe_day, rate_simulado, linewidth=5,color='#890304',label='Modelo')
# ax.axvline(5.57, color='k', linestyle='-', label='Maximo de la derivada')
ax.axvspan(time_sound_RoNe[np.argmax(sound_derivative_RoNe)-1], time_sound_RoNe[np.argmax(sound_derivative_RoNe)+1], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='--',label='Maximo de la derivada')

ax.set_xlim([-23,29.54])
ax.set_ylim([1.6,3.25])
ax.legend(fancybox=True, shadow=True,fontsize=12,prop={'size': 20})
plt.tight_layout()

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('rungekutta RoNe dia.pdf')

#%%

# fig, ax = plt.subplots(1,1,figsize=(14,7),sharex=True)

# # ax[0].errorbar(time_pressure_RoNe,pressure_RoNe,error_pressure_RoNe)
# ax.plot(time_sound_RoNe,yhat)
# ax.plot(time_rate_RoNe,rate_simulado, linewidth=5, label = 'Modelo')
# # ax[0].errorbar(time_pressure_RoVio,pressure_RoVio,error_pressure_RoVio)
# # ax[0].errorbar(time_pressure_NaRo,pressure_NaRo,error_pressure_NaRo)



# # ax[0].errorbar(time_rate_RoNe,rate_RoNe,error_rate_RoNe)
# ax.plot(time_pressure_RoNe,presion_simulado, linewidth=5, label = 'Modelo')
# # ax[1].errorbar(time_rate_RoVio,rate_RoVio,error_rate_RoVio)
# # ax[1].errorbar(time_rate_NaRo,rate_NaRo,error_rate_NaRo)

