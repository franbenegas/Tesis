# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:34:55 2025

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

#%%
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

###################Calculo del rate previo al avion##############################
indice_0 = np.argmin(abs(time_rate_RoNe - 0))

w0_RoNe = np.mean(rate_RoNe[:indice_0])
std_w0_RoNe = np.std(rate_RoNe[:indice_0])

########################Calculo de la longitud del intervalo del paso del avion########################

index_at_0 = (np.abs(time_sound_RoNe - 0)).argmin()
# Get the value of the sound at time 0
value_at_0 = sound_RoNe[index_at_0]

# Exclude the point at index 0 and find the closest index to value_at_0
excluded_indices = np.arange(len(sound_RoNe)) != index_at_0
closest_index_not_0 = (np.abs(sound_RoNe[excluded_indices] - value_at_0)).argmin()

# Since we excluded indices, convert back to the original index space
actual_closest_index = np.arange(len(sound_RoNe))[excluded_indices][closest_index_not_0]

# Get the time at that index
time_at_value = time_sound_RoNe[actual_closest_index]

############ calculo del tiempo maximo del avion ##############################

n=5
top_n_indices = np.argpartition(sound_RoNe, -n)[-n:]  # Indices of the top 'n' values

# Step 2: Get the corresponding times and values of the top 'n' points
time_partition = time_sound_RoNe[top_n_indices]     # Times of the top 'n' points
partition = sound_RoNe[top_n_indices] 
median_top_n_sonido = np.median(partition)
median_top_n_time = np.median(time_partition)

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

indice_0 = np.argmin(abs(time_rate_RoVio - 0))

w0_RoVio = np.mean(rate_RoVio[:indice_0])
std_w0_RoVio = np.std(rate_RoVio[:indice_0])
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

indice_0 = np.argmin(abs(time_rate_NaRo - 0))

w0_NaRo = np.mean(rate_NaRo[:indice_0])
std_w0_NaRo = np.std(rate_NaRo[:indice_0])
#%%
colors = ['#0E2862','#2F4F4F','#152534']
colors2=['#890304']
plt.close()

fig, ax = plt.subplots(3,1,figsize=(15,9),sharex=True)
ax[0].set_ylabel("Audio (u. a.)", fontsize=20)
ax[1].set_ylabel("Presion (u. a.)", fontsize=20)
ax[2].set_ylabel("Rate (Hz)", fontsize=20)
ax[2].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[2].tick_params(axis='both', labelsize=10)


# Plot the data and extract the first element for the legend
ax[0].errorbar(time_sound_RoNe, sound_RoNe, error_s_RoNe,color='royalblue')
ax[0].errorbar(time_sound_RoVio, sound_RoVio, error_s_RoVio,color=colors[1])
ax[0].errorbar(time_sound_NaRo, sound_NaRo, error_s_NaRo,color=colors2[0])
ax[0].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--')
ax[0].axvspan(time_partition[-1], time_partition[0], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='-')
# ax[0].axvline(median_top_n_time, color='k', linestyle='--', zorder=50)
ax[0].text(20, 4.5, f"{int(round(time_at_value, 0))} s", fontsize=12, bbox=dict(facecolor='#B35F9F', alpha=0.3))


ax[1].errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe,color='royalblue')
ax[1].errorbar(time_pressure_RoVio, pressure_RoVio, error_pressure_RoVio,color=colors[1])
ax[1].errorbar(time_pressure_NaRo, pressure_NaRo, error_pressure_NaRo,color=colors2[0])
ax[1].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--')
ax[1].axvspan(time_partition[-1], time_partition[0], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='-')
# ax[1].axvline(median_top_n_time, color='k', linestyle='--')

ax[2].errorbar(time_rate_RoNe, rate_RoNe/w0_RoNe, error_rate_RoNe,color='royalblue')
ax[2].errorbar(time_rate_RoVio, rate_RoVio/w0_RoVio, error_rate_RoVio,color=colors[1])
ax[2].errorbar(time_rate_NaRo, rate_NaRo/w0_NaRo, error_rate_NaRo,color=colors2[0])
ax[2].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--')
ax[2].axvspan(time_partition[-1], time_partition[0], facecolor='gray', alpha=0.5, edgecolor='k', linestyle='-')
# ax[2].axvline(median_top_n_time, color='k', linestyle='--')

# Add the legend above the plot
fig.legend(["Estimulo", "Maximo estimulo", "RoNe", "RoVio", "NaRo"], loc='outside upper center', ncol=5, fontsize=12,mode='expand',prop={'size': 15})

plt.tight_layout(h_pad=5, rect=[0, 0, 1, 0.96]) 
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Graficos defensa'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('3 promedios sin normalizar.png')
#%%



# RoNe-NaRo
# Find the index in time_maximos_RoNe closest to time_at_value (for pressure)
index_at_value_presion_RoNe = (np.abs(time_pressure_RoNe - time_at_value)).argmin()
value_at_time_presion_RoNe = pressure_RoNe[index_at_value_presion_RoNe]

index_at_value_presion_0_RoNe = (np.abs(time_pressure_RoNe - 0)).argmin()
value_at_time_presion_0_RoNe = pressure_RoNe[index_at_value_presion_0_RoNe]

# Find the index in time_rate_RoNe closest to time_at_value (for rate)
index_at_value_rate_RoNe = (np.abs(time_rate_RoNe - time_at_value)).argmin()
value_at_time_rate_RoNe = rate_RoNe[index_at_value_rate_RoNe]

index_at_value_rate_0_RoNe = (np.abs(time_rate_RoNe - 0)).argmin()
value_at_time_rate_0_RoNe = rate_RoNe[index_at_value_rate_0_RoNe]

# RoVio-NaRo
# Find the index in time_pressure_RoVio closest to time_at_value (for pressure)
index_at_value_presion_RoVio = (np.abs(time_pressure_RoVio - time_at_value)).argmin()
value_at_time_presion_RoVio = pressure_RoVio[index_at_value_presion_RoVio]

index_at_value_presion_0_RoVio = (np.abs(time_pressure_RoVio - 0)).argmin()
value_at_time_presion_0_RoVio = pressure_RoVio[index_at_value_presion_0_RoVio]

# Find the index in time_rate_RoVio closest to time_at_value (for rate)
index_at_value_rate_RoVio = (np.abs(time_rate_RoVio - time_at_value)).argmin()
value_at_time_rate_RoVio = rate_RoVio[index_at_value_rate_RoVio]

index_at_value_rate_0_RoVio = (np.abs(time_rate_RoVio - 0)).argmin()
value_at_time_rate_0_RoVio = rate_RoVio[index_at_value_rate_0_RoVio]

# NaRo
# Find the index in time_pressure_NaRo closest to time_at_value (for pressure)
index_at_value_presion_NaRo = (np.abs(time_pressure_NaRo - time_at_value)).argmin()
value_at_time_presion_NaRo = pressure_NaRo[index_at_value_presion_NaRo]

index_at_value_presion_0_NaRo = (np.abs(time_pressure_NaRo - 0)).argmin()
value_at_time_presion_0_NaRo = pressure_NaRo[index_at_value_presion_0_NaRo]

# Find the index in time_rate_NaRo closest to time_at_value (for rate)
index_at_value_rate_NaRo = (np.abs(time_rate_NaRo - time_at_value)).argmin()
value_at_time_rate_NaRo = rate_NaRo[index_at_value_rate_NaRo]

index_at_value_rate_0_NaRo = (np.abs(time_rate_NaRo - 0)).argmin()
value_at_time_rate_0_NaRo = rate_NaRo[index_at_value_rate_0_NaRo]


#%%
# Calculate errors at specific points (value_at_time and value_at_time_0)
error_at_value_presion_RoNe = error_pressure_RoNe[index_at_value_presion_RoNe]
error_at_value_presion_0_RoNe = error_pressure_RoNe[index_at_value_presion_0_RoNe]

error_at_value_rate_RoNe = error_rate_RoNe[index_at_value_rate_RoNe]
error_at_value_rate_0_RoNe =  error_rate_RoNe[index_at_value_rate_0_RoNe]

# Repeat for RoVio
error_at_value_presion_RoVio = error_pressure_RoVio[index_at_value_presion_RoVio]
error_at_value_presion_0_RoVio = error_pressure_RoVio[index_at_value_presion_0_RoVio]

error_at_value_rate_RoVio = error_rate_RoVio[index_at_value_rate_RoVio]
error_at_value_rate_0_RoVio = error_rate_RoVio[index_at_value_rate_0_RoVio]

# Repeat for NaRo
error_at_value_presion_NaRo = error_pressure_NaRo[index_at_value_presion_NaRo]
error_at_value_presion_0_NaRo = error_pressure_NaRo[index_at_value_presion_0_NaRo]

error_at_value_rate_NaRo = error_rate_NaRo[index_at_value_rate_NaRo]
error_at_value_rate_0_NaRo = error_rate_NaRo[index_at_value_rate_0_NaRo]

#%%
value_at_time_presion_0_RoNe=1.1

species = ("RoNe", "RoVio", "NaRo")

# Calculate percentage change for pressure and rate

pressure_values = [
    round(100 * (value_at_time_presion_RoNe - value_at_time_presion_0_RoNe) / value_at_time_presion_0_RoNe, 2),
    round(100 * (value_at_time_presion_RoVio - value_at_time_presion_0_RoVio) / value_at_time_presion_0_RoVio, 2),
    round(100 * (value_at_time_presion_NaRo - value_at_time_presion_0_NaRo) / value_at_time_presion_0_NaRo, 2)
]



rate_values = [
    round(100 * (value_at_time_rate_RoNe - value_at_time_rate_0_RoNe) / value_at_time_rate_0_RoNe, 2),
    round(100 * (value_at_time_rate_RoVio - value_at_time_rate_0_RoVio) / value_at_time_rate_0_RoVio, 2),
    abs(round(100 * (value_at_time_rate_NaRo - value_at_time_rate_0_NaRo) / value_at_time_rate_0_NaRo, 2))
]

penguin_means = {
    'Pressure': pressure_values,
    'Rate': rate_values
}

# # Calculate errors using the previous and next indices, then convert to percentage
# pressure_errors = [
#     round(100 * (np.abs(pressure_RoNe[index_at_value_presion_RoNe + 1] - pressure_RoNe[index_at_value_presion_RoNe - 1])) / 2*value_at_time_presion_0_RoNe, 2),
#     round(100 * (np.abs(pressure_RoVio[index_at_value_presion_RoVio + 1] - pressure_RoVio[index_at_value_presion_RoVio - 1])) / 2*value_at_time_presion_0_RoVio, 2),
#     round(100 * (np.abs(pressure_NaRo[index_at_value_presion_NaRo + 1] - pressure_NaRo[index_at_value_presion_NaRo - 1])) / 2*value_at_time_presion_0_NaRo, 2)
# ]


# rate_errors = [
#     round(100 * (np.abs(rate_RoNe[index_at_value_rate_RoNe + 1] - rate_RoNe[index_at_value_rate_RoNe - 1])) /2* value_at_time_presion_0_RoNe, 2),
#     round(100 * (np.abs(rate_RoVio[index_at_value_rate_RoVio + 1] - rate_RoVio[index_at_value_rate_RoVio - 1])) / 2*value_at_time_presion_0_RoVio, 2),
#     round(100 * (np.abs(rate_NaRo[index_at_value_rate_NaRo + 1] - rate_NaRo[index_at_value_rate_NaRo - 1])) / 2*value_at_time_presion_0_NaRo, 2)
# ]


# Compute the propagated error for the percentage change
pressure_errors = [
    round(100 * np.sqrt(
        (error_at_value_presion_RoNe / value_at_time_presion_0_RoNe) ** 2 +
        (error_at_value_presion_0_RoNe * value_at_time_presion_RoNe / value_at_time_presion_0_RoNe ** 2) ** 2
    ), 2),

    round(100 * np.sqrt(
        (error_at_value_presion_RoVio / value_at_time_presion_0_RoVio) ** 2 +
        (error_at_value_presion_0_RoVio * value_at_time_presion_RoVio / value_at_time_presion_0_RoVio ** 2) ** 2
    ), 2),

    round(100 * np.sqrt(
        (error_at_value_presion_NaRo / value_at_time_presion_0_NaRo) ** 2 +
        (error_at_value_presion_0_NaRo * value_at_time_presion_NaRo / value_at_time_presion_0_NaRo ** 2) ** 2
    ), 2)
]

rate_errors = [
    round(100 * np.sqrt(
        (error_at_value_rate_RoNe / value_at_time_rate_0_RoNe) ** 2 +
        (error_at_value_rate_0_RoNe * value_at_time_rate_RoNe / value_at_time_rate_0_RoNe ** 2) ** 2
    ), 2),

    round(100 * np.sqrt(
        (error_at_value_rate_RoVio / value_at_time_rate_0_RoVio) ** 2 +
        (error_at_value_rate_0_RoVio * value_at_time_rate_RoVio / value_at_time_rate_0_RoVio ** 2) ** 2
    ), 2),

    round(100 * np.sqrt(
        (error_at_value_rate_NaRo / value_at_time_rate_0_NaRo) ** 2 +
        (error_at_value_rate_0_NaRo * value_at_time_rate_NaRo / value_at_time_rate_0_NaRo ** 2) ** 2
    ), 2)
]


penguin_errors = {
    'Pressure': pressure_errors,
    'Rate': rate_errors
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0


# Define colors for each category
colors = {'Pressure': 'royalblue', 'Rate': '#2F4F4F'}

# plt.close()
fig, ax = plt.subplots(figsize=(15, 9), layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    errors = penguin_errors[attribute]  # Get the corresponding error values
    rects = ax.bar(x + offset, measurement, width, label=attribute, yerr=errors, capsize=5, color=colors[attribute])
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_ylabel('Percentaje respecto al basal (%)', fontsize=15)
ax.set_xticks(x + width / 2, species, fontsize=15)
ax.legend(fancybox=True, shadow=True, loc='upper left', ncols=2, fontsize=12)
ax.set_ylim(0, 80)

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('porcentaje despues del avion.pdf')
#%%

fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)

ax[0].set_ylabel("Presion (u. a.)", fontsize=20)
ax[1].set_ylabel("Rate (Hz)", fontsize=20)
ax[1].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)


ax[0].errorbar(time_pressure_RoNe, pressure_RoNe, error_pressure_RoNe,color='royalblue',label='RoNe')
ax[0].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--',label='Avion')
ax[0].scatter(time_pressure_RoNe[index_at_value_presion_RoNe],value_at_time_presion_0_RoNe,color='k',marker='_',zorder=50)
ax[0].scatter(time_pressure_RoNe[index_at_value_presion_RoNe],value_at_time_presion_RoNe,color='k',s=50,zorder=50,label='Valor despues del paso del avion')
ax[0].hlines(y=value_at_time_presion_0_RoNe, xmin=time_pressure_RoNe[index_at_value_presion_0_RoNe], xmax=time_pressure_RoNe[index_at_value_presion_RoNe],color='k', linestyle='--')
ax[0].vlines(time_pressure_RoNe[index_at_value_presion_RoNe],ymin=value_at_time_presion_0_RoNe,ymax=value_at_time_presion_RoNe,color='k')



ax[1].errorbar(time_rate_RoNe, rate_RoNe, error_rate_RoNe,color='royalblue')
ax[1].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--')
ax[1].scatter(time_rate_RoNe[index_at_value_rate_RoNe],value_at_time_rate_0_RoNe,color='k',marker='_',zorder=50)
ax[1].scatter(time_rate_RoNe[index_at_value_rate_RoNe],value_at_time_rate_RoNe,color='k',s=50,zorder=50)
ax[1].hlines(y=value_at_time_rate_0_RoNe, xmin=time_rate_RoNe[index_at_value_rate_0_RoNe], xmax=time_rate_RoNe[index_at_value_rate_RoNe],color='k', linestyle='--')
ax[1].vlines(time_rate_RoNe[index_at_value_rate_RoNe],ymin=value_at_time_rate_0_RoNe,ymax=value_at_time_rate_RoNe,color='k')


ax[0].legend(fancybox=True,shadow=True,loc='upper left', fontsize=12,prop={'size': 20})
plt.tight_layout()
# plt.savefig('esquema calculo porcentaje.pdf')

#%%
cocientes = np.load(r"C:\Users\beneg\Downloads\cocientes.npy")

fig, ax = plt.subplots(1,1,figsize=(15,9))
ax.set_xlabel(r'$\mathcal{A}_{bajas}/\mathcal{A}_{altas}$',fontsize=20)
ax.set_ylabel('Cantidad de archivos',fontsize=20)
ax.tick_params(axis='both', labelsize=10)
ax.set_xlim(0,35)

ax.hist(cocientes,100,color='royalblue')
ax.axvline(10, c='k', zorder=10,linewidth=5,linestyle='--',label='Umbral = 10')

ax.legend(fancybox=True,shadow=True,fontsize=12,prop={'size': 20})
plt.tight_layout()

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('cocientes.pdf')
#%% "Ajuste exponencial"

colors = ['#0E2862','#2F4F4F','#152534']
colors2=['#890304']

def f(t, a, b, t0, c):
    return a * np.exp( b * (t-t0)) + c


fig, ax = plt.subplots(1,1,figsize=(15,9))
ax.set_xlabel('Tiempo (s)',fontsize=20)
ax.set_ylabel('Rate (Hz)',fontsize=20)
ax.tick_params(axis='both', labelsize=10)
ax.set_ylim([1,3])

ax.errorbar(time_rate_RoNe, rate_RoNe, yerr=error_rate_RoNe,color='royalblue')
ax.plot(time_rate_RoNe, f(time_rate_RoNe, 1, .75, 3, 1.25),linewidth=5,color=colors[1],label=r'$b=0.75$')
ax.plot(time_rate_RoNe, f(time_rate_RoNe, 1, -.05, 3, 1.5),linewidth=5,color=colors2[0],label=r'$b=-0.05$')

# # phase field equations
# eq2 = (
#        r"$\mathcal{F} = a e^{b(t-t_0)} + c$")
# ax.text(-40,2, eq2, color="k", fontsize=24)

ax.legend(title=r"$\mathcal{F}(t) = a e^{b(t-t_0)} + c$",title_fontsize=20,loc='upper left',fontsize=12,prop={'size': 24})


fig.tight_layout()

# plt.savefig('ajuste exponencial.pdf')
#%%
from scipy.io import wavfile
from scipy import signal
import parselmouth

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory) 

Sound = "sound_CaFF028-RoNe_2023_01_30-22.18.17.wav"
# fs, audio = wavfile.read(Sound)


# # Generate a spectrogram of the filtered signal
# f, t, Sxx = signal.spectrogram(audio, fs)

# # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
# frequency_cutoff = 1000  # Frequency cutoff in Hz
# threshold = -10 #10.5 para NaRo  # Threshold for spectrogram in dB
# Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale

# # Find where frequencies exceed the threshold
# freq_indices = np.where(f > frequency_cutoff)[0]
# time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)

# time_above_threshold = t[time_indices]
# longest_interval = find_longest_interval(time_above_threshold)




# plt.tight_layout()
plt.close('all')
sound = parselmouth.Sound(Sound)
x, y = sound.xs(), sound.values.T

def draw_spectrogram(ax, spectrogram, fig, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    # sg_db = 10 * np.log10(spectrogram.values)
    sg_db = np.log(spectrogram.values)
    ax.clear()

    # Crear el pcolormesh
    mesh = ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max()-10, cmap='viridis')

    # Añadir colorbar horizontal más delgada con padding ajustado
    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.15, shrink=1, aspect=50)
    cbar.set_label(r'$\log(\text{A})$', fontsize=15)

    # Etiquetas
    ax.set_ylim([spectrogram.ymin, spectrogram.ymax])
    ax.set_xlim([spectrogram.xmin, spectrogram.xmax])
    ax.set_xlabel("Tiempo (s)", fontsize=15)
    ax.set_ylabel("Frecuencia (Hz)", fontsize=15)

    return mesh

fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

ax[0].set_ylabel("Sonido (u. a.)", fontsize=15)

ax[0].tick_params(axis='both', labelsize=13)
ax[1].tick_params(axis='both', labelsize=13)
l, = ax[0].plot(x, y, color='royalblue')

spectrogram = sound.to_spectrogram()
j = draw_spectrogram(ax[1], spectrogram, fig)

plt.tight_layout()

# plt.savefig('espectograma.png')

#%%
Pressure = "pressure_CaFF028-RoNe_2023_01_30-22.18.17.wav"
Sound_2 = "sound_CaFF028-RoNe_2023_01_28-20.13.35.wav"
fs, audio = wavfile.read(Sound)
fs, pressure = wavfile.read(Pressure)
fs, audio_2 = wavfile.read(Sound_2)
time = np.linspace(0, len(pressure) / fs, len(pressure))

# Normalize the audio and pressure data
audio = audio - np.mean(audio)  # Remove mean
audio_norm = audio / np.max(audio)  # Normalize by max value

audio_2 = audio_2 - np.mean(audio_2)
audio_norm_2 = audio_2/ np.max(audio_2)

# --- Normalizar en un intervalo [0, 1] segundos ---
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

audio_norm = norm11_interval(audio_norm, 0, 1, 44150)
audio_norm_2 = norm11_interval(audio_norm_2, 0, 1, 44150)

fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)
ax[0].set_ylabel("Sonido (u. a.)", fontsize=15)
ax[1].set_ylabel("Sonido (u. a.)", fontsize=15)
ax[1].set_xlabel("tiempo (s)", fontsize=15)
ax[0].tick_params(axis='both', labelsize=13)
ax[1].tick_params(axis='both', labelsize=13)

ax[0].plot(time-27.46, audio_norm_2,color='royalblue',label='RoNe_2023_01_28-20.13.35')
ax[1].plot(time-9.69, audio_norm,color='royalblue',label='RoNe_2023_01_30-22.18.17')
ax[0].legend(fancybox=True,shadow=True,loc='upper right',fontsize=12,prop={'size': 15})
ax[1].legend(fancybox=True,shadow=True,loc='upper right',fontsize=12,prop={'size': 15})
# ax[0].axvline(9.69,color='k',linewidth=3, linestyle='-')
# ax[1].axvline(0,color='k',linewidth=3, linestyle='-')
plt.tight_layout()


#%%
from scipy.signal import find_peaks


Pressure_2 = "pressure_CaFF028-RoNe_2023_01_28-20.13.35.wav"

fs, pressure_2 = wavfile.read(Pressure_2)

pressure_2 = pressure_2 - np.mean(pressure_2)
pressure_norm_2 = pressure_2/np.max(pressure_2)

pressure_norm_2 = norm11_interval(pressure_norm_2, 0, 1, 44150)

peaks_maximos, _ = find_peaks(pressure_norm_2, prominence=1, height=0, distance=int(fs * 0.1))

fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)

ax[0].set_ylabel("Sonido (u. a.)", fontsize=15)
ax[1].set_ylabel("Presion (u. a.)", fontsize=15)
ax[1].set_xlabel("tiempo (s)", fontsize=15)
ax[0].tick_params(axis='both', labelsize=13)
ax[1].tick_params(axis='both', labelsize=13)

ax[0].plot(time, audio_norm_2,color='royalblue')
ax[1].plot(time,pressure_norm_2,color='royalblue')

plt.xlim([25,45])
ax[0].set_ylim([-11,11])
ax[1].set_ylim([-3,3.5])

plt.tight_layout()
#%%

fs, audio = wavfile.read(Sound)

# Normalize the audio and pressure data
audio = audio - np.mean(audio)  # Remove mean
audio_norm = audio / np.max(audio)  # Normalize by max value

# --- Normalizar en un intervalo [0, 1] segundos ---
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

audio_norm = norm11_interval(audio_norm, 0, 1, 44150)

# --- Encontrar el intervalo más largo ---
def find_longest_interval(times, max_gap=0.1):
    if len(times) == 0:
        return []
    longest_interval = []
    current_interval = [times[0]]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] <= max_gap:
            current_interval.append(times[i])
        else:
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]
    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval
    return longest_interval

# Generate a spectrogram of the filtered signal
f, t, Sxx = signal.spectrogram(audio_norm, fs)

# Spectrogram processing to find the longest interval where the frequency exceeds a threshold
frequency_cutoff = 1000  # Frequency cutoff in Hz
threshold = -11 #10.5 para NaRo  # Threshold for spectrogram in dB
Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale

# Find where frequencies exceed the threshold
freq_indices = np.where(f > frequency_cutoff)[0]
time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
time_above_threshold = t[time_indices]
longest_interval = find_longest_interval(time_above_threshold)



longest_interval = find_longest_interval(time_above_threshold)

def draw_spectrogram_from_Sxx(ax, f, t, Sxx_dB, fig):
    # Sxx_dB = 10*Sxx_dB
    mesh = ax.pcolormesh(t, f, Sxx_dB, cmap='viridis', vmin=Sxx_dB.max() - 10)

    cbar = fig.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.15, shrink=1, aspect=50)
    cbar.set_label(r'$\log(\text{A})$', fontsize=15)
    ax.set_ylim([0, 5000])
    ax.set_xlim([0, 60])
    ax.set_xlabel("Tiempo (s)", fontsize=15)
    ax.set_ylabel("Frecuencia (Hz)", fontsize=15)
    return mesh


fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

x = np.linspace(0, len(audio_norm) / fs, len(audio_norm))  # <-- Definir x

ax[0].set_ylabel("Sonido (u. a.)", fontsize=15)
ax[0].tick_params(axis='both', labelsize=13)
ax[1].tick_params(axis='both', labelsize=13)

ax[0].plot(x, audio_norm, color='royalblue')
ax[1].axhline(frequency_cutoff, color='k',linewidth=3, linestyle='--',label='Frecuencia de corte')
if len(longest_interval) > 0:
    ax[0].axvline(longest_interval[0], color='k',linewidth=3, linestyle='-',label='Inicio del avion')
    ax[1].axvline(longest_interval[0], color='k',linewidth=3, linestyle='-',label='Inicio del avion')

draw_spectrogram_from_Sxx(ax[1], f, t, Sxx_dB, fig)
ax[1].legend(fancybox=True,shadow=True,loc='upper right',fontsize=12,prop={'size': 15})
plt.tight_layout()

# plt.savefig('corte_avion.png')

#%%
from scipy.signal import find_peaks

pressure = "pressure_CaFF028-RoNe_2023_01_30-22.18.17.wav"

fs, pressure = wavfile.read(pressure)

time = np.linspace(0, len(pressure) / fs, len(pressure))

# Normalize the audio and pressure data
pressure = pressure - np.mean(pressure)  # Remove mean
pressure_norm = pressure / np.max(pressure)  # Normalize by max value

# --- Normalizar en un intervalo [0, 1] segundos ---
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

pressure_norm = norm11_interval(pressure_norm, 0, 1, fs)

peaks_maximos, _ = find_peaks(pressure_norm, prominence=1, height=0, distance=int(fs * 0.1))

lugar_maximos = []  # Read manually identified maxima from a text file
maximos = np.loadtxt('CaFF028-RoNe_2023_01_30-22.18.17_maximos.txt')
for i in maximos:
    lugar_maximos.append(int(i))
    
fig, ax = plt.subplots(1,1,figsize=(15,9))
ax.set_xlabel("Tiempo (s)", fontsize=20)
ax.set_ylabel("Presion (u. a.)", fontsize=20)
#color rojo lindo #890304
ax.plot(time,pressure_norm, color='royalblue',label='Presion')
ax.plot(time[lugar_maximos],pressure_norm[lugar_maximos],'.',ms=20,color='darkslategray',label='Picos seleccionados')
# ax.plot(time[peaks_maximos],pressure_norm[peaks_maximos],'.',ms=15,color='C1',label='Picos encontrados')
# ax.set_xlim(0.7,1.7)
# ax.set_xlim([12.8,13.8])
# ax.set_ylim([-1.5,1.5])
ax.legend(fancybox=True,shadow=True,fontsize=12,prop={'size': 15})

plt.tight_layout()
# plt.savefig('picos_anomalos.pdf')

#%% maximos y minimos

from scipy.signal import find_peaks

pressure = "pressure_CaFF028-RoNe_2023_01_30-22.18.17.wav"
# pressure = "pressure_CaFF028-RoNe_2023_01_28-20.13.35.wav"

fs, pressure = wavfile.read(pressure)

time = np.linspace(0, len(pressure) / fs, len(pressure))

# Normalize the audio and pressure data
pressure = pressure - np.mean(pressure)  # Remove mean
pressure_norm = pressure / np.max(pressure)  # Normalize by max value

# --- Normalizar en un intervalo [0, 1] segundos ---
def norm11_interval(x, ti, tf, fs):
    x_int = x[int(ti * fs):int(tf * fs)]
    return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

# Function to filter maxima: removes maxima without a minimum in between and keeps only the highest max between minima
def filter_maxima(peaks_max, peaks_min, pressure):
    filtered_maxima = []
    for i in range(1, len(peaks_min)):
        # Find maxima between consecutive minima
        max_between_min = [p for p in peaks_max if peaks_min[i-1] < p < peaks_min[i]]
        
        if max_between_min:
            # Keep the highest max within the minima range
            highest_max = max(max_between_min, key=lambda p: pressure[p])
            filtered_maxima.append(highest_max)
    
    return filtered_maxima

pressure_norm = norm11_interval(pressure_norm, 0, 1, fs)

peaks_maximos, _ = find_peaks(pressure_norm, prominence=1, height=0, distance=int(fs * 0.1))
peaks_minimos,_ = find_peaks(-pressure_norm, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure_norm)

lugar_maximos = []  # Read manually identified maxima from a text file
maximos = np.loadtxt('CaFF028-RoNe_2023_01_30-22.18.17_maximos.txt')
# maximos = np.loadtxt('CaFF028-RoNe_2023_01_28-20.13.35_maximos.txt')
for i in maximos:
    lugar_maximos.append(int(i))
    
fig, ax = plt.subplots(1,1,figsize=(15,9))
ax.set_xlabel("Tiempo (s)", fontsize=20)
ax.set_ylabel("Presion (u. a.)", fontsize=20)
#color rojo lindo #890304
ax.plot(time,pressure_norm, color='royalblue',label='Presion')
ax.plot(time[lugar_maximos],pressure_norm[lugar_maximos],'X',ms=20,color='darkslategray',label='Picos seleccionados')
ax.plot(time[peaks_maximos],pressure_norm[peaks_maximos],'.',ms=15,color='C1',label='Picos encontrados')
ax.plot(time[peaks_minimos],pressure_norm[peaks_minimos],'.k',ms=15,label='Minimos')
ax.plot(time[peaks_maximos_filtered],pressure_norm[peaks_maximos_filtered],'.C3',ms=15,label='Picos con algoritmo')
# ax.set_xlim([47.6,48.6])
# ax.set_ylim([-2,2.3])
ax.legend(fancybox=True,shadow=True,fontsize=12,prop={'size': 15},loc='upper right')

plt.tight_layout()
# plt.savefig('minimos.pdf')