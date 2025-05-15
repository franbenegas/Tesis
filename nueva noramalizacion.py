# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 17:18:55 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import pickle
from scipy.signal import find_peaks
from scipy.signal import butter, sosfiltfilt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
import json
# Set the font to 'STIX'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['mathtext.fontset'] = 'stix'


def process_night_data(directories):
    """
    Process data from multiple directories and return a combined time series list.

    Args:
        directories (list of str): List of directory paths.

    Returns:
        list of dict: Combined time series data from all directories.
    """
    def datos_normalizados_nuevo(Sonidos,Presiones,indice):
        
        def normalizar(x, mean, max):
          return (x / np.max(np.abs(x))) * max + mean
      
        Sound, Pressure = Sonidos[indice], Presiones[indice]

        name = Pressure[9:-4]
        
        fs,audio = wavfile.read(Sound)
        fs,pressure = wavfile.read(Pressure)
        
        data_norm = np.loadtxt(name + '.txt',delimiter=',',skiprows=1)
        
        mean_s, max_s = data_norm[0], data_norm[2]
        mean_p, max_p = data_norm[1], data_norm[3]
        
        # time = np.arange(0, len(p_wav)/fs, 1/fs)
        p_wav = np.array(pressure, dtype=np.float64)
        s_wav = np.array(audio, dtype=np.float64)
        
        p_norm = normalizar(p_wav, mean_p, max_p)
        s_norm = normalizar(s_wav, mean_s, max_s)
        
        return s_norm, p_norm, name, fs
    
    # Function to design a Butterworth bandpass filter
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist  # Normalize lowcut frequency
        high = highcut / nyquist  # Normalize highcut frequency
        sos = butter(order, [low, high], btype='band', output='sos')  # Generate filter coefficients
        return sos

    # Apply the Butterworth bandpass filter to the data
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        filtered_signal = sosfiltfilt(sos, data)  # Filter the data
        return filtered_signal
    
    # Define filter parameters
    lowcut = 0.01  # Low cutoff frequency in Hz
    highcut = 1500  # High cutoff frequency in Hz
    fs = 44150  # Sampling frequency in Hz
    order = 6  # Filter order
        
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
            elif file[-4:] == 'json':
                datos.append(file)
            

        # Load the first JSON file
        with open(datos[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        for indice in range(len(datos)):
            tiempo_inicial = datos[indice]['Tiempo inicial avion']
            
            audio, pressure, name, fs = datos_normalizados_nuevo(sonidos, presiones, indice)
            
            time = np.linspace(0, len(pressure)/fs, len(pressure))
            
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            # peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), prominence=.001)
            
            lugar_maximos = []
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))

            periodo = np.diff(time[lugar_maximos])
            tiempo = time - tiempo_inicial
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time rate': tiempo[lugar_maximos][1:],
                'rate': 1/periodo
            })

    return combined_time_series_list


def interpolate_single_data(time_series_list, data_key, time_key, common_time_length):
    """
    Interpolates a specific data series from the time series list.

    Args:
        time_series_list (list of dict): Combined time series data from all directories.
        data_key (str): The key to access the data to be interpolated (e.g., 'sonido', 'rate', 'presion').
        time_key (str): The key to access the corresponding time series (e.g., 'time', 'time rate', 'time maximos').
        common_time_length (int): Length of the common time base for interpolation.

    Returns:
        common_time_base (ndarray): The interpolated common time base.
        interpolated_data (ndarray): The interpolated data for the specified key.
    """
    # Calculate the common time base
    start_time = min(ts[time_key][0] for ts in time_series_list)
    end_time = max(ts[time_key][-1] for ts in time_series_list)
    common_time_base = np.linspace(start_time, end_time, common_time_length)
    
    # Interpolate the data onto the common time base
    interpolated_data = []
    for ts in time_series_list:
        interp_func = interp1d(ts[time_key], ts[data_key], bounds_error=False, fill_value=np.nan)
        interpolated_data.append(interp_func(common_time_base))
    
    interpolated_data = np.array(interpolated_data)
    
    return common_time_base, interpolated_data

def compute_average_and_std(data):
    average = np.nanmean(data, axis=0)
    count_non_nan = np.sum(~np.isnan(data), axis=0)
    std_error = np.nanstd(data, axis=0) / np.sqrt(count_non_nan)
    return average, std_error, count_non_nan

    
#%%

# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoNe' folder

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)

# Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
pajaritos = '\Aviones\Aviones y pajaros' #Agregar V2 para NaRo
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder

# Store all directories in a list
directories = [noches_1,noches_2, noches_3]

# Process the night data from the directories using the process_night_data function
RoNe_noche = process_night_data(directories)


#%%

# # Set up subplots
# fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
# ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
# ax[2].set_ylabel("Rate (Hz)", fontsize=14)
# ax[2].set_xlabel("Time (s)", fontsize=14)
# plt.tight_layout()

# # Choose a colormap
# cmap = get_cmap('plasma')  # You can choose any other colormap, e.g., 'plasma', 'inferno'
# norm_audio = Normalize(vmin=min([np.min(RoNe_noche[i]['sonido']) for i in range(len(RoNe_noche))]),
#                        vmax=max([np.max(RoNe_noche[i]['sonido']) for i in range(len(RoNe_noche))]))
# norm_pressure = Normalize(vmin=min([np.min(RoNe_noche[i]['presion']) for i in range(len(RoNe_noche))]),
#                           vmax=max([np.max(RoNe_noche[i]['presion']) for i in range(len(RoNe_noche))]))
# norm_rate = Normalize(vmin=min([np.min(RoNe_noche[i]['rate']) for i in range(len(RoNe_noche))]),
#                       vmax=max([np.max(RoNe_noche[i]['rate']) for i in range(len(RoNe_noche))]))

# # Plot all the data
# for i in range(len(RoNe_noche)):
#     # Audio plot
#     tiempo_sonido, sonido = RoNe_noche[i]['time'], RoNe_noche[i]['sonido']
#     # ax[0].plot(tiempo_sonido, sonido, color='k', alpha=0.1, solid_capstyle='projecting')
    
#     # Highlight higher audio values
#     ax[0].scatter(tiempo_sonido, sonido, c=sonido, cmap=cmap, norm=norm_audio, s=10, alpha=0.8)

#     # Pressure plot
#     tiempo_presion, presion = RoNe_noche[i]['time maximos'], RoNe_noche[i]['presion']
#     # ax[1].plot(tiempo_presion, presion, color='k', alpha=0.1, solid_capstyle='projecting')
    
#     # Highlight higher pressure values
#     ax[1].scatter(tiempo_presion, presion, c=presion, cmap=cmap, norm=norm_pressure, s=10, alpha=0.8)

#     # Rate plot
#     tiempo_rate, rate = RoNe_noche[i]['time rate'], RoNe_noche[i]['rate']
#     # ax[2].plot(tiempo_rate, rate, color='k', alpha=0.1, solid_capstyle='projecting')
    
#     # Highlight higher rate values
#     ax[2].scatter(tiempo_rate, rate, c=rate, cmap=cmap, norm=norm_rate, s=10, alpha=0.8)

# # Optional: Add colorbars
# fig.colorbar(plt.cm.ScalarMappable(norm=norm_audio, cmap=cmap), ax=ax[0], label="Audio Intensity")
# fig.colorbar(plt.cm.ScalarMappable(norm=norm_pressure, cmap=cmap), ax=ax[1], label="Pressure Intensity")
# fig.colorbar(plt.cm.ScalarMappable(norm=norm_rate, cmap=cmap), ax=ax[2], label="Rate Intensity")
    
# #%% Grafico de sonido,pressure,rate con color en funcion del maximo del sonido

# # Set up subplots
# fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
# ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
# ax[2].set_ylabel("Rate (Hz)", fontsize=14)
# ax[2].set_xlabel("Time (s)", fontsize=14)
# plt.tight_layout()

# # Step 1: Calculate maximums of each 'sonido' list
# max_sonidos = [np.max(RoNe_noche[i]['sonido']) for i in range(len(RoNe_noche))]

# # Step 2: Normalize the maximum values
# norm = Normalize(vmin=min(max_sonidos), vmax=max(max_sonidos))

# # Step 3: Create a colormap based on the maximum values
# cmap = get_cmap('plasma')

# # Step 4: Plot each list of 'sonido' with the color corresponding to its maximum
# for i in range(len(RoNe_noche)):
#     tiempo_sonido, sonido = RoNe_noche[i]['time'], RoNe_noche[i]['sonido']
    
#     # Get the maximum value for this 'sonido' and normalize it
#     max_sonido = max_sonidos[i]
#     color = cmap(norm(max_sonido))  # Get color for this max value
    
#     # Plot the 'sonido' data using the color based on its maximum
#     ax[0].plot(tiempo_sonido, sonido, color=color, solid_capstyle='projecting', alpha=0.5)
    
#     # Optionally, plot corresponding pressure and rate with the same color
#     tiempo_presion, presion = RoNe_noche[i]['time maximos'], RoNe_noche[i]['presion']
#     tiempo_rate, rate = RoNe_noche[i]['time rate'], RoNe_noche[i]['rate']
    
#     ax[1].plot(tiempo_presion, presion, color=color, solid_capstyle='projecting', alpha=0.5)
#     ax[2].plot(tiempo_rate, rate, color=color, solid_capstyle='projecting', alpha=0.5)

# # Step 5: Add colorbar to reflect the maximums
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])  # We don't need to pass data, just the colormap and normalization
# fig.colorbar(sm, ax=ax[:], label="Max Sound Intensity")


# #%% aca hace lo que queria gabo


# # Step 1: Calculate max values for each sonido list
# max_sonidos = [np.max(RoNe_noche[i]['sonido']) for i in range(len(RoNe_noche))]

# # Step 2: Get the sorted indices of max_sonidos in ascending order
# sorted_indices = np.argsort(max_sonidos)

# # Step 3: Sort RoNe_noche lists based on the sorted_indices
# sorted_RoNe_noche = [RoNe_noche[i] for i in sorted_indices]

# # Step 4: Extract sorted lists of sonidos, pressure, and rate
# sorted_sonidos = [sorted_RoNe_noche[i]['sonido'] for i in range(len(sorted_RoNe_noche))]
# sorted_pressure = [sorted_RoNe_noche[i]['presion'] for i in range(len(sorted_RoNe_noche))]
# sorted_rate = [sorted_RoNe_noche[i]['rate'] for i in range(len(sorted_RoNe_noche))]

# # Step 5: Normalize based on max_sonidos
# norm = Normalize(vmin=min(max_sonidos), vmax=max(max_sonidos))

# # Step 6: Choose a colormap
# cmap = get_cmap('plasma')

# # Step 7: Set up subplots
# fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# fig.suptitle('RoVio em funcion de los maximos')
# ax[0].set_ylabel("Max Audio (arb. u.)", fontsize=14)
# ax[1].set_ylabel(" Max Pressure (arb. u.)", fontsize=14)
# ax[2].set_ylabel(" Max Rate (Hz)", fontsize=14)
# ax[2].set_xlabel("Index", fontsize=14)
# plt.tight_layout()

# # Step 8: Plot sorted data with colors based on max_sonidos
# for i in range(len(sorted_RoNe_noche)):
#     # Get the color based on the corresponding max_sonido value
#     color = cmap(norm(max_sonidos[sorted_indices[i]]))

#     # Plot sorted sonido values (max values) as scatter points
#     ax[0].scatter(i, max(sorted_sonidos[i]), color=color, s=10, alpha=0.8)

#     # Plot sorted pressure values (max values) as scatter points
#     ax[1].scatter(i, max(sorted_pressure[i]), color=color, s=10, alpha=0.8)

#     # Plot sorted rate values (max values) as scatter points
#     ax[2].scatter(i, max(sorted_rate[i]), color=color, s=10, alpha=0.8)

# # Step 9: Perform linear regression on sorted rate
# x_indices = np.arange(len(sorted_rate))
# y_values = [max(sorted_rate[i]) for i in range(len(sorted_rate))]

# # Fit a linear regression line
# coefficients, covariance_matrix = np.polyfit(x_indices, y_values, 1, cov=True)  # Linear fit with covariance
# linear_fit = np.polyval(coefficients, x_indices)  # Evaluate the polynomial

# # The errors are the square root of the diagonal elements of the covariance matrix
# errors = np.sqrt(np.diag(covariance_matrix))

# # Step 10: Plot the linear regression line
# ax[2].plot(x_indices, linear_fit, color='C0', linewidth=2, label=fr'Linear Fit = a:{round(coefficients[0],2)}$\pm${round(errors[0],2)}, b:{round(coefficients[1],2)}$\pm${round(errors[1],2)}')

# ## lo mismo para la amplitud de la presion

# # Step 9: Perform linear regression on sorted rate
# x_indices_p = np.arange(len(sorted_pressure))
# y_values_p = [max(sorted_pressure[i]) for i in range(len(sorted_pressure))]

# # Fit a linear regression line
# coefficients_p, covariance_matrix_p = np.polyfit(x_indices_p, y_values_p, 1, cov=True)  # Linear fit with covariance
# linear_fit_p = np.polyval(coefficients_p, x_indices_p)  # Evaluate the polynomial

# # The errors are the square root of the diagonal elements of the covariance matrix
# errors_p= np.sqrt(np.diag(covariance_matrix_p))

# # Step 10: Plot the linear regression line
# ax[1].plot(x_indices_p, linear_fit_p, color='C0', linewidth=2, label=fr'Linear Fit = a:{round(coefficients_p[0],2)}$\pm${round(errors_p[0],2)}, b:{round(coefficients_p[1],2)}$\pm${round(errors_p[1],2)}')


# # Optional: Add a colorbar based on max_sonidos
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=ax, orientation='vertical', label='Max Sonido Intensity')

# # Optional: Add legend
# ax[1].legend(fancybox=True,shadow=True,loc='upper left')
# ax[2].legend(fancybox=True,shadow=True,loc='upper left')
#%% aca con la mediana de los maximos

# Define n for the number of greatest values to consider
n = 5  # Change this value as needed

# Step 1: Calculate the median of the top n values for each sonido list
median_sonidos = [np.median(np.partition(RoNe_noche[i]['sonido'], -n)[-n:]) for i in range(len(RoNe_noche))]

# Step 2: Get the sorted indices of the median_sonidos in ascending order
sorted_indices = np.argsort(median_sonidos)

# Step 3: Sort RoNe_noche lists based on the sorted_indices
sorted_RoNe_noche = [RoNe_noche[i] for i in sorted_indices]

# Step 4: Extract sorted lists of sonidos, pressure, and rate
sorted_sonidos = [sorted_RoNe_noche[i]['sonido'] for i in range(len(sorted_RoNe_noche))]
sorted_pressure = [sorted_RoNe_noche[i]['presion'] for i in range(len(sorted_RoNe_noche))]
sorted_rate = [sorted_RoNe_noche[i]['rate'] for i in range(len(sorted_RoNe_noche))]

# Step 5: Normalize based on median_sonidos
norm = Normalize(vmin=min(median_sonidos), vmax=max(median_sonidos))

# Step 6: Choose a colormap
cmap = get_cmap('viridis')

# Step 7: Set up subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# fig.suptitle('RoVio en función de los medianas')
ax[0].set_ylabel("Audio (u. a.)", fontsize=20)
ax[1].set_ylabel("Presion (u. a.)", fontsize=20)
ax[2].set_ylabel("Rate (Hz)", fontsize=20)
ax[2].set_xlabel("Indice", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[2].tick_params(axis='both', labelsize=10)
plt.tight_layout()

# Step 8: Plot sorted data with colors based on median_sonidos
for i in range(len(sorted_RoNe_noche)):
    # Get the color based on the corresponding median_sonido value
    color = cmap(norm(median_sonidos[sorted_indices[i]]))

    # Plot the median of the top n sonido values as scatter points
    median_top_n_sonido = np.median(np.partition(sorted_sonidos[i], -n)[-n:])
    ax[0].scatter(i, median_top_n_sonido, color=color, s=10, alpha=0.8)

    # Plot the median of the top n pressure values as scatter points
    median_top_n_pressure = np.median(np.partition(sorted_pressure[i], -n)[-n:])
    ax[1].scatter(i, median_top_n_pressure, color=color, s=10, alpha=0.8)

    # Plot the median of the top n rate values as scatter points
    median_top_n_rate = np.median(np.partition(sorted_rate[i], -n)[-n:])
    ax[2].scatter(i, median_top_n_rate, color=color, s=10, alpha=0.8)

# Step 9: Perform linear regression on sorted rate medians
x_indices = np.arange(len(sorted_rate))
y_values = [np.median(np.partition(sorted_rate[i], -n)[-n:]) for i in range(len(sorted_rate))]

# Fit a linear regression line
coefficients, covariance_matrix = np.polyfit(x_indices, y_values, 1, cov=True)  # Linear fit with covariance
linear_fit = np.polyval(coefficients, x_indices)  # Evaluate the polynomial

# The errors are the square root of the diagonal elements of the covariance matrix
errors = np.sqrt(np.diag(covariance_matrix))

# Step 10: Plot the linear regression line for rate
ax[2].plot(x_indices, linear_fit, color='#890304', linewidth=2, label=fr'Pendiente = {round(coefficients[0],4)}$\pm${round(errors[0],4)}')#, b:{round(coefficients[1],2)}$\pm${round(errors[1],2)}

## Same for pressure medians
x_indices_p = np.arange(len(sorted_pressure))
y_values_p = [np.median(np.partition(sorted_pressure[i], -n)[-n:]) for i in range(len(sorted_pressure))]

# Fit a linear regression line
coefficients_p, covariance_matrix_p = np.polyfit(x_indices_p, y_values_p, 1, cov=True)  # Linear fit with covariance
linear_fit_p = np.polyval(coefficients_p, x_indices_p)  # Evaluate the polynomial

# The errors are the square root of the diagonal elements of the covariance matrix
errors_p = np.sqrt(np.diag(covariance_matrix_p))

# Step 10: Plot the linear regression line for pressure
ax[1].plot(x_indices_p, linear_fit_p, color='#890304', linewidth=2, label=fr'Pendiente = {round(coefficients_p[0],3)}$\pm${round(errors_p[0],3)}')#, b:{round(coefficients_p[1],2)}$\pm${round(errors_p[1],2)}

# Optional: Add a colorbar based on median_sonidos
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
cbar.set_label(label='Intensidad sonido', fontsize=20)
# Optional: Add legends
ax[1].legend(fancybox=True, shadow=True, loc='upper left',fontsize=12,prop={'size': 15})
ax[2].legend(fancybox=True, shadow=True, loc='upper left',fontsize=12,prop={'size': 15})

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory) 
# plt.savefig('ajuste_RoNe_pendientes.pdf')
#%% aca dibujo los sonidos con los puntos para las medianas


# Example number of top values to consider for the partition (median calculation)
n = 5  # Example: top 5 values

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(14, 7))

# Choose a colormap
cmap = get_cmap('viridis')

# Loop over each 'sonido' data in RoNe_noche and plot
for i in range(1):#len(RoNe_noche)
    # Extract the time and sound data for the current entry
    tiempo_sonido, sonido = RoNe_noche[i]['time'], RoNe_noche[i]['sonido']
    
    # Plot the full sound signal in black
    # ax.plot(tiempo_sonido, sonido, color='royalblue', alpha=0.1,solid_capstyle='projecting', label=f"Sound {i}" if i == 0 else "")
    ax.plot(tiempo_sonido, sonido, color='royalblue',zorder=0)
    # Step 1: Get the indices of the top 'n' values used for the median calculation
    top_n_indices = np.argpartition(sonido, -n)[-n:]  # Indices of the top 'n' values
    
    # Step 2: Get the corresponding times and values of the top 'n' points
    time_partition = tiempo_sonido[top_n_indices]     # Times of the top 'n' points
    partition = sonido[top_n_indices]                 # Values of the top 'n' points
    
    # Step 3: Normalize the top 'n' values for coloring
    norm = Normalize(vmin=min(partition), vmax=max(partition))
    
    # Step 4: Get the colors based on the normalized partition values
    colors = cmap(norm(partition))
    
    median_top_n_sonido = np.median(partition)
    median_top_n_time = np.median(time_partition)
    # Step 5: Plot the top 'n' points in color
    ax.scatter(time_partition, partition, color=colors, s=50, edgecolor='k')#, label=f"Top {n} maximos" if i == 0 else "",zorder=50
    ax.scatter(median_top_n_time, median_top_n_sonido,s=50,color='#890304',label='Mediana',zorder=50)
# Optional: Add color bar to indicate the intensity of the top values
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label=f"Top {n} maximos")
cbar.set_label(label='Intensidad sonido', fontsize=20)

# Add labels and title
# ax.set_title(f"Sound Plot with Top {n} Points for Median Calculation Highlighted", fontsize=16)
ax.set_xlabel("Tiempo (s)", fontsize=20)
ax.set_ylabel("Audio (u. a.)", fontsize=20)
ax.tick_params(axis='both', labelsize=13)
# Show the legend
ax.legend(fancybox=True, shadow=True, loc='upper left',fontsize=12,prop={'size': 15})
plt.tight_layout()
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory)
# plt.savefig('top_5_RoNe.pdf')
#%% El mayor y el menor altura del sonido
# sorted_sonidos_time = [sorted_RoNe_noche[i]['time'] for i in range(len(sorted_RoNe_noche))]
# sorted_pressure_time = [sorted_RoNe_noche[i]['time maximos'] for i in range(len(sorted_RoNe_noche))]
# sorted_rate_time = [sorted_RoNe_noche[i]['time rate'] for i in range(len(sorted_RoNe_noche))]


# fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# ax[0].set_ylabel("Max Audio (arb. u.)", fontsize=14)
# ax[1].set_ylabel(" Max Pressure (arb. u.)", fontsize=14)
# ax[2].set_ylabel(" Max Rate (Hz)", fontsize=14)
# ax[2].set_xlabel("Index", fontsize=14)
# plt.tight_layout()

# ax[0].plot(sorted_sonidos_time[0],sorted_sonidos[0])
# ax[0].plot(sorted_sonidos_time[-1],sorted_sonidos[-1])

# ax[1].plot(sorted_pressure_time[0],sorted_pressure[0])
# ax[1].plot(sorted_pressure_time[-1],sorted_pressure[-1])

# ax[2].plot(sorted_rate_time[0],sorted_rate[0])
# ax[2].plot(sorted_rate_time[-1],sorted_rate[-1])