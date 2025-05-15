# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:42:17 2024

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
import json
from scipy import stats
from scipy.signal import butter, sosfiltfilt
# Set the font to 'STIX'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['mathtext.fontset'] = 'stix'


#%%

def process_night_data(directories):
    """
    Process data from multiple directories and return a combined time series list.

    Args:
        directories (list of str): List of directory paths.

    Returns:
        list of dict: Combined time series data from all directories.
    """
    def datos_normalizados(Sonidos,Presiones,indice, ti, tf):
        
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
    
        
    combined_time_series_list = []


    
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
            
        # Loop through the data and process each one (currently only processing the first two entries)
        for indice in range(len(datos)):  # len(datos)
            # Extract normalization times and plane time from the data
            ti, tf = datos[indice]['Tiempo inicial normalizacion'], datos[indice]['Tiempo final normalizacion']
            tiempo_inicial = datos[indice]['Tiempo inicial avion']
            
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure)/fs, len(pressure))
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            # peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), prominence=.001)
            # spline_amplitude_sound = UnivariateSpline(time[peaks_sonido], audio[peaks_sonido], s=0, k=3)
            
            # interpolado = spline_amplitude_sound(time)
            
            lugar_maximos = []
            maximos = np.loadtxt(f'{name}_maximos.txt')
            # maximos = np.loadtxt(f'{name}_maximos_prueba.txt')
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

def interpolate_to_target_time_base(common_time_base, interpolated_data, target_time_base):
    """
    Interpolates the provided interpolated_data from common_time_base onto target_time_base.

    Args:
        common_time_base (numpy array): The original time base of the interpolated data.
        interpolated_data (numpy array): The data to be interpolated, in shape (n_series, n_time_points).
        target_time_base (numpy array): The new time base onto which the data will be interpolated.

    Returns:
        numpy array: Interpolated data on target_time_base.
    """
    # Cut the data to match the bounds of the target time base
    data_cortado = interpolated_data[:, (common_time_base >= target_time_base[0]) & 
                                     (common_time_base <= target_time_base[-1])]
    time_cortado = common_time_base[(common_time_base >= target_time_base[0]) & 
                                    (common_time_base <= target_time_base[-1])]

    # Interpolate onto the target time base
    interpolated_to_target = []
    for ts in data_cortado:
        interp_func = interp1d(time_cortado, ts, bounds_error=False, fill_value=np.nan)
        interpolated_to_target.append(interp_func(target_time_base))
    
    # Convert the list to a numpy array
    interpolated_to_target = np.array(interpolated_to_target)
    
    return interpolated_to_target

#%% Celda donde importo y proceso los datos

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
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder

# Store all directories in a list
directories = [noches_1, noches_2, noches_3]

# Process the night data from the directories using the process_night_data function
RoNe_noche = process_night_data(directories)

# Interpolate the sound data from RoNe_noche using 'sonido' as the data key and 'time' as the time key
# Specify a common time length of 44150 samples for sound
time_sonido_RoNe_noche, interpolated_sonido_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='sonido', time_key='time', common_time_length=300#44150
)

# Compute the average and standard deviation of the interpolated sound data
average_RoNe_noche_sonido, std_RoNe_noche_sonido, _ = compute_average_and_std(interpolated_sonido_RoNe_noche)

# Interpolate the pressure data from RoNe_noche using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_RoNe_noche, interpolated_presion_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_RoNe_noche_presion, std_RoNe_noche_presion, _ = compute_average_and_std(interpolated_presion_RoNe_noche)

# Interpolate the rate data from RoNe_noche using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_RoNe_noche, interpolated_rate_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_RoNe_noche_rate, std_RoNe_noche_rate, _ = compute_average_and_std(interpolated_rate_RoNe_noche)

# Notify the user when the program finishes execution
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # Optional: Path to an icon file
    timeout=10  # Notification will disappear after 10 seconds
)

#%% Grafico de todos los datos con su promedio
fig, ax = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
# fig.suptitle('NaRo viejos V2')
ax[0].set_ylabel("Audio (u. a.)", fontsize=20)
ax[1].set_ylabel("Presion (u. a.)", fontsize=20)
ax[2].set_ylabel("Rate (Hz)", fontsize=20)
ax[2].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
ax[2].tick_params(axis='both', labelsize=10)
plt.tight_layout()

colores1 = ["#0e2862","#152534","#1c2926","#b35f9f","#a99893"]
colores2 = ['#2c2836', '#477cec', '#9f5793', '#718464', '#be6157']

# Grafico de todas juntas 
for i in range(len(RoNe_noche)):
    tiempo_sonido, sonido = RoNe_noche[i]['time'],RoNe_noche[i]['sonido']
    ax[0].plot(tiempo_sonido,sonido,color='k',alpha=0.05,solid_capstyle='projecting')


    tiempo_presion, presion = RoNe_noche[i]['time maximos'],RoNe_noche[i]['presion']
    ax[1].plot(tiempo_presion,presion,color='k',alpha=0.05,solid_capstyle='projecting')


    timepo_rate, rate = RoNe_noche[i]['time rate'],RoNe_noche[i]['rate']
    ax[2].plot(timepo_rate,rate,color='k',alpha=0.05,solid_capstyle='projecting')
    

# Grafico del promedio 
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color= 'royalblue')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color= 'royalblue')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color= 'royalblue')

ax[0].set_ylim([0,10])
ax[1].set_ylim([0,4])
ax[2].set_ylim([0.5,4])

legend_handles = [
    mpatches.Patch(color= 'C0', label='Promedio'),
    mpatches.Patch(color='k', label='Datos'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper left',fontsize=12,prop={'size': 24})
# ax[1].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
# ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
os.chdir(directory)  # Change the working directory to the specified path

# plt.savefig('promedio_noche.png')

#%%

# np.savetxt('average RoNe rate', np.array([time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate]),delimiter=',')
#%%  Grafico de los 3 regiones de interes

fig, ax = plt.subplots(3,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Find the index at time 0
index_at_0 = (np.abs(time_sonido_RoNe_noche - 0)).argmin()
# Get the value of the sound at time 0
value_at_0 = average_RoNe_noche_sonido[index_at_0]

# Exclude the point at index 0 and find the closest index to value_at_0
excluded_indices = np.arange(len(average_RoNe_noche_sonido)) != index_at_0
closest_index_not_0 = (np.abs(average_RoNe_noche_sonido[excluded_indices] - value_at_0)).argmin()

# Since we excluded indices, convert back to the original index space
actual_closest_index = np.arange(len(average_RoNe_noche_sonido))[excluded_indices][closest_index_not_0]

# Get the time at that index
time_at_value = time_sonido_RoNe_noche[actual_closest_index]

# Find the index in time_maximos_RoNe_noche closest to time_at_value (for pressure)
index_at_value_presion = (np.abs(time_maximos_RoNe_noche - time_at_value)).argmin()
value_at_time_presion = average_RoNe_noche_presion[index_at_value_presion]

index_at_value_presion_0 = (np.abs(time_maximos_RoNe_noche - 0)).argmin()
value_at_time_presion_0 = average_RoNe_noche_presion[index_at_value_presion_0]

# Find the index in time_rate_RoNe_noche closest to time_at_value (for rate)
index_at_value_rate = (np.abs(time_rate_RoNe_noche - time_at_value)).argmin()
value_at_time_rate = average_RoNe_noche_rate[index_at_value_rate]


index_at_value_rate_0 = (np.abs(time_rate_RoNe_noche - 0)).argmin()
value_at_time_rate_0 = average_RoNe_noche_rate[index_at_value_rate_0]

# Plot for ax[0] with time_sonido_RoNe
ax[0].axvspan(time_sonido_RoNe_noche[0], 0, facecolor='g', alpha=0.3,edgecolor='k',linestyle='--',label='Before')
ax[0].axvspan(0, time_at_value, facecolor='b', alpha=0.3,edgecolor='k',linestyle='--',label='During')
ax[0].axvspan(time_at_value, time_sonido_RoNe_noche[-1], facecolor='r', alpha=0.3,edgecolor='k',linestyle='--',label='After')
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='C0')
ax[0].text(20,3, f"{round(time_at_value,2)}s" , fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[0].legend(fancybox=True,shadow=True)
# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))


# Plot for ax[1] with time_maximos_RoNe
ax[1].axvspan(time_sonido_RoNe_noche[0], 0, facecolor='g', alpha=0.3,edgecolor='k',linestyle='--',label='Before')
ax[1].axvspan(0, time_at_value, facecolor='b', alpha=0.3,edgecolor='k',linestyle='--',label='During')
ax[1].axvspan(time_at_value, time_sonido_RoNe_noche[-1], facecolor='r', alpha=0.3,edgecolor='k',linestyle='--',label='After')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0')
# ax[1].text(-5,3, round(value_at_time_presion_0,2) , fontsize=12, bbox=dict(facecolor='k', alpha=0.5))
ax[1].text(20,3, f"{round(100*(value_at_time_presion-value_at_time_presion_0)/value_at_time_presion_0,2)}%" , fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[1].legend(fancybox=True,shadow=True)
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))


# Plot for ax[2] with time_rate_RoNe
ax[2].axvspan(time_sonido_RoNe_noche[0], 0, facecolor='g', alpha=0.3,edgecolor='k',linestyle='--',label='Before')
ax[2].axvspan(0, time_at_value, facecolor='b', alpha=0.3,edgecolor='k',linestyle='--',label='During')
ax[2].axvspan(time_at_value, time_sonido_RoNe_noche[-1], facecolor='r', alpha=0.3,edgecolor='k',linestyle='--',label='After')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color='C0')
ax[2].text(20,2, f"{round(100*(value_at_time_rate-value_at_time_rate_0)/value_at_time_rate_0,2)}%" , fontsize=12, bbox=dict(facecolor='k', alpha=0.1))

ax[2].legend(fancybox=True,shadow=True)
# ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))

#%% Quiero ver durante el dia aca

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia'
os.chdir(directory)
carpetas = os.listdir(directory)
pajaro = carpetas[0]

subdirectory = os.path.join(directory, pajaro)


dias = os.listdir(subdirectory)
pajaritos = '\Aviones y pajaros'
dia_1 = subdirectory + '/' + dias[0] + pajaritos
dia_2 = subdirectory + '/' + dias[1] + pajaritos
dia_3 = subdirectory + '/' + dias[2] + pajaritos
dia_4 = subdirectory + '/' + dias[3] + pajaritos

directories = [dia_1,dia_2,dia_3,dia_4]

RoNe_dia = process_night_data(directories)


# Interpolating the sound data for the day
time_sonido_RoNe_dia, interpolated_sonido_RoNe_dia = interpolate_single_data(
    RoNe_dia, data_key='sonido', time_key='time', common_time_length=44150)

# Compute average and standard deviation for the sound data
average_RoNe_dia_sonido, std_RoNe_dia_sonido, _ = compute_average_and_std(interpolated_sonido_RoNe_dia)


# Interpolating the pressure data for the day
time_maximos_RoNe_dia, interpolated_presion_RoNe_dia = interpolate_single_data(
    RoNe_dia, data_key='presion', time_key='time maximos', common_time_length=300)

# Compute average and standard deviation for the pressure data
average_RoNe_dia_presion, std_RoNe_dia_presion, _ = compute_average_and_std(interpolated_presion_RoNe_dia)


# Interpolating the rate data for the day
time_rate_RoNe_dia, interpolated_rate_RoNe_dia = interpolate_single_data(
    RoNe_dia, data_key='rate', time_key='time rate', common_time_length=300)

# Compute average and standard deviation for the rate data
average_RoNe_dia_rate, std_RoNe_dia_rate, _ = compute_average_and_std(interpolated_rate_RoNe_dia)


notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)


#%% Graficos de todos los plots del dia supuerpuestos 
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

for i in range(len(RoNe_dia)):
    tiempo_sonido, sonido = RoNe_dia[i]['time'],RoNe_dia[i]['sonido']
    ax[0].plot(tiempo_sonido, sonido, color='k', alpha=0.1, solid_capstyle='projecting')


    tiempo_presion, presion = RoNe_dia[i]['time maximos'],RoNe_dia[i]['presion']
    ax[1].plot(tiempo_presion, presion, color='k', alpha=0.1, solid_capstyle='projecting')


    tiempo_rate, rate = RoNe_dia[i]['time rate'],RoNe_dia[i]['rate']
    ax[2].plot(tiempo_rate, rate, color='k', alpha=0.1, solid_capstyle='projecting')
    
ax[0].errorbar(time_sonido_RoNe_dia, average_RoNe_dia_sonido, yerr=std_RoNe_dia_sonido, color='C0')    
ax[1].errorbar(time_maximos_RoNe_dia, average_RoNe_dia_presion, yerr=std_RoNe_dia_presion, color='C0') 
ax[2].errorbar(time_rate_RoNe_dia, average_RoNe_dia_rate, yerr=std_RoNe_dia_rate, color='C0') 

# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))

legend_handles = [
    mpatches.Patch(color='C0', label='Average'),
    mpatches.Patch(color='k', label='Data'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')

#%% Grafico de comparacion del dia y la noche
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

## Noche
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='C0', label='Noche')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0', label='Noche')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate/np.mean(average_RoNe_noche_rate[0:100]), std_RoNe_noche_rate, color='C0', label='Noche')
## Dia
ax[0].errorbar(time_sonido_RoNe_dia, average_RoNe_dia_sonido, yerr=std_RoNe_dia_sonido, color='C2', label='Dia')    
ax[1].errorbar(time_maximos_RoNe_dia, average_RoNe_dia_presion/np.mean(average_RoNe_dia_presion[0:100]), yerr=std_RoNe_dia_presion, color='C2', label='Dia') 
ax[2].errorbar(time_rate_RoNe_dia, average_RoNe_dia_rate/np.mean(average_RoNe_dia_rate[0:100]), yerr=std_RoNe_dia_rate, color='C2', label='Dia') 

# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))

ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)
ax[2].legend(fancybox=True,shadow=True)



#%% Aca interpolo los datos del rate de la noche sobre los tiempos del dia


interpolated_rate_on_day_base = interpolate_to_target_time_base(
    time_rate_RoNe_noche, interpolated_rate_RoNe_noche, time_rate_RoNe_dia
)
rate_interpolated_rate_noche_cortado, rate_std_interpolated_rate_noche_cortado, _ = compute_average_and_std(interpolated_rate_on_day_base)

p_values_rate = []

# Iterate over the time points
for i in range(len(time_rate_RoNe_dia)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_RoNe_dia[:, i]/np.mean(average_RoNe_dia_rate[0:100]), interpolated_rate_on_day_base[:, i]/np.nanmean(rate_interpolated_rate_noche_cortado[0:100]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_rate = np.array(p_values_rate)

fig, ax = plt.subplots(2,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Rate (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"P-value", fontsize=14)
ax[1].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[0].errorbar(time_rate_RoNe_dia, average_RoNe_dia_rate/np.mean(average_RoNe_dia_rate[0:100]), yerr= std_RoNe_dia_rate, color='C0', label='Dia') 
ax[0].errorbar(time_rate_RoNe_dia,rate_interpolated_rate_noche_cortado/np.nanmean(rate_interpolated_rate_noche_cortado[0:100]), yerr=rate_std_interpolated_rate_noche_cortado,color='C2', label='Noche')
ax[1].plot(time_rate_RoNe_dia,p_values_rate)
ax[0].legend(fancybox=True,shadow=True)

# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
fig, ax = plt.subplots(2,1,figsize=(15,10),sharex=True)
ax[0].set_ylabel("Presion (u. a.)", fontsize=15)
ax[1].set_ylabel("Rate (Hz)", fontsize=15)
ax[1].set_xlabel("Tiempo (s)", fontsize=15)
ax[0].tick_params(axis='both', labelsize=13)
ax[1].tick_params(axis='both', labelsize=13)

ax[1].errorbar(time_rate_RoNe_dia, average_RoNe_dia_rate/np.mean(average_RoNe_dia_rate[0:indice_rate]), yerr= std_RoNe_dia_rate, color=colors[0]) 
ax[1].errorbar(time_rate_RoNe_dia,rate_interpolated_rate_noche_cortado/np.nanmean(rate_interpolated_rate_noche_cortado[0:indice_rate]), yerr=rate_std_interpolated_rate_noche_cortado,color='royalblue')
# ax[1].errorbar(time_rate_RoNe_basal, average_RoNe_basal_rate/np.mean(average_RoNe_basal_rate[0:indice_rate]), yerr= std_RoNe_basal_rate, color=colors[1], label='Basal') 
ax[0].errorbar(time_maximos_RoNe_dia, average_RoNe_dia_presion/np.mean(average_RoNe_dia_presion[0:indice_maximos]),yerr= std_RoNe_dia_presion, color=colors[0], label='Dia')
ax[0].errorbar(time_maximos_RoNe_dia,presion_interpolated_presion_noche_cortado/np.nanmean(presion_interpolated_presion_noche_cortado[0:indice_maximos]), yerr=presion_std_interpolated_presion_noche_cortado,color='royalblue', label='Noche')
# ax[0].errorbar(time_maximos_RoNe_basal, average_RoNe_basal_presion/np.mean(average_RoNe_basal_presion[0:indice_maximos]), yerr= std_RoNe_basal_presion, color=colors[1], label='Basal')
ax[1].axvspan(1.58,22.22,color='#b35f9f',alpha=0.3, edgecolor='k', linestyle='--')
ax[0].axvspan(1.34,20.5, facecolor='#B35F9F',alpha=0.3, edgecolor='k', linestyle='--',label= 'P-value < 0.05')
# ax[3].plot(time_rate_RoNe_dia, p_values_rate)
# ax[2].plot(time_maximos_RoNe_dia, p_values_presion)
ax[1].set_xlim(-23,29)
# plt.tight_layout()
# ax[0].legend(fancybox=True,shadow=True,loc='upper left')
# ax[1].legend(fancybox=True,shadow=True,loc='upper left')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.951), bbox_transform=fig.transFigure, ncol=3, fontsize=12)
# plt.savefig('Noche vs dia vs basal.pdf')
#%% Aca interpolo los datos de la presion de la noche sobre los tiempos del dia
# Interpolating the presion data to the time base of the day
interpolated_presion_on_day_base = interpolate_to_target_time_base(
    time_maximos_RoNe_noche, interpolated_presion_RoNe_noche, time_maximos_RoNe_dia
)

# Compute the average and standard deviation for the interpolated presion data
presion_interpolated_presion_noche_cortado, presion_std_interpolated_presion_noche_cortado, _ = compute_average_and_std(interpolated_presion_on_day_base)

p_values_presion = []

# Iterate over the time points for presion data
for i in range(len(time_maximos_RoNe_dia)):
    # Perform t-test between corresponding data columns from interpolated day and night data
    _, p_value = stats.ttest_ind(interpolated_presion_RoNe_dia[:, i]/np.mean(average_RoNe_dia_presion[0:100]), 
                                 interpolated_presion_on_day_base[:, i]/np.nanmean(presion_interpolated_presion_noche_cortado[0:100]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_presion = np.array(p_values_presion)

# Plotting the results
fig, ax = plt.subplots(2, 1, figsize=(14, 7),sharex=True)
ax[0].set_ylabel(r"Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"P-value", fontsize=14)
ax[1].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

# Plot the interpolated presion data for day and night with error bars
ax[0].errorbar(time_maximos_RoNe_dia, average_RoNe_dia_presion/np.mean(average_RoNe_dia_presion[0:100]), 
               yerr=std_RoNe_dia_presion, color='C0', label='Dia') 
ax[0].errorbar(time_maximos_RoNe_dia, presion_interpolated_presion_noche_cortado/np.nanmean(presion_interpolated_presion_noche_cortado[0:100]), 
               yerr=presion_std_interpolated_presion_noche_cortado, color='C2', label='Noche')

# Plot the p-values for presion data
ax[1].plot(time_maximos_RoNe_dia, p_values_presion)

# Add legend
ax[0].legend(fancybox=True, shadow=True)

# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))

#%% Aca importo RoVio para ver si puedo calcular el p-value entre ambos

# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoVio
pajaro = carpetas[1]  # Select the second folder (assumed to be related to 'RoVio')

subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoVio' folder

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)

# Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder

# Store all directories in a list
directories = [noches_1, noches_2, noches_3]

# Process the night data from the directories using the process_night_data function
RoVio_noche = process_night_data(directories)

# Interpolate the sound data from RoVio_noche using 'sonido' as the data key and 'time' as the time key
# Specify a common time length of 44150 samples for sound
time_sonido_RoVio_noche, interpolated_sonido_RoVio_noche = interpolate_single_data(
    RoVio_noche, data_key='sonido', time_key='time', common_time_length=44150
)

# Compute the average and standard deviation of the interpolated sound data
average_RoVio_noche_sonido, std_RoVio_noche_sonido, _ = compute_average_and_std(interpolated_sonido_RoVio_noche)

# Interpolate the pressure data from RoVio_noche using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_RoVio_noche, interpolated_presion_RoVio_noche = interpolate_single_data(
    RoVio_noche, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_RoVio_noche_presion, std_RoVio_noche_presion, _ = compute_average_and_std(interpolated_presion_RoVio_noche)

# Interpolate the rate data from RoVio_noche using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_RoVio_noche, interpolated_rate_RoVio_noche = interpolate_single_data(
    RoVio_noche, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_RoVio_noche_rate, std_RoVio_noche_rate, _ = compute_average_and_std(interpolated_rate_RoVio_noche)

#%%
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()
ax[0].errorbar(time_sonido_RoVio_noche, average_RoVio_noche_sonido, std_RoVio_noche_sonido, color='C0', label='Noche')
ax[1].errorbar(time_maximos_RoVio_noche, average_RoVio_noche_presion, std_RoVio_noche_presion, color='C0', label='Noche')
ax[2].errorbar(time_rate_RoVio_noche, average_RoVio_noche_rate, std_RoVio_noche_rate, color='C0', label='Noche')


#%%
interpolated_rate_on_day_base = interpolate_to_target_time_base(
    time_rate_RoVio_noche, interpolated_rate_RoVio_noche, time_rate_RoNe_noche
)
rate_interpolated_rate_noche_cortado, rate_std_interpolated_rate_noche_cortado, _ = compute_average_and_std(interpolated_rate_on_day_base)

p_values_rate = []

# Iterate over the time points
for i in range(len(time_rate_RoNe_noche)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_RoNe_noche[:, i]/np.mean(average_RoNe_noche_rate[0:100]), interpolated_rate_on_day_base[:, i]/np.nanmean(rate_interpolated_rate_noche_cortado[0:100]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_rate = np.array(p_values_rate)

fig, ax = plt.subplots(2,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Rate (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"P-value", fontsize=14)
ax[1].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[0].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate/np.mean(average_RoNe_noche_rate[0:100]), yerr= std_RoNe_noche_rate, color='C0', label='RoNe') 
ax[0].errorbar(time_rate_RoNe_noche,rate_interpolated_rate_noche_cortado/np.nanmean(rate_interpolated_rate_noche_cortado[0:100]), yerr=rate_std_interpolated_rate_noche_cortado,color='C2', label='RoVio')
ax[1].plot(time_rate_RoNe_noche,p_values_rate)
ax[0].legend(fancybox=True,shadow=True)

# ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))
# ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: r"$%g$" % val))


