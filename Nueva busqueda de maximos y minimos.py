# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:15:29 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pickle
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
import json
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt
from scipy import signal
from scipy.interpolate import interp1d
# Set the font to 'STIX'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['mathtext.fontset'] = 'stix'

def plot_night_data(directories, name):
    # Create a PDF object to save the plots
    pdf = PdfPages(name)
    
    # Function to normalize the audio and pressure data in a specific time interval [ti, tf]
    def datos_normalizados(Sonidos, Presiones, indice, ti, tf):
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]  # Extract the name from the Pressure file

        # Read audio and pressure files
        fs, audio = wavfile.read(Sound)
        fs, pressure = wavfile.read(Pressure)
       
        # Normalize the audio and pressure data
        audio = audio - np.mean(audio)  # Remove mean
        audio_norm = audio / np.max(audio)  # Normalize by max value
        
        pressure = pressure - np.mean(pressure)  # Remove mean
        pressure_norm = pressure / np.max(pressure)  # Normalize by max value
        
        # Function to normalize the data to [-1, 1] in a given time interval
        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        # Normalize audio and pressure within the interval
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)
    
        return audio_norm, pressure_norm, name, fs
    
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

    # Function to find the longest interval of consecutive time values that are within a gap threshold
    def find_longest_interval(times, max_gap=1):
        longest_interval = []
        current_interval = [times[0]]

        # Iterate through time values and check the gap between consecutive times
        for i in range(1, len(times)):
            if times[i] - times[i-1] <= max_gap:
                current_interval.append(times[i])
            else:
                # If current interval is longer than the previous longest, update it
                if len(current_interval) > len(longest_interval):
                    longest_interval = current_interval
                current_interval = [times[i]]

        # Final check for the last interval
        if len(current_interval) > len(longest_interval):
            longest_interval = current_interval

        return longest_interval
    
    
    
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = []  # To store pressure files
        sonidos = []  # To store sound files
        datos = []  # To store data files
        
        # Classify files into sonidos, presiones, and datos based on their names
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
            
            # Get normalized audio and pressure data for the current index
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
            peaks_sonido_no_limpio, _ = find_peaks(audio, height=0, distance=int(fs * 0.1), prominence=.001)  # Find peaks in raw audio
            
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
            # Filter maxima by ensuring there's a minimum between them
            peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure)
            
            # Generate a spectrogram of the filtered signal
            f, t, Sxx = signal.spectrogram(filtered_signal, fs)
            
            # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
            frequency_cutoff = 1000  # Frequency cutoff in Hz
            threshold = -10  # Threshold for spectrogram in dB
            Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale
            
            # Find where frequencies exceed the threshold
            freq_indices = np.where(f > frequency_cutoff)[0]
            time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
            time_above_threshold = t[time_indices]
            longest_interval = find_longest_interval(time_above_threshold)
            
            # Calculate periods between maxima (filtered and unfiltered)
            periodo = np.diff(time[peaks_maximos_filtered])
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            periodo_sin_filtrar = np.diff(time[lugar_maximos])
            
            # Create a figure with three subplots (audio, pressure, and rate)
            fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
            fig.suptitle(f'{name}', fontsize=16)  # Title with the name
            ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
            ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
            ax[2].set_ylabel("Rate (Hz)", fontsize=14)
            ax[2].set_xlabel("Time (s)", fontsize=14)
            plt.tight_layout()
            
            # Plot audio with and without filtering
            ax[0].plot(time[peaks_sonido_no_limpio], audio[peaks_sonido_no_limpio], color='C0', label='Sin Filtrado')
            ax[0].plot(time[peaks_sonido], audio[peaks_sonido], color='C3', linestyle='--',linewidth=2, alpha=0.7, label='Filtrado')
            ax[0].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            ax[0].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[0].text(10,3,round(tiempo_inicial,2), fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
            
            # Plot pressure with manually identified maxima and filtered maxima
            ax[1].plot(time, pressure, color='C0',alpha=0.7)
            ax[1].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            ax[1].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[1].plot(time[lugar_maximos], pressure[lugar_maximos], marker='X', linestyle='', color='darkslategray', ms=11, label='A mano')
            ax[1].plot(time[peaks_maximos_filtered], pressure[peaks_maximos_filtered], '.C1', ms=10, label='con minimos')
            ax[1].plot(time[peaks_minimos], pressure[peaks_minimos], '.C2', ms=10)
            
            # Plot rate based on periods (filtered and unfiltered)
            ax[2].plot(time[lugar_maximos][1:], 1 / periodo_sin_filtrar, 'darkslategray', label='A mano')
            ax[2].plot(time[peaks_maximos_filtered][1:], 1 / periodo, color='C1',linestyle='--',linewidth=2, label='con minimos')
            ax[2].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            ax[2].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            
            # Add legends
            ax[0].legend(fancybox=True, shadow=True, loc='upper right')
            ax[1].legend(fancybox=True, shadow=True, loc='upper right')
            ax[2].legend(fancybox=True, shadow=True, loc='upper right')
            
            # Save the current figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)  # Close the figure to free memory
            
    pdf.close()  # Close the PDF document


#%%
# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[2]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoNe' folder

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)

# Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
pajaritos = '\Aviones\Aviones y pajaros V2'
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-28-night\poster'
# Store all directories in a list
directories = [directory]

# Process the night data from the directories using the process_night_data function
RoNe_noche = plot_night_data(directories,'RoNe plots nuevo vs viejo V3.pdf')

#%%
# Create a figure with three subplots (audio, pressure, and rate)
# fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
# fig.suptitle('RoNe nuevo', fontsize=16)  # Title with the name
# ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
# ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
# ax[2].set_ylabel("Rate (Hz)", fontsize=14)
# ax[2].set_xlabel("Time (s)", fontsize=14)
# plt.tight_layout()

def plot_night_data(directories):
      
    # Function to normalize the audio and pressure data in a specific time interval [ti, tf]
    def datos_normalizados(Sonidos, Presiones, indice, ti, tf):
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]  # Extract the name from the Pressure file

        # Read audio and pressure files
        fs, audio = wavfile.read(Sound)
        fs, pressure = wavfile.read(Pressure)
       
        # Normalize the audio and pressure data
        audio = audio - np.mean(audio)  # Remove mean
        audio_norm = audio / np.max(audio)  # Normalize by max value
        
        pressure = pressure - np.mean(pressure)  # Remove mean
        pressure_norm = pressure / np.max(pressure)  # Normalize by max value
        
        # Function to normalize the data to [-1, 1] in a given time interval
        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        # Normalize audio and pressure within the interval
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)
    
        return audio_norm, pressure_norm, name, fs
    
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

    # Function to find the longest interval of consecutive time values that are within a gap threshold
    def find_longest_interval(times, max_gap=1):
        longest_interval = []
        current_interval = [times[0]]

        # Iterate through time values and check the gap between consecutive times
        for i in range(1, len(times)):
            if times[i] - times[i-1] <= max_gap:
                current_interval.append(times[i])
            else:
                # If current interval is longer than the previous longest, update it
                if len(current_interval) > len(longest_interval):
                    longest_interval = current_interval
                current_interval = [times[i]]

        # Final check for the last interval
        if len(current_interval) > len(longest_interval):
            longest_interval = current_interval

        return longest_interval
    
    combined_time_series_list = []
    
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = []  # To store pressure files
        sonidos = []  # To store sound files
        datos = []  # To store data files
        
        # Classify files into sonidos, presiones, and datos based on their names
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
            
            
            # Get normalized audio and pressure data for the current index
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
           
            
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
            # Filter maxima by ensuring there's a minimum between them
            peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure)
            
            
            
            # Generate a spectrogram of the filtered signal
            f, t, Sxx = signal.spectrogram(filtered_signal, fs)
            
            # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
            frequency_cutoff = 1000  # Frequency cutoff in Hz
            threshold = -10  # Threshold for spectrogram in dB
            Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale
            
            # Find where frequencies exceed the threshold
            freq_indices = np.where(f > frequency_cutoff)[0]
            time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
            time_above_threshold = t[time_indices]
            longest_interval = find_longest_interval(time_above_threshold)
            periodo = np.diff(time[peaks_maximos_filtered])
            
            tiempo = time - longest_interval[0]
            
            # Calculate periods between maxima (filtered and unfiltered)
            
            
           
            # ax[0].plot(tiempo[peaks_sonido], audio[peaks_sonido], color='k',alpha=0.1,solid_capstyle='projecting', label='Filtrado')
           
            # ax[1].plot(tiempo[peaks_maximos_filtered], pressure[peaks_maximos_filtered],color='k',alpha=0.1,solid_capstyle='projecting', label='con minimos')
           
            # ax[2].plot(tiempo[peaks_maximos_filtered][1:], 1 / periodo,color='k',alpha=0.1,solid_capstyle='projecting', label='con minimos')
           
            
            # # Add legends
            # ax[0].legend(fancybox=True, shadow=True, loc='upper right')
            # ax[1].legend(fancybox=True, shadow=True, loc='upper right')
            # ax[2].legend(fancybox=True, shadow=True, loc='upper right')
            
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[peaks_maximos_filtered],
                'presion': pressure[peaks_maximos_filtered],
                'time rate': tiempo[peaks_maximos_filtered][1:],
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
pajaritos = '\Aviones\Aviones y pajaros'
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder

# Store all directories in a list
directories = [noches_1, noches_2, noches_3]

# Process the night data from the directories using the process_night_data function
RoNe_noche = plot_night_data_2(directories)


time_sonido_RoNe_noche, interpolated_sonido_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='sonido', time_key='time', common_time_length=44150
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
# notification.notify(
#     title='Program Finished',
#     message='Your Python program has finished running.',
#     app_icon=None,  # Optional: Path to an icon file
#     timeout=10  # Notification will disappear after 10 seconds
# )
#%%

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
fig.suptitle('NaRo nuevos maximos V2')
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Grafico de todas juntas 
for i in range(len(RoNe_noche)):
    tiempo_sonido, sonido = RoNe_noche[i]['time'],RoNe_noche[i]['sonido']
    ax[0].plot(tiempo_sonido,sonido,color='k',alpha=0.1,solid_capstyle='projecting')


    tiempo_presion, presion = RoNe_noche[i]['time maximos'],RoNe_noche[i]['presion']
    ax[1].plot(tiempo_presion,presion,color='k',alpha=0.1,solid_capstyle='projecting')


    timepo_rate, rate = RoNe_noche[i]['time rate'],RoNe_noche[i]['rate']
    ax[2].plot(timepo_rate,rate,color='k',alpha=0.1,solid_capstyle='projecting')
    

# Grafico del promedio 
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='C0')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color='C0')


legend_handles = [
    mpatches.Patch(color='C0', label='Average'),
    mpatches.Patch(color='k', label='Data'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')

#%% RoVio

pajaro = carpetas[1]  # Select the first folder (assumed to be related to 'RoNe')

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

RoVio_noche = plot_night_data_2(directories)

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
#%% NaRo

pajaro = carpetas[2]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoNe' folder

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)

# Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
pajaritos = '\Aviones\Aviones y pajaros V2'
noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder

# Store all directories in a list
directories = [noches_1, noches_2, noches_3]

NaRo_noche = plot_night_data_2(directories)

time_sonido_NaRo_noche, interpolated_sonido_NaRo_noche = interpolate_single_data(
    NaRo_noche, data_key='sonido', time_key='time', common_time_length=44150
)

# Compute the average and standard deviation of the interpolated sound data
average_NaRo_noche_sonido, std_NaRo_noche_sonido, _ = compute_average_and_std(interpolated_sonido_NaRo_noche)

# Interpolate the pressure data from NaRo_noche using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_NaRo_noche, interpolated_presion_NaRo_noche = interpolate_single_data(
    NaRo_noche, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_NaRo_noche_presion, std_NaRo_noche_presion, _ = compute_average_and_std(interpolated_presion_NaRo_noche)

# Interpolate the rate data from NaRo_noche using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_NaRo_noche, interpolated_rate_NaRo_noche = interpolate_single_data(
    NaRo_noche, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_NaRo_noche_rate, std_NaRo_noche_rate, _ = compute_average_and_std(interpolated_rate_NaRo_noche)


#%%

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
fig.suptitle('Nuevos maximos db=-10.5')
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# RoNe Data
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='C0', label='RoNe')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0', label='RoNe')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate/np.mean(average_RoNe_noche_rate[:100]), std_RoNe_noche_rate, color='C0', label='RoNe')

# RoVio Data
ax[0].errorbar(time_sonido_RoVio_noche, average_RoVio_noche_sonido, std_RoVio_noche_sonido, color='C1', label='RoVio')
ax[1].errorbar(time_maximos_RoVio_noche, average_RoVio_noche_presion, std_RoVio_noche_presion, color='C1', label='RoVio')
ax[2].errorbar(time_rate_RoVio_noche, average_RoVio_noche_rate/np.mean(average_RoVio_noche_rate[:100]), std_RoVio_noche_rate, color='C1', label='RoVio')

# NaRo Data
ax[0].errorbar(time_sonido_NaRo_noche, average_NaRo_noche_sonido, std_NaRo_noche_sonido, color='C2', label='NaRo')
ax[1].errorbar(time_maximos_NaRo_noche, average_NaRo_noche_presion, std_NaRo_noche_presion, color='C2', label='NaRo')
ax[2].errorbar(time_rate_NaRo_noche, average_NaRo_noche_rate/np.mean(average_NaRo_noche_rate[:100]), std_NaRo_noche_rate, color='C2', label='NaRo')

# Add legends for each axis
ax[0].legend(fancybox=True, shadow=True)
ax[1].legend(fancybox=True, shadow=True)
ax[2].legend(fancybox=True, shadow=True)

#%% aca uso los maximos que yo limpie a mano
def plot_night_data_2(directories):
      
    # Function to normalize the audio and pressure data in a specific time interval [ti, tf]
    def datos_normalizados(Sonidos, Presiones, indice, ti, tf):
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]  # Extract the name from the Pressure file

        # Read audio and pressure files
        fs, audio = wavfile.read(Sound)
        fs, pressure = wavfile.read(Pressure)
       
        # Normalize the audio and pressure data
        audio = audio - np.mean(audio)  # Remove mean
        audio_norm = audio / np.max(audio)  # Normalize by max value
        
        pressure = pressure - np.mean(pressure)  # Remove mean
        pressure_norm = pressure / np.max(pressure)  # Normalize by max value
        
        # Function to normalize the data to [-1, 1] in a given time interval
        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        # Normalize audio and pressure within the interval
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)
    
        return audio_norm, pressure_norm, name, fs
    
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

    # Function to find the longest interval of consecutive time values that are within a gap threshold
    def find_longest_interval(times, max_gap=1):
        longest_interval = []
        current_interval = [times[0]]

        # Iterate through time values and check the gap between consecutive times
        for i in range(1, len(times)):
            if times[i] - times[i-1] <= max_gap:
                current_interval.append(times[i])
            else:
                # If current interval is longer than the previous longest, update it
                if len(current_interval) > len(longest_interval):
                    longest_interval = current_interval
                current_interval = [times[i]]

        # Final check for the last interval
        if len(current_interval) > len(longest_interval):
            longest_interval = current_interval

        return longest_interval
    
    combined_time_series_list = []
    
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = []  # To store pressure files
        sonidos = []  # To store sound files
        datos = []  # To store data files
        
        # Classify files into sonidos, presiones, and datos based on their names
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
            
            # Get normalized audio and pressure data for the current index
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
           
            
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            # peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            # peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
            # Filter maxima by ensuring there's a minimum between them
            # peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure)
            
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            periodo_sin_filtrar = np.diff(time[lugar_maximos])
            
            # Generate a spectrogram of the filtered signal
            f, t, Sxx = signal.spectrogram(filtered_signal, fs)
            
            # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
            frequency_cutoff = 1000  # Frequency cutoff in Hz
            threshold = -10.5  # Threshold for spectrogram in dB
            Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale
            
            # Find where frequencies exceed the threshold
            freq_indices = np.where(f > frequency_cutoff)[0]
            time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
            time_above_threshold = t[time_indices]
            longest_interval = find_longest_interval(time_above_threshold)
            # periodo = np.diff(time[peaks_maximos_filtered])
            
            tiempo = time - longest_interval[0]
            # tiempo = time - tiempo_inicial
            
            
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time rate': tiempo[lugar_maximos][1:],
                'rate': 1/periodo_sin_filtrar
            })
            
    return combined_time_series_list
#%% aca voy a usar solo los de presion basal

def plot_night_data_basal(directories):
      
    def datos_normalizados_basal(Presiones,indice, ti, tf):
        Pressure = Presiones[indice]
        name = Pressure[9:-4]
        
        file_path = os.path.join(directory, Pressure)
        fs, pressure = wavfile.read(file_path)
        
        pressure = pressure - np.mean(pressure)
        pressure_norm = pressure / np.max(pressure)

        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        return pressure_norm, name, fs
    
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
    
    
    combined_time_series_list = []
    
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = [file for file in files if file[0] == 'p']  # Files starting with 'p'
        datos = [file for file in files if file.endswith('json')]  # Files ending with 'json'
        
        # Load the first JSON file
        with open(datos[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)
            
        # Loop through the data and process each one (currently only processing the first two entries)
        for indice in range(len(datos)):  # len(datos)
            # Extract normalization times and plane time from the data
            ti, tf = datos[indice]['Tiempo inicial normalizacion'], datos[indice]['Tiempo final normalizacion']
            tiempo_inicial = datos[indice]['Tiempo inicial avion']
            
            # Get normalized audio and pressure data for the current index
            pressure, name, fs = datos_normalizados_basal(presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
           
            audio = np.sin(time)
            peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs * 0.1), prominence=.001)
            # Apply bandpass filter to audio signal
            # filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            # peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            # peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            # peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
            # Filter maxima by ensuring there's a minimum between them
            # peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure)
            
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            periodo_sin_filtrar = np.diff(time[lugar_maximos])
            
            # Generate a spectrogram of the filtered signal
            # f, t, Sxx = signal.spectrogram(filtered_signal, fs)
            
            # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
            # frequency_cutoff = 1000  # Frequency cutoff in Hz
            # threshold = -10.5  # Threshold for spectrogram in dB
            # Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale
            
            # # Find where frequencies exceed the threshold
            # freq_indices = np.where(f > frequency_cutoff)[0]
            # time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
            # time_above_threshold = t[time_indices]
            # longest_interval = find_longest_interval(time_above_threshold)
            # periodo = np.diff(time[peaks_maximos_filtered])
            
            # tiempo = time - longest_interval[0]
            tiempo = time - 30
            
            
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time rate': tiempo[lugar_maximos][1:],
                'rate': 1/periodo_sin_filtrar
            })
            
    return combined_time_series_list

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory)  # List all folders within the directory

# pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

# subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoNe' folder

# # List all subdirectories (representing different days)
# dias = os.listdir(subdirectory)

# # Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
# basal = '/'+dias[3] + '/' + 'Basal'
# Store all directories in a list
directories = [directory]

RoNe_basal = plot_night_data_basal(directories)

# time_sonido_RoNe_basal, interpolated_sonido_RoNe_basal = interpolate_single_data(
#     RoNe_basal, data_key='sonido', time_key='time', common_time_length=44150
# )

# Compute the average and standard deviation of the interpolated sound data
# average_RoNe_basal_sonido, std_RoNe_basal_sonido, _ = compute_average_and_std(interpolated_sonido_RoNe_basal)

# Interpolate the pressure data from RoNe_basal using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_RoNe_basal, interpolated_presion_RoNe_basal = interpolate_single_data(
    RoNe_basal, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_RoNe_basal_presion, std_RoNe_basal_presion, _ = compute_average_and_std(interpolated_presion_RoNe_basal)

# Interpolate the rate data from RoNe_basal using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_RoNe_basal, interpolated_rate_RoNe_basal = interpolate_single_data(
    RoNe_basal, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_RoNe_basal_rate, std_RoNe_basal_rate, _ = compute_average_and_std(interpolated_rate_RoNe_basal)

#%%

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle('RoNe basal')
# ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[0].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel("Rate (Hz)", fontsize=14)
ax[1].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Grafico de todas juntas 
for i in range(len(RoNe_basal)):
    # tiempo_sonido, sonido = RoNe_basal[i]['time'],RoNe_basal[i]['sonido']
    # ax[0].plot(tiempo_sonido,sonido,color='k',alpha=0.1,solid_capstyle='projecting')


    tiempo_presion, presion = RoNe_basal[i]['time maximos'],RoNe_basal[i]['presion']
    ax[0].plot(tiempo_presion,presion,color='k',alpha=0.1,solid_capstyle='projecting')


    timepo_rate, rate = RoNe_basal[i]['time rate'],RoNe_basal[i]['rate']
    ax[1].plot(timepo_rate,rate,color='k',alpha=0.1,solid_capstyle='projecting')
    

# Grafico del promedio 
# ax[0].errorbar(time_sonido_RoNe_basal, average_RoNe_basal_sonido, std_RoNe_basal_sonido, color='C0')
ax[0].errorbar(time_maximos_RoNe_basal, average_RoNe_basal_presion, std_RoNe_basal_presion, color='C0')
ax[1].errorbar(time_rate_RoNe_basal, average_RoNe_basal_rate, std_RoNe_basal_rate, color='C0')


legend_handles = [
    mpatches.Patch(color='C0', label='Average'),
    mpatches.Patch(color='k', label='Data'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')
# ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper right')

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path

directories = [directory]

RoVio_basal = plot_night_data_basal(directories)

# time_sonido_RoVio_basal, interpolated_sonido_RoVio_basal = interpolate_single_data(
#     RoVio_basal, data_key='sonido', time_key='time', common_time_length=44150
# )

# # Compute the average and standard deviation of the interpolated sound data
# average_RoVio_basal_sonido, std_RoVio_basal_sonido, _ = compute_average_and_std(interpolated_sonido_RoVio_basal)

# Interpolate the pressure data from RoVio_basal using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_RoVio_basal, interpolated_presion_RoVio_basal = interpolate_single_data(
    RoVio_basal, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_RoVio_basal_presion, std_RoVio_basal_presion, _ = compute_average_and_std(interpolated_presion_RoVio_basal)

# Interpolate the rate data from RoVio_basal using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_RoVio_basal, interpolated_rate_RoVio_basal = interpolate_single_data(
    RoVio_basal, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_RoVio_basal_rate, std_RoVio_basal_rate, _ = compute_average_and_std(interpolated_rate_RoVio_basal)

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path

directories = [directory]

NaRo_basal = plot_night_data_basal(directories)

# time_sonido_NaRo_basal, interpolated_sonido_NaRo_basal = interpolate_single_data(
#     NaRo_basal, data_key='sonido', time_key='time', common_time_length=44150
# )

# # Compute the average and standard deviation of the interpolated sound data
# average_NaRo_basal_sonido, std_NaRo_basal_sonido, _ = compute_average_and_std(interpolated_sonido_NaRo_basal)

# Interpolate the pressure data from NaRo_basal using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_NaRo_basal, interpolated_presion_NaRo_basal = interpolate_single_data(
    NaRo_basal, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_NaRo_basal_presion, std_NaRo_basal_presion, _ = compute_average_and_std(interpolated_presion_NaRo_basal)

# Interpolate the rate data from NaRo_basal using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_NaRo_basal, interpolated_rate_NaRo_basal = interpolate_single_data(
    NaRo_basal, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_NaRo_basal_rate, std_NaRo_basal_rate, _ = compute_average_and_std(interpolated_rate_NaRo_basal)

#%%

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
fig.suptitle('Datos basales')
# ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[0].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel("Rate (Hz)", fontsize=14)
ax[1].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

ax[0].errorbar(time_maximos_RoNe_basal, average_RoNe_basal_presion, std_RoNe_basal_presion, label='RoNe')
ax[1].errorbar(time_rate_RoNe_basal, average_RoNe_basal_rate, std_RoNe_basal_rate, label='RoNe')

ax[0].errorbar(time_maximos_RoVio_basal, average_RoVio_basal_presion, std_RoVio_basal_presion, label='RoVio')
ax[1].errorbar(time_rate_RoVio_basal, average_RoVio_basal_rate, std_RoVio_basal_rate, label='RoVio')

ax[0].errorbar(time_maximos_NaRo_basal, average_NaRo_basal_presion, std_NaRo_basal_presion, label='NaRo')
ax[1].errorbar(time_rate_NaRo_basal, average_NaRo_basal_rate, std_NaRo_basal_rate, label='NaRo')

# Add legends for each axis
ax[0].legend(fancybox=True, shadow=True)
ax[1].legend(fancybox=True, shadow=True)


#%%

def plot_night_data_4(directories, name):
    # Create a PDF object to save the plots
    # pdf = PdfPages(name)
    
    # Function to normalize the audio and pressure data in a specific time interval [ti, tf]
    def datos_normalizados(Sonidos, Presiones, indice, ti, tf):
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]  # Extract the name from the Pressure file

        # Read audio and pressure files
        fs, audio = wavfile.read(Sound)
        fs, pressure = wavfile.read(Pressure)
       
        # Normalize the audio and pressure data
        audio = audio - np.mean(audio)  # Remove mean
        audio_norm = audio / np.max(audio)  # Normalize by max value
        
        pressure = pressure - np.mean(pressure)  # Remove mean
        pressure_norm = pressure / np.max(pressure)  # Normalize by max value
        
        # Function to normalize the data to [-1, 1] in a given time interval
        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        # Normalize audio and pressure within the interval
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)
    
        return audio_norm, pressure_norm, name, fs
    
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

    # Function to find the longest interval of consecutive time values that are within a gap threshold
    def find_longest_interval(times, max_gap=1):
        longest_interval = []
        current_interval = [times[0]]

        # Iterate through time values and check the gap between consecutive times
        for i in range(1, len(times)):
            if times[i] - times[i-1] <= max_gap:
                current_interval.append(times[i])
            else:
                # If current interval is longer than the previous longest, update it
                if len(current_interval) > len(longest_interval):
                    longest_interval = current_interval
                current_interval = [times[i]]

        # Final check for the last interval
        if len(current_interval) > len(longest_interval):
            longest_interval = current_interval

        return longest_interval
    
    
    
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = []  # To store pressure files
        sonidos = []  # To store sound files
        datos = []  # To store data files
        
        # Classify files into sonidos, presiones, and datos based on their names
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
            
            # Get normalized audio and pressure data for the current index
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
            peaks_sonido_no_limpio, _ = find_peaks(audio, height=0, distance=int(fs * 0.1), prominence=.001)  # Find peaks in raw audio
            
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)
            peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
            # Filter maxima by ensuring there's a minimum between them
            peaks_maximos_filtered = filter_maxima(peaks_maximos, peaks_minimos, pressure)
            
            # Generate a spectrogram of the filtered signal
            f, t, Sxx = signal.spectrogram(filtered_signal, fs)
            
            # Spectrogram processing to find the longest interval where the frequency exceeds a threshold
            frequency_cutoff = 1000  # Frequency cutoff in Hz
            threshold = -10  # Threshold for spectrogram in dB
            Sxx_dB = np.log(Sxx)  # Convert the spectrogram to dB scale
            
            # Find where frequencies exceed the threshold
            freq_indices = np.where(f > frequency_cutoff)[0]
            time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
            time_above_threshold = t[time_indices]
            longest_interval = find_longest_interval(time_above_threshold)
            
            # Calculate periods between maxima (filtered and unfiltered)
            periodo = np.diff(time[peaks_maximos_filtered])
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            periodo_sin_filtrar = np.diff(time[lugar_maximos])
            
            # Create a figure with three subplots (audio, pressure, and rate)
            fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            # fig.suptitle(f'{name}', fontsize=16)  # Title with the name
            ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
            ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
            # ax[2].set_ylabel("Rate (Hz)", fontsize=14)
            ax[1].set_xlabel("Time (s)", fontsize=14)
            
            
            # Plot audio with and without filtering
            # ax[0].plot(time[peaks_sonido_no_limpio], audio[peaks_sonido_no_limpio], color='C0', label='Sin Filtrado')
            ax[0].plot(time, filtered_signal,color='#152534')
            ax[0].plot(time[peaks_sonido], audio[peaks_sonido], color='royalblue', marker='.', linestyle='',ms=11, label='Maximums')
            # ax[0].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[0].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            # ax[0].text(10,3,round(tiempo_inicial,2), fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
            
            # Plot pressure with manually identified maxima and filtered maxima
            ax[1].plot(time, pressure, color='#152534')
            # ax[1].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[1].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[1].plot(time[lugar_maximos], pressure[lugar_maximos], marker='.', linestyle='', color='royalblue', ms=15, label='Maximums')
            # ax[1].plot(time[peaks_maximos_filtered], pressure[peaks_maximos_filtered], '.C1', ms=10, label='con minimos')
            # ax[1].plot(time[peaks_minimos], pressure[peaks_minimos], '.C2', ms=10)
            
            # Plot rate based on periods (filtered and unfiltered)
            # ax[2].plot(time[lugar_maximos][1:], 1 / periodo_sin_filtrar, 'darkslategray', label='A mano')
            # ax[2].plot(time[peaks_maximos_filtered][1:], 1 / periodo, color='C1',linestyle='--',linewidth=2, label='con minimos')
            # ax[2].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[2].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[1].set_xlim(0,20)
            ax[0].set_ylim(-2.5,2.5)
            
            # Add legends
            ax[0].legend(fancybox=True, shadow=True, loc='upper left')
            ax[1].legend(fancybox=True, shadow=True, loc='upper left')
            # ax[2].legend(fancybox=True, shadow=True, loc='upper right')
            plt.tight_layout()
            
            plt.savefig('Basal.pdf')
            
            fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
            # fig.suptitle(f'{name}', fontsize=16)  # Title with the name
            ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
            ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
            # ax[2].set_ylabel("Rate (Hz)", fontsize=14)
            ax[1].set_xlabel("Time (s)", fontsize=14)
            
            
            # Plot audio with and without filtering
            # ax[0].plot(time[peaks_sonido_no_limpio], audio[peaks_sonido_no_limpio], color='C0', label='Sin Filtrado')
            ax[0].plot(time, filtered_signal,color='#152534')
            ax[0].plot(time[peaks_sonido], audio[peaks_sonido], color='royalblue', marker='.', linestyle='',ms=11, label='Maximums')
            # ax[0].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[0].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            # ax[0].text(10,3,round(tiempo_inicial,2), fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
            
            # Plot pressure with manually identified maxima and filtered maxima
            ax[1].plot(time, pressure, color='#152534')
            # ax[1].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[1].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[1].plot(time[lugar_maximos], pressure[lugar_maximos], marker='.', linestyle='', color='royalblue', ms=15, label='Maximums')
            # ax[1].plot(time[peaks_maximos_filtered], pressure[peaks_maximos_filtered], '.C1', ms=10, label='con minimos')
            # ax[1].plot(time[peaks_minimos], pressure[peaks_minimos], '.C2', ms=10)
            
            # Plot rate based on periods (filtered and unfiltered)
            # ax[2].plot(time[lugar_maximos][1:], 1 / periodo_sin_filtrar, 'darkslategray', label='A mano')
            # ax[2].plot(time[peaks_maximos_filtered][1:], 1 / periodo, color='C1',linestyle='--',linewidth=2, label='con minimos')
            # ax[2].axvline(tiempo_inicial, color='k', linestyle='-',linewidth=2, label='Avion sin filtrar')
            # ax[2].axvline(longest_interval[0], color='midnightblue', linestyle='--',linewidth=2, label='Avion con filtrado')
            ax[1].set_xlim(25,45)
            # ax[0].set_ylim(-2.5,2.5)
            
            # Add legends
            ax[0].legend(fancybox=True, shadow=True, loc='upper left')
            ax[1].legend(fancybox=True, shadow=True, loc='upper left')
            # ax[2].legend(fancybox=True, shadow=True, loc='upper right')
            plt.tight_layout()
            # Save the current figure to the PDF
            plt.savefig('Avion.pdf')
            # plt.close(fig)  # Close the figure to free memory
            
    # pdf.close()  # Close the PDF document
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-28-night\poster'
# Store all directories in a list
directories = [directory]

# Process the night data from the directories using the process_night_data function
RoNe_noche = plot_night_data_4(directories,'RoNe plots nuevo vs viejo V3.pdf')