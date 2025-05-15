# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:17:31 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
# import pickle
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
import json
# from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, sosfiltfilt
from scipy import signal
from scipy.interpolate import interp1d
from scipy import stats

def process_night_data(directories, threshold=-10):
      
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
            threshold = threshold #10.5 para NaRo  # Threshold for spectrogram in dB
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
            peaks_maximos, _ = find_peaks(pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Maxima in pressure
            peaks_minimos, _ = find_peaks(-pressure, prominence=1, height=0, distance=int(fs * 0.1))  # Minima in pressure
            
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
            tiempo = time - tiempo_inicial
            # tiempo = time - 30 # esto es si tengo que hacer lo basal
            
            
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time rate': tiempo[lugar_maximos][1:],
                'rate': 1/periodo_sin_filtrar
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






#%% RoNe
########################## RoNe noche ######################################
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


time_sonido_RoNe_noche, interpolated_sonido_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='sonido', time_key='time', common_time_length=300
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


########################## RoNe dia ######################################

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
    RoNe_dia, data_key='sonido', time_key='time', common_time_length=300)#44150

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


########################## RoNe basal ######################################

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path

directories = [directory]

RoNe_basal = plot_night_data_basal(directories)

# ## caclulo la interpolacion del sonido basal
# time_RoNe_basal, interpolated_RoNe_basal = interpolate_single_data(
#     RoNe_basal, data_key='sonido', time_key='time', common_time_length=300
# )

# # Compute the average and standard deviation of the interpolated pressure data
# average_RoNe_basal_sonido, std_RoNe_basal_sonido, _ = compute_average_and_std(interpolated_RoNe_basal)

##
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
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
# os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory)  # List all folders within the directory

# ## RoNe
# pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

# subdirectory = os.path.join(directory, pajaro)

# os.chdir(subdirectory)
# np.savetxt('average RoNe sonido 300 day', np.array([time_sonido_RoNe_dia, average_RoNe_dia_sonido, std_RoNe_dia_sonido]),delimiter=',')
# np.savetxt('average RoNe pressure day', np.array([time_maximos_RoNe_dia, average_RoNe_dia_presion, std_RoNe_dia_presion]),delimiter=',')
# np.savetxt('average RoNe rate day', np.array([time_rate_RoNe_dia, average_RoNe_dia_rate, std_RoNe_dia_rate]),delimiter=',')

#%% RoVio
############################### RoVio noche ####################
# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoVio
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

RoVio_noche = process_night_data(directories)

time_sonido_RoVio_noche, interpolated_sonido_RoVio_noche = interpolate_single_data(
    RoVio_noche, data_key='sonido', time_key='time', common_time_length=300
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

############################### RoVio dia ####################
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia'
os.chdir(directory)
carpetas = os.listdir(directory)
pajaro = carpetas[1]  # RoVio is assumed to be the second folder

subdirectory = os.path.join(directory, pajaro)

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)
pajaritos = '\Aviones\Aviones y pajaros'

# Paths for RoVio day data
dia_1 = subdirectory + '/' + dias[0] + pajaritos
dia_2 = subdirectory + '/' + dias[1] + pajaritos
dia_3 = subdirectory + '/' + dias[2] + pajaritos


directories = [dia_1, dia_2, dia_3]

# Process RoVio day data
RoVio_dia = process_night_data(directories)

# Interpolate the sound data
time_sonido_RoVio_dia, interpolated_sonido_RoVio_dia = interpolate_single_data(
    RoVio_dia, data_key='sonido', time_key='time', common_time_length=300)

# Compute average and standard deviation for the sound data
average_RoVio_dia_sonido, std_RoVio_dia_sonido, _ = compute_average_and_std(interpolated_sonido_RoVio_dia)

# Interpolate the pressure data
time_maximos_RoVio_dia, interpolated_presion_RoVio_dia = interpolate_single_data(
    RoVio_dia, data_key='presion', time_key='time maximos', common_time_length=300)

# Compute average and standard deviation for the pressure data
average_RoVio_dia_presion, std_RoVio_dia_presion, _ = compute_average_and_std(interpolated_presion_RoVio_dia)

# Interpolate the rate data
time_rate_RoVio_dia, interpolated_rate_RoVio_dia = interpolate_single_data(
    RoVio_dia, data_key='rate', time_key='time rate', common_time_length=300)

# Compute average and standard deviation for the rate data
average_RoVio_dia_rate, std_RoVio_dia_rate, _ = compute_average_and_std(interpolated_rate_RoVio_dia)



############################### RoVio basal ####################

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path

directories = [directory]

RoVio_basal = plot_night_data_basal(directories)

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
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
# os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory)  # List all folders within the directory

# ## RoNe
# pajaro = carpetas[1]  # Select the first folder (assumed to be related to 'RoNe')

# subdirectory = os.path.join(directory, pajaro)

# os.chdir(subdirectory)
# np.savetxt('average RoVio sonido 300 day', np.array([time_sonido_RoVio_dia, average_RoVio_dia_sonido, std_RoVio_dia_sonido]),delimiter=',')
# np.savetxt('average RoVio pressure day', np.array([time_maximos_RoVio_dia, average_RoVio_dia_presion, std_RoVio_dia_presion]),delimiter=',')
# np.savetxt('average RoVio rate day', np.array([time_rate_RoVio_dia, average_RoVio_dia_rate, std_RoVio_dia_rate]),delimiter=',')

#%% NaRo

############################# NaRo noche #####################################
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory
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

NaRo_noche = process_night_data(directories,threshold=-10.5)

time_sonido_NaRo_noche, interpolated_sonido_NaRo_noche = interpolate_single_data(
    NaRo_noche, data_key='sonido', time_key='time', common_time_length=300
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


############################# NaRo dia #####################################
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia'
os.chdir(directory)
carpetas = os.listdir(directory)
pajaro = carpetas[2]  # NaRo is assumed to be the third folder

subdirectory = os.path.join(directory, pajaro)

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)
pajaritos = '\Aviones'

# Paths for NaRo day data
dia_1 = subdirectory + '/' + dias[0] + pajaritos
dia_2 = subdirectory + '/' + dias[1] + pajaritos
dia_3 = subdirectory + '/' + dias[2] + pajaritos

directories = [dia_1, dia_2, dia_3]

# Process NaRo day data
NaRo_dia = process_night_data(directories,threshold=-10.5)

# Interpolate the sound data
time_sonido_NaRo_dia, interpolated_sonido_NaRo_dia = interpolate_single_data(
    NaRo_dia, data_key='sonido', time_key='time', common_time_length=300)

# Compute average and standard deviation for the sound data
average_NaRo_dia_sonido, std_NaRo_dia_sonido, _ = compute_average_and_std(interpolated_sonido_NaRo_dia)

# Interpolate the pressure data
time_maximos_NaRo_dia, interpolated_presion_NaRo_dia = interpolate_single_data(
    NaRo_dia, data_key='presion', time_key='time maximos', common_time_length=300)

# Compute average and standard deviation for the pressure data
average_NaRo_dia_presion, std_NaRo_dia_presion, _ = compute_average_and_std(interpolated_presion_NaRo_dia)

# Interpolate the rate data
time_rate_NaRo_dia, interpolated_rate_NaRo_dia = interpolate_single_data(
    NaRo_dia, data_key='rate', time_key='time rate', common_time_length=300)

# Compute average and standard deviation for the rate data
average_NaRo_dia_rate, std_NaRo_dia_rate, _ = compute_average_and_std(interpolated_rate_NaRo_dia)


############################# NaRo basal #####################################

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\Datos basal\Basal'
os.chdir(directory)  # Change the working directory to the specified path

directories = [directory]

NaRo_basal = plot_night_data_basal(directories)

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

# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
# os.chdir(directory)  # Change the working directory to the specified path
# carpetas = os.listdir(directory)  # List all folders within the directory

# ## RoNe
# pajaro = carpetas[2]  # Select the first folder (assumed to be related to 'RoNe')

# subdirectory = os.path.join(directory, pajaro)

# os.chdir(subdirectory)
# np.savetxt('average NaRo sonido 300 day', np.array([time_sonido_NaRo_dia, average_NaRo_dia_sonido, std_NaRo_dia_sonido]),delimiter=',')
# np.savetxt('average NaRo pressure day', np.array([time_maximos_NaRo_dia, average_NaRo_dia_presion, std_NaRo_dia_presion]),delimiter=',')
# np.savetxt('average NaRo rate day', np.array([time_rate_NaRo_dia, average_NaRo_dia_rate, std_NaRo_dia_rate]),delimiter=',')



#%% Grafico de todos con el promedio

fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
# fig.suptitle('RoNe')
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Grafico de todas juntas 
for i in range(len(RoNe_noche)):
    tiempo_sonido, sonido = RoNe_noche[i]['time'],RoNe_noche[i]['sonido']
    ax[0].plot(tiempo_sonido,sonido,color='#152534',alpha=0.05,solid_capstyle='projecting')


    tiempo_presion, presion = RoNe_noche[i]['time maximos'],RoNe_noche[i]['presion']
    ax[1].plot(tiempo_presion,presion,color='#152534',alpha=0.05,solid_capstyle='projecting')


    timepo_rate, rate = RoNe_noche[i]['time rate'],RoNe_noche[i]['rate']
    ax[2].plot(timepo_rate,rate,color='#152534',alpha=0.05,solid_capstyle='projecting')
    

# Grafico del promedio 
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='royalblue')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='royalblue')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color='royalblue')

ax[0].set_ylim(-0.5,10)
ax[1].set_ylim(-0.5,5)
ax[2].set_ylim(-0.5,4)

legend_handles = [
    mpatches.Patch(color='royalblue', label='Average'),
    mpatches.Patch(color='#152534', label='Data'),
]
ax[0].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper left')
ax[1].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper left')
ax[2].legend(handles=legend_handles, fancybox=True, shadow=True,loc='upper left')

# plt.savefig('promedio RoNe.pdf')
#%% Promedios 3 noches juntas
colors = ['#0E2862','#2F4F4F','#152534']
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
fig.suptitle('Promedios')
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# RoNe Data
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='royalblue', label='RoNe')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='royalblue', label='RoNe')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate/np.mean(average_RoNe_noche_rate[:100]), std_RoNe_noche_rate, color='royalblue', label='RoNe')

# RoVio Data
ax[0].errorbar(time_sonido_RoVio_noche, average_RoVio_noche_sonido, std_RoVio_noche_sonido, color=colors[1], label='RoVio')
ax[1].errorbar(time_maximos_RoVio_noche, average_RoVio_noche_presion, std_RoVio_noche_presion, color=colors[1], label='RoVio')
ax[2].errorbar(time_rate_RoVio_noche, average_RoVio_noche_rate/np.mean(average_RoVio_noche_rate[:100]), std_RoVio_noche_rate, color=colors[1], label='RoVio')

# NaRo Data
ax[0].errorbar(time_sonido_NaRo_noche, average_NaRo_noche_sonido, std_NaRo_noche_sonido, color=colors[2], label='NaRo')
ax[1].errorbar(time_maximos_NaRo_noche, average_NaRo_noche_presion, std_NaRo_noche_presion, color=colors[2], label='NaRo')
ax[2].errorbar(time_rate_NaRo_noche, average_NaRo_noche_rate/np.mean(average_NaRo_noche_rate[:100]), std_NaRo_noche_rate, color=colors[2], label='NaRo')

# Add legends for each axis
ax[0].legend(fancybox=True, shadow=True)
ax[1].legend(fancybox=True, shadow=True)
ax[2].legend(fancybox=True, shadow=True)


#%% Promedio noche y dia RoNe

interpolated_rate_on_day_base = interpolate_to_target_time_base(
    time_rate_RoNe_noche, interpolated_rate_RoNe_noche, time_rate_RoNe_dia
)
rate_interpolated_rate_noche_cortado, rate_std_interpolated_rate_noche_cortado, _ = compute_average_and_std(interpolated_rate_on_day_base)

p_values_rate = []

indice_rate = np.argmin(abs(time_rate_RoNe_dia - 0))
# Iterate over the time points
for i in range(len(time_rate_RoNe_dia)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_RoNe_dia[:, i]/np.mean(average_RoNe_dia_rate[0:indice_rate]), interpolated_rate_on_day_base[:, i]/np.nanmean(rate_interpolated_rate_noche_cortado[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_rate = np.array(p_values_rate)



interpolated_presion_on_day_base = interpolate_to_target_time_base(
    time_maximos_RoNe_noche, interpolated_presion_RoNe_noche, time_maximos_RoNe_dia
)

# Compute the average and standard deviation for the interpolated presion data
presion_interpolated_presion_noche_cortado, presion_std_interpolated_presion_noche_cortado, _ = compute_average_and_std(interpolated_presion_on_day_base)

p_values_presion = []

indice_maximos = np.argmin(abs(time_maximos_RoNe_dia - 0))
# Iterate over the time points for presion data
for i in range(len(time_maximos_RoNe_dia)):
    # Perform t-test between corresponding data columns from interpolated day and night data
    _, p_value = stats.ttest_ind(interpolated_presion_RoNe_dia[:, i]/np.mean(average_RoNe_dia_presion[0:indice_maximos]), 
                                 interpolated_presion_on_day_base[:, i]/np.nanmean(presion_interpolated_presion_noche_cortado[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_presion = np.array(p_values_presion)

colors = ['#0E2862','#2F4F4F','#152534']
fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)
ax[0].set_ylabel("Presion (u. a.)", fontsize=20)
ax[1].set_ylabel("Rate (Hz)", fontsize=20)
ax[1].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)

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
ax[0].legend(fancybox=True,shadow=True,loc='upper left',fontsize=12,prop={'size': 20})
plt.tight_layout()
# ax[1].legend(fancybox=True,shadow=True,loc='upper left')
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.951), bbox_transform=fig.transFigure, ncol=3, fontsize=12)
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path
# plt.savefig('Noche vs dia vs basal.pdf')

#%% Promedio noche y dia RoVio
# Interpolating RoVio rate data on day base
interpolated_rate_on_day_base_RoVio = interpolate_to_target_time_base(
    time_rate_RoVio_noche, interpolated_rate_RoVio_noche, time_rate_RoVio_dia
)
rate_interpolated_rate_noche_cortado_RoVio, rate_std_interpolated_rate_noche_cortado_RoVio, _ = compute_average_and_std(interpolated_rate_on_day_base_RoVio)

p_values_rate_RoVio = []

indice_rate = np.argmin(abs(time_rate_RoVio_dia - 0))
# Iterate over the time points for rate data
for i in range(len(time_rate_RoVio_dia)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_RoVio_dia[:, i]/np.mean(average_RoVio_dia_rate[0:indice_rate]), 
                                 interpolated_rate_on_day_base_RoVio[:, i]/np.nanmean(rate_interpolated_rate_noche_cortado_RoVio[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate_RoVio.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_rate_RoVio = np.array(p_values_rate_RoVio)

# Interpolating RoVio pressure data on day base
interpolated_presion_on_day_base_RoVio = interpolate_to_target_time_base(
    time_maximos_RoVio_noche, interpolated_presion_RoVio_noche, time_maximos_RoVio_dia
)

# Compute the average and standard deviation for the interpolated pressure data
presion_interpolated_presion_noche_cortado_RoVio, presion_std_interpolated_presion_noche_cortado_RoVio, _ = compute_average_and_std(interpolated_presion_on_day_base_RoVio)

p_values_presion_RoVio = []

indice_maximos = np.argmin(abs(time_maximos_RoVio_dia - 0))
# Iterate over the time points for pressure data
for i in range(len(time_maximos_RoVio_dia)):
    # Perform t-test between corresponding data columns from interpolated day and night data
    _, p_value = stats.ttest_ind(interpolated_presion_RoVio_dia[:, i]/np.mean(average_RoVio_dia_presion[0:indice_maximos]), 
                                 interpolated_presion_on_day_base_RoVio[:, i]/np.nanmean(presion_interpolated_presion_noche_cortado_RoVio[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion_RoVio.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_presion_RoVio = np.array(p_values_presion_RoVio)

# Plotting results for RoVio
fig, ax = plt.subplots(4,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"Rate (arb. u.)", fontsize=14)
ax[2].set_ylabel(r"P-value", fontsize=14)
ax[3].set_ylabel(r"P-value", fontsize=14)
ax[3].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[1].errorbar(time_rate_RoVio_dia, average_RoVio_dia_rate/np.mean(average_RoVio_dia_rate[0:indice_rate]), yerr= std_RoVio_dia_rate, color='C0', label='Dia') 
ax[1].errorbar(time_rate_RoVio_dia,rate_interpolated_rate_noche_cortado_RoVio/np.nanmean(rate_interpolated_rate_noche_cortado_RoVio[0:indice_rate]), yerr=rate_std_interpolated_rate_noche_cortado_RoVio,color='C2', label='Noche')
ax[0].errorbar(time_maximos_RoVio_dia, average_RoVio_dia_presion/np.mean(average_RoVio_dia_presion[0:indice_maximos]),yerr= std_RoVio_dia_presion, color='C0', label='Dia')
ax[0].errorbar(time_maximos_RoVio_dia,presion_interpolated_presion_noche_cortado_RoVio/np.nanmean(presion_interpolated_presion_noche_cortado_RoVio[0:indice_maximos]), yerr=presion_std_interpolated_presion_noche_cortado_RoVio,color='C2', label='Noche')
ax[3].plot(time_rate_RoVio_dia, p_values_rate_RoVio)
ax[2].plot(time_maximos_RoVio_dia, p_values_presion_RoVio)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)



#%% Promedio noche y dia NaRo

# Interpolating NaRo rate data on day base
interpolated_rate_on_day_base_NaRo = interpolate_to_target_time_base(
    time_rate_NaRo_noche, interpolated_rate_NaRo_noche, time_rate_NaRo_dia
)
rate_interpolated_rate_noche_cortado_NaRo, rate_std_interpolated_rate_noche_cortado_NaRo, _ = compute_average_and_std(interpolated_rate_on_day_base_NaRo)

p_values_rate_NaRo = []

indice_rate = np.argmin(abs(time_rate_NaRo_dia - 0))
# Iterate over the time points for rate data
for i in range(len(time_rate_NaRo_dia)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_NaRo_dia[:, i]/np.mean(average_NaRo_dia_rate[0:indice_rate]), 
                                 interpolated_rate_on_day_base_NaRo[:, i]/np.nanmean(rate_interpolated_rate_noche_cortado_NaRo[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate_NaRo.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_rate_NaRo = np.array(p_values_rate_NaRo)

# Interpolating NaRo pressure data on day base
interpolated_presion_on_day_base_NaRo = interpolate_to_target_time_base(
    time_maximos_NaRo_noche, interpolated_presion_NaRo_noche, time_maximos_NaRo_dia
)

# Compute the average and standard deviation for the interpolated pressure data
presion_interpolated_presion_noche_cortado_NaRo, presion_std_interpolated_presion_noche_cortado_NaRo, _ = compute_average_and_std(interpolated_presion_on_day_base_NaRo)

p_values_presion_NaRo = []

indice_maximos = np.argmin(abs(time_maximos_NaRo_dia - 0))
# Iterate over the time points for pressure data
for i in range(len(time_maximos_NaRo_dia)):
    # Perform t-test between corresponding data columns from interpolated day and night data
    _, p_value = stats.ttest_ind(interpolated_presion_NaRo_dia[:, i]/np.mean(average_NaRo_dia_presion[0:indice_maximos]), 
                                 interpolated_presion_on_day_base_NaRo[:, i]/np.nanmean(presion_interpolated_presion_noche_cortado_NaRo[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion_NaRo.append(p_value)

# Convert p-values list to a numpy array if needed
p_values_presion_NaRo = np.array(p_values_presion_NaRo)

# Plotting results for NaRo
fig, ax = plt.subplots(4,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"Rate (arb. u.)", fontsize=14)
ax[2].set_ylabel(r"P-value", fontsize=14)
ax[3].set_ylabel(r"P-value", fontsize=14)
ax[3].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[1].errorbar(time_rate_NaRo_dia, average_NaRo_dia_rate/np.mean(average_NaRo_dia_rate[0:indice_rate]), yerr= std_NaRo_dia_rate, color='C0', label='Dia') 
ax[1].errorbar(time_rate_NaRo_dia,rate_interpolated_rate_noche_cortado_NaRo/np.nanmean(rate_interpolated_rate_noche_cortado_NaRo[0:indice_rate]), yerr=rate_std_interpolated_rate_noche_cortado_NaRo,color='C2', label='Noche')
ax[0].errorbar(time_maximos_NaRo_dia, average_NaRo_dia_presion/np.mean(average_NaRo_dia_presion[0:indice_maximos]),yerr= std_NaRo_dia_presion, color='C0', label='Dia')
ax[0].errorbar(time_maximos_NaRo_dia,presion_interpolated_presion_noche_cortado_NaRo/np.nanmean(presion_interpolated_presion_noche_cortado_NaRo[0:indice_maximos]), yerr=presion_std_interpolated_presion_noche_cortado_NaRo,color='C2', label='Noche')
ax[3].plot(time_rate_NaRo_dia, p_values_rate_NaRo)
ax[2].plot(time_maximos_NaRo_dia, p_values_presion_NaRo)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)

#%% Promedio Noche y basal RoNe

# Interpolating the rate data for comparison between night and basal
interpolated_rate_on_basal_base = interpolate_to_target_time_base(
    time_rate_RoNe_noche, interpolated_rate_RoNe_noche, time_rate_RoNe_basal
)
rate_interpolated_rate_basal_cortado, rate_std_interpolated_rate_basal_cortado, _ = compute_average_and_std(interpolated_rate_on_basal_base)

p_values_rate = []

indice_rate = np.argmin(abs(time_rate_RoNe_basal - 0))
# Iterate over the time points
for i in range(len(time_rate_RoNe_basal)):
    # Perform t-test between corresponding data columns from interpolated and expanded data
    _, p_value = stats.ttest_ind(interpolated_rate_RoNe_basal[:, i]/np.mean(average_RoNe_basal_rate[0:indice_rate]), interpolated_rate_on_basal_base[:, i]/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

p_values_rate = np.array(p_values_rate)


# Interpolating the pressure data for comparison between night and basal
interpolated_presion_on_basal_base = interpolate_to_target_time_base(
    time_maximos_RoNe_noche, interpolated_presion_RoNe_noche, time_maximos_RoNe_basal
)

# Compute the average and standard deviation for the interpolated pressure data
presion_interpolated_presion_basal_cortado, presion_std_interpolated_presion_basal_cortado, _ = compute_average_and_std(interpolated_presion_on_basal_base)

p_values_presion = []

indice_maximos = np.argmin(abs(time_maximos_RoNe_basal - 0))
# Iterate over the time points for pressure data
for i in range(len(time_maximos_RoNe_basal)):
    # Perform t-test between corresponding data columns from interpolated night and basal data
    _, p_value = stats.ttest_ind(interpolated_presion_RoNe_basal[:, i]/np.mean(average_RoNe_basal_presion[0:indice_maximos]), 
                                 interpolated_presion_on_basal_base[:, i]/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion.append(p_value)

p_values_presion = np.array(p_values_presion)

# Plotting the results for the comparison between night and basal
fig, ax = plt.subplots(2,1,figsize=(15,9),sharex=True)
# fig.suptitle('RoNe')
ax[0].set_ylabel("Presion (u. a.)", fontsize=20)
ax[1].set_ylabel("Rate (Hz)", fontsize=20)
ax[1].set_xlabel("Tiempo (s)", fontsize=20)
ax[0].tick_params(axis='both', labelsize=10)
ax[1].tick_params(axis='both', labelsize=10)
# Plot for rate data comparison between night and basal
ax[1].errorbar(time_rate_RoNe_basal, average_RoNe_basal_rate/np.mean(average_RoNe_basal_rate[0:indice_rate]), yerr= std_RoNe_basal_rate, color=colors[1], label='Basal') 
ax[1].errorbar(time_rate_RoNe_basal,rate_interpolated_rate_basal_cortado/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), yerr=rate_std_interpolated_rate_basal_cortado,color='royalblue', label='Noche')
ax[1].axvspan(0,time_rate_RoNe_basal[-1], facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--', label='P-value < 0.05')
# Plot for pressure data comparison between night and basal
ax[0].errorbar(time_maximos_RoNe_basal, average_RoNe_basal_presion/np.mean(average_RoNe_basal_presion[0:indice_maximos]), yerr= std_RoNe_basal_presion, color=colors[1], label='Basal')
ax[0].errorbar(time_maximos_RoNe_basal,presion_interpolated_presion_basal_cortado/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), yerr=presion_std_interpolated_presion_basal_cortado,color='royalblue', label='Noche')
ax[0].axvspan(-0.64,time_maximos_RoNe_basal[-1], facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--', label='P-value < 0.05')
ax[1].set_xlim(-26,26)
ax[0].legend(fancybox=True,shadow=True,fontsize=12,prop={'size': 20})
plt.tight_layout()
# ax[0].spines.right.set_visible(False)
# ax[0].spines.top.set_visible(False)
# ax[1].spines.right.set_visible(False)
# ax[1].spines.top.set_visible(False)
# Plot p-values for rate and pressure
# ax[3].plot(time_rate_RoNe_basal, p_values_rate)
# ax[2].plot(time_maximos_RoNe_basal, p_values_presion)

# ax[1].errorbar(time_rate_RoNe_dia, average_RoNe_dia_rate/np.mean(average_RoNe_dia_rate[0:indice_rate]), yerr= std_RoNe_dia_rate, color='C0', label='Dia') 
# ax[1].errorbar(time_rate_RoNe_dia,rate_interpolated_rate_noche_cortado/np.nanmean(rate_interpolated_rate_noche_cortado[0:indice_rate]), yerr=rate_std_interpolated_rate_noche_cortado,color='C2', label='Noche')
# ax[0].errorbar(time_maximos_RoNe_dia, average_RoNe_dia_presion/np.mean(average_RoNe_dia_presion[0:indice_maximos]),yerr= std_RoNe_dia_presion, color='C0', label='Dia')
# ax[0].errorbar(time_maximos_RoNe_dia,presion_interpolated_presion_noche_cortado/np.nanmean(presion_interpolated_presion_noche_cortado[0:indice_maximos]), yerr=presion_std_interpolated_presion_noche_cortado,color='C2', label='Noche')
# ax[3].plot(time_rate_RoNe_dia, p_values_rate)
# ax[2].plot(time_maximos_RoNe_dia, p_values_presion)
# plt.tight_layout()
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.951), bbox_transform=fig.transFigure, ncol=3, fontsize=12)
# # Legends

# ax[1].legend(fancybox=True,shadow=True)
# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Poster'
# os.chdir(directory)  # Change the working directory to the specified path

# plt.savefig('Noche vs basal.pdf')
#%% Promedio noche y basal RoVio

# Interpolating the rate data for comparison between RoVio night and basal
interpolated_rate_on_basal_base = interpolate_to_target_time_base(
    time_rate_RoVio_noche, interpolated_rate_RoVio_noche, time_rate_RoVio_basal
)
rate_interpolated_rate_basal_cortado, rate_std_interpolated_rate_basal_cortado, _ = compute_average_and_std(interpolated_rate_on_basal_base)

p_values_rate = []

indice_rate = np.argmin(abs(time_rate_RoVio_basal - 0))
# Iterate over the time points
for i in range(len(time_rate_RoVio_basal)):
    _, p_value = stats.ttest_ind(interpolated_rate_RoVio_basal[:, i]/np.mean(average_RoVio_basal_rate[0:indice_rate]), interpolated_rate_on_basal_base[:, i]/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

p_values_rate = np.array(p_values_rate)

# Interpolating the pressure data for comparison between RoVio night and basal
interpolated_presion_on_basal_base = interpolate_to_target_time_base(
    time_maximos_RoVio_noche, interpolated_presion_RoVio_noche, time_maximos_RoVio_basal
)

presion_interpolated_presion_basal_cortado, presion_std_interpolated_presion_basal_cortado, _ = compute_average_and_std(interpolated_presion_on_basal_base)

p_values_presion = []

indice_maximos = np.argmin(abs(time_maximos_RoVio_basal - 0))
for i in range(len(time_maximos_RoVio_basal)):
    _, p_value = stats.ttest_ind(interpolated_presion_RoVio_basal[:, i]/np.mean(average_RoVio_basal_presion[0:indice_maximos]), 
                                 interpolated_presion_on_basal_base[:, i]/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion.append(p_value)

p_values_presion = np.array(p_values_presion)

# Plotting RoVio Night vs Basal
fig, ax = plt.subplots(4,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"Rate (arb. u.)", fontsize=14)
ax[2].set_ylabel(r"P-value", fontsize=14)
ax[3].set_ylabel(r"P-value", fontsize=14)
ax[3].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[1].errorbar(time_rate_RoVio_basal, average_RoVio_basal_rate/np.mean(average_RoVio_basal_rate[0:indice_rate]), yerr= std_RoVio_basal_rate, color='C0', label='Basal') 
ax[1].errorbar(time_rate_RoVio_basal,rate_interpolated_rate_basal_cortado/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), yerr=rate_std_interpolated_rate_basal_cortado,color='C2', label='Noche')
ax[0].errorbar(time_maximos_RoVio_basal, average_RoVio_basal_presion/np.mean(average_RoVio_basal_presion[0:indice_maximos]), yerr= std_RoVio_basal_presion, color='C0', label='Basal')
ax[0].errorbar(time_maximos_RoVio_basal,presion_interpolated_presion_basal_cortado/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), yerr=presion_std_interpolated_presion_basal_cortado,color='C2', label='Noche')
ax[3].plot(time_rate_RoVio_basal, p_values_rate)
ax[2].plot(time_maximos_RoVio_basal, p_values_presion)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)


#%% Promedio NaRo y basal

# Interpolating the rate data for comparison between NaRo night and basal
interpolated_rate_on_basal_base = interpolate_to_target_time_base(
    time_rate_NaRo_noche, interpolated_rate_NaRo_noche, time_rate_NaRo_basal
)
rate_interpolated_rate_basal_cortado, rate_std_interpolated_rate_basal_cortado, _ = compute_average_and_std(interpolated_rate_on_basal_base)

p_values_rate = []

indice_rate = np.argmin(abs(time_rate_NaRo_basal - 0))
for i in range(len(time_rate_NaRo_basal)):
    _, p_value = stats.ttest_ind(interpolated_rate_NaRo_basal[:, i]/np.mean(average_NaRo_basal_rate[0:indice_rate]), interpolated_rate_on_basal_base[:, i]/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), 
                                 equal_var=False, nan_policy='omit')
    p_values_rate.append(p_value)

p_values_rate = np.array(p_values_rate)

# Interpolating the pressure data for comparison between NaRo night and basal
interpolated_presion_on_basal_base = interpolate_to_target_time_base(
    time_maximos_NaRo_noche, interpolated_presion_NaRo_noche, time_maximos_NaRo_basal
)

presion_interpolated_presion_basal_cortado, presion_std_interpolated_presion_basal_cortado, _ = compute_average_and_std(interpolated_presion_on_basal_base)

p_values_presion = []

indice_maximos = np.argmin(abs(time_maximos_NaRo_basal - 0))
for i in range(len(time_maximos_NaRo_basal)):
    _, p_value = stats.ttest_ind(interpolated_presion_NaRo_basal[:, i]/np.mean(average_NaRo_basal_presion[0:indice_maximos]), 
                                 interpolated_presion_on_basal_base[:, i]/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), 
                                 equal_var=False, nan_policy='omit')
    p_values_presion.append(p_value)

p_values_presion = np.array(p_values_presion)

# Plotting NaRo Night vs Basal
fig, ax = plt.subplots(4,1,figsize=(14,7),sharex=True)
ax[0].set_ylabel(r"Pressure (arb. u.)", fontsize=14)
ax[1].set_ylabel(r"Rate (Hz)", fontsize=14)
ax[2].set_ylabel(r"P-value", fontsize=14)
ax[3].set_ylabel(r"P-value", fontsize=14)
ax[3].set_xlabel(r"Time (s)", fontsize=14)
plt.tight_layout()

ax[1].errorbar(time_rate_NaRo_basal, average_NaRo_basal_rate/np.mean(average_NaRo_basal_rate[0:indice_rate]), yerr= std_NaRo_basal_rate, color='C0', label='Basal') 
ax[1].errorbar(time_rate_NaRo_basal,rate_interpolated_rate_basal_cortado/np.nanmean(rate_interpolated_rate_basal_cortado[0:indice_rate]), yerr=rate_std_interpolated_rate_basal_cortado,color='C2', label='Noche')
ax[0].errorbar(time_maximos_NaRo_basal, average_NaRo_basal_presion/np.mean(average_NaRo_basal_presion[0:indice_maximos]), yerr= std_NaRo_basal_presion, color='C0', label='Basal')
ax[0].errorbar(time_maximos_NaRo_basal,presion_interpolated_presion_basal_cortado/np.nanmean(presion_interpolated_presion_basal_cortado[0:indice_maximos]), yerr=presion_std_interpolated_presion_basal_cortado,color='C2', label='Noche')
ax[3].plot(time_rate_NaRo_basal, p_values_rate)
ax[2].plot(time_maximos_NaRo_basal, p_values_presion)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)

#%%

from scipy import stats
fig, ax = plt.subplots(3,1,figsize=(14, 7),sharex=True)
fig.suptitle('Histogram of Noche, Day and Basal Time Series')
ax[2].set_xlabel(r'Pressure (arb. u.)')
ax[0].set_ylabel('Frequency')
ax[1].set_ylabel('Frequency')
ax[2].set_ylabel('Frequency')

# Plot the histogram of interpolated presion NaRo noche
ax[2].hist(interpolated_presion_NaRo_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo Noche')
ax[1].hist(interpolated_presion_RoVio_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio Noche')
ax[0].hist(interpolated_presion_RoNe_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe Noche')

# Plot the histogram of interpolated presion NaRo dia
ax[2].hist(interpolated_presion_NaRo_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo dia')
ax[1].hist(interpolated_presion_RoVio_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio dia')
ax[0].hist(interpolated_presion_RoNe_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe dia')


# Plot the histogram of time maximos NaRo basal
ax[2].hist(interpolated_presion_NaRo_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo Basal')
ax[1].hist(interpolated_presion_RoVio_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio Basal')
ax[0].hist(interpolated_presion_RoNe_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe Basal')

mean_NaRo = np.nanmedian(interpolated_presion_NaRo_basal.flatten())
mean_RoVio = np.nanmedian(interpolated_presion_RoVio_basal.flatten())
mean_RoNe = np.nanmedian(interpolated_presion_RoNe_basal.flatten())

# Plot the mean as vertical lines if the means are within visible range
ax[2].axvline(mean_NaRo, color='green', linestyle='dashed', linewidth=2, label=f'NaRo basal Mean: {mean_NaRo:.2f}')
ax[1].axvline(mean_RoVio, color='green', linestyle='dashed', linewidth=2, label=f'RoVio basal Mean: {mean_RoVio:.2f}')
ax[0].axvline(mean_RoNe, color='green', linestyle='dashed', linewidth=2, label=f'RoNe basal Mean: {mean_RoNe:.2f}')

ax[2].set_xlim(-1,5)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)
ax[2].legend(fancybox=True,shadow=True)

fig, ax = plt.subplots(3,1,figsize=(14, 7),sharex=True)
fig.suptitle('Histogram of Noche, Day and Basal Time Series')
ax[2].set_xlabel(r'Rate (Hz)')
ax[0].set_ylabel('Frequency')
ax[1].set_ylabel('Frequency')
ax[2].set_ylabel('Frequency')

# Plot the histogram of interpolated presion NaRo noche
ax[2].hist(interpolated_rate_NaRo_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo Noche')
ax[1].hist(interpolated_rate_RoVio_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio Noche')
ax[0].hist(interpolated_rate_RoNe_noche.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe Noche')

# Plot the histogram of interpolated presion NaRo dia
ax[2].hist(interpolated_rate_NaRo_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo dia')
ax[1].hist(interpolated_rate_RoVio_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio dia')
ax[0].hist(interpolated_rate_RoNe_dia.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe dia')


# Plot the histogram of time maximos NaRo basal
ax[2].hist(interpolated_rate_NaRo_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='NaRo Basal')
ax[1].hist(interpolated_rate_RoVio_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='RoVio Basal')
ax[0].hist(interpolated_rate_RoNe_basal.flatten(), bins=50, histtype='step', alpha=0.5, label='RoNe Basal')

mean_NaRo = np.nanmedian(interpolated_rate_NaRo_basal.flatten())
mean_RoVio = np.nanmedian(interpolated_rate_RoVio_basal.flatten())
mean_RoNe = np.nanmedian(interpolated_rate_RoNe_basal.flatten())

filtered_rate = interpolated_rate_NaRo_basal.flatten()
filtered_rate = filtered_rate[~np.isnan(filtered_rate)]

# Now calculate the mode
mode_value = stats.mode(filtered_rate)

# Plot the mean as vertical lines if the means are within visible range
ax[2].axvline(mean_NaRo, color='green', linestyle='dashed', linewidth=2, label=f'NaRo basal Mean: {mean_NaRo:.2f}')
ax[1].axvline(mean_RoVio, color='green', linestyle='dashed', linewidth=2, label=f'RoVio basal Mean: {mean_RoVio:.2f}')
ax[0].axvline(mean_RoNe, color='green', linestyle='dashed', linewidth=2, label=f'RoNe basal Mean: {mean_RoNe:.2f}')

ax[2].set_xlim(-1,4)
ax[0].legend(fancybox=True,shadow=True)
ax[1].legend(fancybox=True,shadow=True)
ax[2].legend(fancybox=True,shadow=True)
#%% Histograma de una cantidad que quiera de datos
time_range = range(indice_maximos)  # You can change this range if you want a different number of points

plt.figure(figsize=(10, 6))

# Loop over the first 20 time points and plot histograms for each
for i in time_range:
    plt.hist(interpolated_presion_NaRo_noche[:, i], bins=50, color='C0', alpha=0.5, label=f'NaRo Noche - Time {i}')
    

plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histograms for the First 20 Time Points in NaRo Noche and Basal')
# plt.legend()

#%% 3 regiones
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


# Plot for ax[1] with time_maximos_RoNe
ax[1].axvspan(time_sonido_RoNe_noche[0], 0, facecolor='g', alpha=0.3,edgecolor='k',linestyle='--',label='Before')
ax[1].axvspan(0, time_at_value, facecolor='b', alpha=0.3,edgecolor='k',linestyle='--',label='During')
ax[1].axvspan(time_at_value, time_sonido_RoNe_noche[-1], facecolor='r', alpha=0.3,edgecolor='k',linestyle='--',label='After')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0')
ax[1].text(20,3, f"{round(100*(value_at_time_presion-value_at_time_presion_0)/value_at_time_presion_0,2)}%" , fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[1].legend(fancybox=True,shadow=True)



# Plot for ax[2] with time_rate_RoNe
ax[2].axvspan(time_sonido_RoNe_noche[0], 0, facecolor='g', alpha=0.3,edgecolor='k',linestyle='--',label='Before')
ax[2].axvspan(0, time_at_value, facecolor='b', alpha=0.3,edgecolor='k',linestyle='--',label='During')
ax[2].axvspan(time_at_value, time_sonido_RoNe_noche[-1], facecolor='r', alpha=0.3,edgecolor='k',linestyle='--',label='After')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color='C0')
ax[2].text(20,2, f"{round(100*(value_at_time_rate-value_at_time_rate_0)/value_at_time_rate_0,2)}%" , fontsize=12, bbox=dict(facecolor='k', alpha=0.1))

ax[2].legend(fancybox=True,shadow=True)

#%% RoVio 3 regiones
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Find the index at time 0
index_at_0 = (np.abs(time_sonido_RoVio_noche - 0)).argmin()
# Get the value of the sound at time 0
value_at_0 = average_RoVio_noche_sonido[index_at_0]

# Exclude the point at index 0 and find the closest index to value_at_0
excluded_indices = np.arange(len(average_RoVio_noche_sonido)) != index_at_0
closest_index_not_0 = (np.abs(average_RoVio_noche_sonido[excluded_indices] - value_at_0)).argmin()

# Convert back to original index space
actual_closest_index = np.arange(len(average_RoVio_noche_sonido))[excluded_indices][closest_index_not_0]

# Get the time at that index
time_at_value = time_sonido_RoVio_noche[actual_closest_index]

# Find the index in time_maximos_RoVio_noche closest to time_at_value (for pressure)
index_at_value_presion = (np.abs(time_maximos_RoVio_noche - time_at_value)).argmin()
value_at_time_presion = average_RoVio_noche_presion[index_at_value_presion]

index_at_value_presion_0 = (np.abs(time_maximos_RoVio_noche - 0)).argmin()
value_at_time_presion_0 = average_RoVio_noche_presion[index_at_value_presion_0]

# Find the index in time_rate_RoVio_noche closest to time_at_value (for rate)
index_at_value_rate = (np.abs(time_rate_RoVio_noche - time_at_value)).argmin()
value_at_time_rate = average_RoVio_noche_rate[index_at_value_rate]

index_at_value_rate_0 = (np.abs(time_rate_RoVio_noche - 0)).argmin()
value_at_time_rate_0 = average_RoVio_noche_rate[index_at_value_rate_0]

# Plot for ax[0] with time_sonido_RoVio
ax[0].axvspan(time_sonido_RoVio_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[0].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[0].axvspan(time_at_value, time_sonido_RoVio_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[0].errorbar(time_sonido_RoVio_noche, average_RoVio_noche_sonido, std_RoVio_noche_sonido, color='C0')
ax[0].text(20, 3, f"{round(time_at_value, 2)}s", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[0].legend(fancybox=True, shadow=True)

# Plot for ax[1] with time_maximos_RoVio
ax[1].axvspan(time_sonido_RoVio_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[1].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[1].axvspan(time_at_value, time_sonido_RoVio_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[1].errorbar(time_maximos_RoVio_noche, average_RoVio_noche_presion, std_RoVio_noche_presion, color='C0')
ax[1].text(20, 3, f"{round(100 * (value_at_time_presion - value_at_time_presion_0) / value_at_time_presion_0, 2)}%", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[1].legend(fancybox=True, shadow=True)

# Plot for ax[2] with time_rate_RoVio
ax[2].axvspan(time_sonido_RoVio_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[2].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[2].axvspan(time_at_value, time_sonido_RoVio_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[2].errorbar(time_rate_RoVio_noche, average_RoVio_noche_rate, std_RoVio_noche_rate, color='C0')
ax[2].text(20, 2, f"{round(100 * (value_at_time_rate - value_at_time_rate_0) / value_at_time_rate_0, 2)}%", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[2].legend(fancybox=True, shadow=True)

ax[0].spines.right.set_visible(False)
ax[0].spines.top.set_visible(False)
ax[1].spines.right.set_visible(False)
ax[1].spines.top.set_visible(False)
ax[2].spines.right.set_visible(False)
ax[2].spines.top.set_visible(False)
#%% NaRo 3 regiones

fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Find the index at time 0
index_at_0 = (np.abs(time_sonido_NaRo_noche - 0)).argmin()
# Get the value of the sound at time 0
value_at_0 = average_NaRo_noche_sonido[index_at_0]

# Exclude the point at index 0 and find the closest index to value_at_0
excluded_indices = np.arange(len(average_NaRo_noche_sonido)) != index_at_0
closest_index_not_0 = (np.abs(average_NaRo_noche_sonido[excluded_indices] - value_at_0)).argmin()

# Convert back to original index space
actual_closest_index = np.arange(len(average_NaRo_noche_sonido))[excluded_indices][closest_index_not_0]

# Get the time at that index
time_at_value = time_sonido_NaRo_noche[actual_closest_index]

# Find the index in time_maximos_NaRo_noche closest to time_at_value (for pressure)
index_at_value_presion = (np.abs(time_maximos_NaRo_noche - time_at_value)).argmin()
value_at_time_presion = average_NaRo_noche_presion[index_at_value_presion]

index_at_value_presion_0 = (np.abs(time_maximos_NaRo_noche - 0)).argmin()
value_at_time_presion_0 = average_NaRo_noche_presion[index_at_value_presion_0]

# Find the index in time_rate_NaRo_noche closest to time_at_value (for rate)
index_at_value_rate = (np.abs(time_rate_NaRo_noche - time_at_value)).argmin()
value_at_time_rate = average_NaRo_noche_rate[index_at_value_rate]

index_at_value_rate_0 = (np.abs(time_rate_NaRo_noche - 0)).argmin()
value_at_time_rate_0 = average_NaRo_noche_rate[index_at_value_rate_0]

# Plot for ax[0] with time_sonido_NaRo
ax[0].axvspan(time_sonido_NaRo_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[0].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[0].axvspan(time_at_value, time_sonido_NaRo_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[0].errorbar(time_sonido_NaRo_noche, average_NaRo_noche_sonido, std_NaRo_noche_sonido, color='C0')
ax[0].text(20, 3, f"{round(time_at_value, 2)}s", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[0].legend(fancybox=True, shadow=True)

# Plot for ax[1] with time_maximos_NaRo
ax[1].axvspan(time_sonido_NaRo_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[1].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[1].axvspan(time_at_value, time_sonido_NaRo_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[1].errorbar(time_maximos_NaRo_noche, average_NaRo_noche_presion, std_NaRo_noche_presion, color='C0')
ax[1].text(20, 3, f"{round(100 * (value_at_time_presion - value_at_time_presion_0) / value_at_time_presion_0, 2)}%", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
ax[1].legend(fancybox=True, shadow=True)

# Plot for ax[2] with time_rate_NaRo
ax[2].axvspan(time_sonido_NaRo_noche[0], 0, facecolor='g', alpha=0.3, edgecolor='k', linestyle='--', label='Before')
ax[2].axvspan(0, time_at_value, facecolor='b', alpha=0.3, edgecolor='k', linestyle='--', label='During')
ax[2].axvspan(time_at_value, time_sonido_NaRo_noche[-1], facecolor='r', alpha=0.3, edgecolor='k', linestyle='--', label='After')
ax[2].errorbar(time_rate_NaRo_noche, average_NaRo_noche_rate, std_NaRo_noche_rate, color='C0')

ax[2].legend(fancybox=True, shadow=True)
ax[2].text(20, 2, f"{round(100 * (value_at_time_rate - value_at_time_rate_0) / value_at_time_rate_0, 2)}%", fontsize=12, bbox=dict(facecolor='k', alpha=0.1))
#%%

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

#RoVio
# Find the index in time_maximos_RoVio_noche closest to time_at_value (for pressure)
index_at_value_presion_RoVio = (np.abs(time_maximos_RoVio_noche - time_at_value)).argmin()
value_at_time_presion_RoVio = average_RoVio_noche_presion[index_at_value_presion_RoVio]

index_at_value_presion_0_RoVio = (np.abs(time_maximos_RoVio_noche - 0)).argmin()
value_at_time_presion_0_RoVio = average_NaRo_noche_presion[index_at_value_presion_0_RoVio]

# Find the index in time_rate_RoVio_noche closest to time_at_value (for rate)
index_at_value_rate_RoVio = (np.abs(time_rate_RoVio_noche - time_at_value)).argmin()
value_at_time_rate_RoVio = average_RoVio_noche_rate[index_at_value_rate_RoVio]

index_at_value_rate_0_RoVio = (np.abs(time_rate_RoVio_noche - 0)).argmin()
value_at_time_rate_0_RoVio = average_NaRo_noche_rate[index_at_value_rate_0_RoVio]

#NaRo
# Find the index in time_maximos_NaRo_noche closest to time_at_value (for pressure)
index_at_value_presion_NaRo = (np.abs(time_maximos_NaRo_noche - time_at_value)).argmin()
value_at_time_presion_NaRo = average_NaRo_noche_presion[index_at_value_presion_NaRo]

index_at_value_presion_0_NaRo = (np.abs(time_maximos_NaRo_noche - 0)).argmin()
value_at_time_presion_0_NaRo = average_NaRo_noche_presion[index_at_value_presion_0_NaRo]

# Find the index in time_rate_NaRo_noche closest to time_at_value (for rate)
index_at_value_rate_NaRo = (np.abs(time_rate_NaRo_noche - time_at_value)).argmin()
value_at_time_rate_NaRo = average_NaRo_noche_rate[index_at_value_rate_NaRo]

index_at_value_rate_0_NaRo = (np.abs(time_rate_NaRo_noche - 0)).argmin()
value_at_time_rate_0_NaRo = average_NaRo_noche_rate[index_at_value_rate_0_NaRo]
#%%
# Graficos
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
# fig.suptitle('Promedios')
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# RoNe Data
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido, std_RoNe_noche_sonido, color='royalblue', label='Average RoNe')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='royalblue', label='Average RoNe')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate/np.mean(average_RoNe_noche_rate[:100]), std_RoNe_noche_rate, color='royalblue', label='Average RoNe')

# RoVio Data
ax[0].errorbar(time_sonido_RoVio_noche, average_RoVio_noche_sonido, std_RoVio_noche_sonido, color=colors[1], label='Average RoVio')
ax[1].errorbar(time_maximos_RoVio_noche, average_RoVio_noche_presion, std_RoVio_noche_presion, color=colors[1], label='Average RoVio')
ax[2].errorbar(time_rate_RoVio_noche, average_RoVio_noche_rate/np.mean(average_RoVio_noche_rate[:100]), std_RoVio_noche_rate, color=colors[1], label='Average RoVio')

# NaRo Data
ax[0].errorbar(time_sonido_NaRo_noche, average_NaRo_noche_sonido, std_NaRo_noche_sonido, color=colors[2], label='Average NaRo')
ax[1].errorbar(time_maximos_NaRo_noche, average_NaRo_noche_presion, std_NaRo_noche_presion, color=colors[2], label='Average NaRo')
ax[2].errorbar(time_rate_NaRo_noche, average_NaRo_noche_rate/np.mean(average_NaRo_noche_rate[:100]), std_NaRo_noche_rate, color=colors[2], label='Average NaRo')


ax[0].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--', label='Airplaine noise time-window')
ax[0].text(20, 3, f"{int(round(time_at_value, 0))}s", fontsize=12, bbox=dict(facecolor='#B35F9F', alpha=0.3))
# ax[0].axvline(0,linestyle='dashed')
ax[1].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--', label='Airplaine noise time-window')
ax[1].text(20, 3, f"{int(round(100 * (value_at_time_presion - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor='royalblue', alpha=1))
ax[1].text(25, 3, f"{int(round(100 * (value_at_time_presion_RoVio - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor=colors[1], alpha=1))
ax[1].text(30, 3, f"{int(round(100 * (value_at_time_presion_NaRo - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor=colors[2], alpha=1))
ax[2].axvspan(0, time_at_value, facecolor='#B35F9F', alpha=0.3, edgecolor='k', linestyle='--', label='Airplaine noise time-window')
ax[2].text(20, 2, f"{int(round(100 * (value_at_time_rate - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor='royalblue', alpha=1))
ax[2].text(26, 2, f"{int(round(100 * (value_at_time_rate_RoVio - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor=colors[1], alpha=1))
ax[2].text(31, 2, f"{int(round(100 * (value_at_time_rate_NaRo - 1) / 1, 0))}%", fontsize=12, bbox=dict(facecolor=colors[2], alpha=1))


# Add legends for each axis
ax[0].legend(fancybox=True, shadow=True,loc='upper left')
ax[1].legend(fancybox=True, shadow=True,loc='upper left')
ax[2].legend(fancybox=True, shadow=True,loc='upper left')

plt.savefig('Promedio de los tres.pdf')


#%%

species = ("RoNe", "RoVio", "NaRo")
penguin_means = {
    'Pressure': (int(round(100 * (value_at_time_presion - 1) / 1, 0)), int(round(100 * (value_at_time_presion_RoVio - 1) / 1, 0)),int(round(100 * (value_at_time_presion_NaRo - 1) / 1, 0))),
    'Rate': (int(round(100 * (value_at_time_rate - 1) / 1, 0)), int(round(100 * (value_at_time_rate_RoVio - 1) / 1, 0)), int(round(100 * (value_at_time_rate_NaRo - 1) / 1, 0))),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(14, 9),layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Porcentage (%)')
ax.set_title('Quantities after the passing of the plane')
ax.set_xticks(x + width/2, species)
ax.legend(fancybox=True, shadow=True,loc='upper left', ncols=2)
ax.set_ylim(0, 150)