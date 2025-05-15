# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 11:22:12 2025

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


#%%
from scipy.fft import fft, fftfreq
def process_night_data(directories):
      
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
            # tiempo_inicial = datos[indice]['Tiempo inicial avion']
            
            # Get normalized audio and pressure data for the current index
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            # time = np.linspace(0, len(pressure) / fs, len(pressure))  # Time axis
            espectro = fft(audio)
            frecuencias = fftfreq(len(audio),fs)
            
            # Apply bandpass filter to audio signal
            # filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            
            # Find peaks in the filtered signal
            # peaks_sonido, _ = find_peaks(filtered_signal, height=0, distance=int(fs * 0.1), prominence=.001)

            
            
            combined_time_series_list.append({
                'frecuencias': frecuencias,
                'espectro': espectro
            })
            
    return combined_time_series_list

#%%

from scipy.fft import fft, fftfreq
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile
import numpy as np
import os
import json

def process_night_data(directories):
    def datos_normalizados(Sonidos, Presiones, indice, ti, tf):
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]

        fs, audio = wavfile.read(Sound)
        _, pressure = wavfile.read(Pressure)

        audio = audio - np.mean(audio)
        audio_norm = audio / np.max(np.abs(audio))
        
        pressure = pressure - np.mean(pressure)
        pressure_norm = pressure / np.max(np.abs(pressure))

        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1

        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)

        return audio_norm, pressure_norm, name, fs
    
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
    
    
    
    fft_magnitudes = []
    fft_magnitudes_filtrado = []
    sample_frequencies = None

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
            elif file.endswith('.json'):
                datos.append(file)

        with open(datos[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)

        for indice in range(len(datos)):
            ti, tf = datos[indice]['Tiempo inicial normalizacion'], datos[indice]['Tiempo final normalizacion']
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            
            
            # Truncate or pad audio to same length (optional, to standardize FFT length)
            n = len(audio)
            espectro = fft(audio)
            magnitud = np.abs(espectro)  # Use magnitude of FFT
            frecuencias = fftfreq(n, 1/fs)

            if sample_frequencies is None:
                sample_frequencies = frecuencias
            else:
                # Check if they match
                if len(frecuencias) != len(sample_frequencies):
                    min_len = min(len(frecuencias), len(sample_frequencies))
                    magnitud = magnitud[:min_len]
                    sample_frequencies = sample_frequencies[:min_len]

            fft_magnitudes.append(magnitud)
            
            
            
            # Apply bandpass filter to audio signal
            filtered_signal = butter_bandpass_filter(audio, lowcut, highcut, fs, order=order)
            espectro_filtrado = fft(filtered_signal)
            magnitud_filtrado = np.abs(espectro_filtrado)  # Use magnitude of FFT
            fft_magnitudes_filtrado.append(magnitud_filtrado)
            
    # Average FFT magnitude
    fft_magnitudes = np.array(fft_magnitudes)
    avg_magnitude = np.mean(fft_magnitudes, axis=0)
    
    fft_magnitudes_filtrado = np.array(fft_magnitudes_filtrado)
    avg_magnitude_filtrado = np.mean(fft_magnitudes_filtrado, axis=0)
    n_half = len(sample_frequencies) // 2
    return {
        'frecuencias': sample_frequencies[:n_half],
        'avg_magnitude': avg_magnitude[:n_half],
        'avg_magnitude_filtrado': avg_magnitude_filtrado[:n_half]
    }
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
directories = [noches_1, noches_2,noches_3]

espectros = process_night_data(directories)


#%%
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(espectros['frecuencias'],espectros['avg_magnitude'])
# plt.semilogy(espectros['frecuencias'],espectros['avg_magnitude_filtrado'])
plt.axvspan(1200, 6000, facecolor='r', alpha=0.5)
plt.axvspan(15000, 20000, facecolor='g', alpha=0.5)
