# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 00:56:45 2025

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
            
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            periodo_sin_filtrar = np.diff(time[lugar_maximos])
            
       
            tiempo = time - tiempo_inicial
            
            
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

def process_night_data(directories):
    def datos_normalizados(Presiones, indice, ti, tf):
        Pressure = Presiones[indice]
        name = Pressure[9:-4]  # Extract the name from the Pressure file
    
        fs, pressure = wavfile.read(Pressure)
     
        pressure = pressure - np.mean(pressure)  # Remove mean
        pressure_norm = pressure / np.max(pressure)  # Normalize by max value
        
        # Function to normalize the data to [-1, 1] in a given time interval
        def norm11_interval(x, ti, tf, fs):
            x_int = x[int(ti * fs):int(tf * fs)]  # Extract the data in the interval
            return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
            
        # Normalize audio and pressure within the interval
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    
        return pressure_norm, name, fs
    
    combined_time_series_list = []
    # Main loop: Process each directory
    for directory in directories:
        os.chdir(directory)  # Change working directory
        files = os.listdir(directory)  # List all files in the directory
        
        presiones = []  # To store pressure files
        datos = []  # To store data files
        
        # Classify files into sonidos, presiones, and datos based on their names
        for file in files:            
            if file[0] == 'p':
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
            pressure, name, fs = datos_normalizados(presiones, indice, ti, tf)
            
           
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            
            combined_time_series_list.append({
                'Pressure' : pressure,
                'Pressure maximum': pressure[lugar_maximos]
            })
    
    return combined_time_series_list
    

#%%

import numpy as np
import os
import json
from scipy.io import wavfile

def process_night_data_for_training(directories, window_size=10, num_negatives_per_signal=50):
    X = []
    y = []

    def extract_features(signal, idx, w):
        start = max(0, idx - w)
        end = min(len(signal), idx + w + 1)
        window = signal[start:end]
        return [
            signal[idx],
            signal[idx] - np.min(window),       # Prominence
            np.std(window),
            np.mean(np.gradient(window)),
        ]

    for directory in directories:
        os.chdir(directory)
        files = os.listdir(directory)
        
        presiones = [f for f in files if f.startswith('p')]
        datos_files = [f for f in files if f.endswith('json')]

        with open(datos_files[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)

        for i in range(len(datos)):
            ti = datos[i]['Tiempo inicial normalizacion']
            tf = datos[i]['Tiempo final normalizacion']

            # Read and normalize pressure signal
            pressure_file = presiones[i]
            fs, pressure = wavfile.read(pressure_file)
            pressure = pressure - np.mean(pressure)
            pressure /= np.max(np.abs(pressure))

            # Normalize in [ti, tf]
            idx_start = int(ti * fs)
            idx_end = int(tf * fs)
            pressure_int = pressure[idx_start:idx_end]
            pressure_norm = 2 * (pressure - np.min(pressure_int)) / (np.max(pressure_int) - np.min(pressure_int)) - 1

            # Read labeled peak positions
            name = pressure_file[9:-4]
            peak_indices = np.loadtxt(f'{name}_maximos.txt', dtype=int)

            # Extract positive samples
            for peak in peak_indices:
                if peak >= window_size and peak < len(pressure_norm) - window_size:
                    features = extract_features(pressure_norm, peak, window_size)
                    X.append(features)
                    y.append(1)

            # Extract negative samples (random non-peak points)
            all_indices = set(range(window_size, len(pressure_norm) - window_size))
            non_peaks = list(all_indices - set(peak_indices))
            random_negatives = np.random.choice(non_peaks, size=min(num_negatives_per_signal, len(non_peaks)), replace=False)
            for idx in random_negatives:
                features = extract_features(pressure_norm, idx, window_size)
                X.append(features)
                y.append(0)

    return np.array(X), np.array(y)


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
directories = [noches_1, noches_2,noches_3]#noches_3 la voy a usar para testear

X, y = process_night_data_for_training(directories)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Accuracy:", clf.score(X_test, y_test))

#%%
import matplotlib.pyplot as plt

def extract_features(signal, idx, w):
    start = max(0, idx - w)
    end = min(len(signal), idx + w + 1)
    window = signal[start:end]
    return [
        signal[idx],
        signal[idx] - np.min(window),
        np.std(window),
        np.mean(np.gradient(window)),
    ]

def predict_peaks_in_directory(model, directory, window_size=10):
    files = os.listdir(directory)
    presiones = [f for f in files if f.startswith('p')]
    datos_files = [f for f in files if f.endswith('json')]

    with open(datos_files[0], 'r', encoding='utf-8') as f:
        datos = json.load(f)

    indice = 0  # test with first file
    ti = datos[indice]['Tiempo inicial normalizacion']
    tf = datos[indice]['Tiempo final normalizacion']

    pressure_file = presiones[indice]
    fs, pressure = wavfile.read(os.path.join(directory, pressure_file))
    pressure = pressure - np.mean(pressure)
    pressure /= np.max(np.abs(pressure))

    idx_start = int(ti * fs)
    idx_end = int(tf * fs)
    pressure_int = pressure[idx_start:idx_end]
    pressure_norm = 2 * (pressure - np.min(pressure_int)) / (np.max(pressure_int) - np.min(pressure_int)) - 1

    # Read manually labeled peaks
    name = pressure_file[9:-4]
    labeled_peaks = np.loadtxt(os.path.join(directory, f'{name}_maximos.txt'), dtype=int)

    # Predict peaks
    candidate_indices = np.arange(window_size, len(pressure_norm) - window_size)
    features = [extract_features(pressure_norm, idx, window_size) for idx in candidate_indices]
    predictions = model.predict(features)

    predicted_peaks = candidate_indices[np.array(predictions) == 1]

    return pressure_norm, labeled_peaks, predicted_peaks, fs

pressure, labeled_peaks, predicted_peaks, fs = predict_peaks_in_directory(clf, noches_3)
#%%
plt.figure(figsize=(15, 5))
time = np.arange(len(pressure)) / fs

plt.plot(time, pressure, label='Normalized Pressure', alpha=0.7)
plt.plot(labeled_peaks / fs, pressure[labeled_peaks], '.r', label='Labeled Peaks',zorder=50)
plt.plot(predicted_peaks / fs, pressure[predicted_peaks], '.b', label='Predicted Peaks', alpha=0.6)

plt.title("Comparison of Labeled vs Predicted Pressure Peaks")
plt.xlabel("Time (s)")
plt.ylabel("Normalized Pressure")
plt.legend()
plt.grid(True)
plt.tight_layout()


#%%

def process_night_data_for_training(directories, window_size=10, num_negatives_per_signal=50):
    X = []
    y = []

    def extract_features(signal, idx, w):
        start = max(0, idx - w)
        end = min(len(signal), idx + w + 1)
        window = signal[start:end]
        return [
            signal[idx],
            signal[idx] - np.min(window),       # Prominence
            np.std(window),
            np.mean(np.gradient(window)),
        ]

    for directory in directories:
        os.chdir(directory)
        files = os.listdir(directory)
        
        presiones = [f for f in files if f.startswith('p')]
        datos_files = [f for f in files if f.endswith('json')]

        with open(datos_files[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)

        for i in range(len(datos)):
            ti = datos[i]['Tiempo inicial normalizacion']
            tf = datos[i]['Tiempo final normalizacion']

            # Read and normalize pressure signal
            pressure_file = presiones[i]
            fs, pressure = wavfile.read(pressure_file)
            pressure = pressure - np.mean(pressure)
            pressure /= np.max(np.abs(pressure))

            # Normalize in [ti, tf]
            idx_start = int(ti * fs)
            idx_end = int(tf * fs)
            pressure_int = pressure[idx_start:idx_end]
            pressure_norm = 2 * (pressure - np.min(pressure_int)) / (np.max(pressure_int) - np.min(pressure_int)) - 1


            peaks_maximos, _ = find_peaks(pressure_norm, prominence=1, height=0, distance=int(fs * 0.1))
            # Read labeled peak positions
            name = pressure_file[9:-4]
            
            lugar_maximos = []  # Read manually identified maxima from a text file
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
            peak_indices = np.loadtxt(f'{name}_maximos.txt', dtype=int)

            # Extract positive samples
            for peak in peak_indices:
                if peak >= window_size and peak < len(pressure_norm) - window_size:
                    features = extract_features(pressure_norm, peak, window_size)
                    X.append(features)
                    y.append(1)

            # Extract negative samples (random non-peak points)
            all_indices = set(range(window_size, len(pressure_norm) - window_size))
            non_peaks = list(all_indices - set(peak_indices))
            random_negatives = np.random.choice(non_peaks, size=min(num_negatives_per_signal, len(non_peaks)), replace=False)
            for idx in random_negatives:
                features = extract_features(pressure_norm, idx, window_size)
                X.append(features)
                y.append(0)

    return np.array(X), np.array(y)

#%%
import numpy as np
from scipy.signal import find_peaks, peak_widths, peak_prominences

def extract_peak_features(signal, peaks, fs):
    prominences = peak_prominences(signal, peaks)[0]
    heights = signal[peaks]
    widths = peak_widths(signal, peaks, rel_height=0.5)[0]
    
    features = []
    for i, peak in enumerate(peaks):
        # Get distance to next peak (if not last peak)
        if i < len(peaks) - 1:
            distance_to_next = (peaks[i+1] - peak) / fs
        else:
            distance_to_next = 0  # or np.nan
        
        # Estimate slopes before and after the peak
        if 1 < peak < len(signal) - 2:
            slope_before = signal[peak] - signal[peak - 1]
            slope_after = signal[peak + 1] - signal[peak]
        else:
            slope_before = 0
            slope_after = 0

        features.append([
            prominences[i],
            heights[i],
            widths[i],
            distance_to_next,
            slope_before,
            slope_after
        ])
    
    return np.array(features)




X_total = []
y_total = []

for directory in directories:
    os.chdir(directory)
    files = os.listdir(directory)
    
    presiones = [f for f in files if f.startswith('p')]
    datos_files = [f for f in files if f.endswith('json')]

    with open(datos_files[0], 'r', encoding='utf-8') as f:
        datos = json.load(f)

    for i in range(len(datos)):
        ti = datos[i]['Tiempo inicial normalizacion']
        tf = datos[i]['Tiempo final normalizacion']

        pressure_file = presiones[i]
        fs, pressure = wavfile.read(pressure_file)
        pressure = pressure - np.mean(pressure)
        pressure /= np.max(np.abs(pressure))

        idx_start = int(ti * fs)
        idx_end = int(tf * fs)
        pressure_int = pressure[idx_start:idx_end]
        pressure_norm = 2 * (pressure - np.min(pressure_int)) / (np.max(pressure_int) - np.min(pressure_int)) - 1

        peaks_maximos, _ = find_peaks(pressure_norm, prominence=1, height=0, distance=int(fs * 0.1))
        features = extract_peak_features(pressure_norm, peaks_maximos, fs)

        # Load manually labeled peaks
        name = pressure_file[9:-4]
        
        labeled_peaks = []  # Read manually identified maxima from a text file
        maximos = np.loadtxt(f'{name}_maximos.txt')
        for i in maximos:
            labeled_peaks.append(int(i))

        labels = [1 if p in labeled_peaks else 0 for p in peaks_maximos]

        X_total.extend(features)
        y_total.extend(labels)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_total, y_total)
        
        