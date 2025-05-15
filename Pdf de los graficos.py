# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 12:21:34 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import pickle
from scipy.signal import find_peaks
from matplotlib.backends.backend_pdf import PdfPages
import json
# Set the font to 'STIX'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'
    
def plot_night_data(directories,name):
    pdf = PdfPages(name)
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
            

        with open(datos[0], 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        for indice in range(len(datos)):
            tiempo_inicial = datos[indice]['Tiempo inicial avion']
            audio, pressure, name, fs = datos_normalizados_nuevo(sonidos, presiones, indice)
            
            time = np.linspace(0, len(pressure)/fs, len(pressure))
            
            peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), prominence=.001)
            
            lugar_maximos = []
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))
                
            periodo = np.diff(time[lugar_maximos])
            
            fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
            fig.suptitle(f'{name}', fontsize=16)
            ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
            ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
            ax[2].set_ylabel("Rate (Hz)", fontsize=14)
            ax[2].set_xlabel("Time (s)", fontsize=14)
            plt.tight_layout()
            
            ax[0].plot(time[peaks_sonido],audio[peaks_sonido],color='C0')
            ax[0].axvline(tiempo_inicial,color='k')
            ax[1].plot(time,pressure,color='C0')
            ax[1].axvline(tiempo_inicial,color='k')
            ax[1].plot(time[lugar_maximos],pressure[lugar_maximos],'.C1',ms=10)
            ax[2].plot(time[lugar_maximos][1:],1/periodo)
            ax[2].axvline(tiempo_inicial,color='k')
            
            pdf.savefig(fig)
    
            # Close the figure to free memory
            plt.close(fig)
            
    pdf.close()
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
RoNe_noche = plot_night_data(directories,'RoNe plots2.pdf')