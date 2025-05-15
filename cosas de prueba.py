# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:05:11 2024

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
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'

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
os.chdir(directories[1])
files = os.listdir(directories[1])

presiones = []
sonidos = []
datos = []

for file in files:
    if file[0] == 's':
        sonidos.append(file)
    elif file[0] == 'p':
        presiones.append(file)
    elif file[-4:] == '.pkl':
        datos.append(file)
    

with open(datos[1], 'rb') as f:
    datos = pickle.load(f)