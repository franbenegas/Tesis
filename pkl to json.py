# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:18:26 2024

@author: beneg
"""

import pickle
import json
import sys
import os
import numpy as np
import pandas as pd
# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[1]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)  # Create the path to the 'RoNe' folder

# List all subdirectories (representing different days)
dias = os.listdir(subdirectory)

# Path to the folder containing the night data for 'Aviones y pajaros' for the first three days
pajaritos = '\Aviones'
# noches_1 = subdirectory + '/' + dias[0] + pajaritos  # First day night folder
# noches_2 = subdirectory + '/' + dias[1] + pajaritos  # Second day night folder
# noches_3 = subdirectory + '/' + dias[2] + pajaritos  # Third day night folder
# noches_4 = subdirectory + '/' + dias[3] + pajaritos  # Third day night folder
basal = subdirectory + '/' + dias[3] + '\\basal'
# Store all directories in a list

directories = [basal]
# directories = [basal]
def convert_to_serializable(obj):
    """
    Convert non-serializable objects like NumPy arrays, pandas DataFrames, etc.,
    to JSON-compatible types.
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to list
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')  # Convert DataFrame to list of dicts
    else:
        return obj  # Return as-is if it’s already serializable

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
        elif file[-4:] == '.pkl':
            datos.append(file)
    
    # Load pickle object
    with open(datos[0], 'rb') as f:
        obj = pickle.load(f)
    
    # Convert complex pickle object to JSON serializable format
    serializable_obj = convert_to_serializable(obj)
    
    # Write the JSON file
    with open(os.path.splitext(datos[0])[0] + '.json', 'w', encoding='utf-8') as outfile:
        json.dump(serializable_obj, outfile, ensure_ascii=False, indent=4)