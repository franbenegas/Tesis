# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 16:56:40 2024

@author: beneg
"""

import parselmouth
import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shutil import copyfile
import sounddevice as sd

get_ipython().run_line_magic('matplotlib', 'qt5')


#%%

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia'
os.chdir(directory)
carpetas = os.listdir(directory)

pajaro = carpetas[1]

if carpetas:
    carpetas_1 = os.listdir(os.path.join(directory, pajaro))
    
    dia = carpetas_1[2]

    def list_files_starting_with(folder_path, prefix):
        try:
            # List all files and directories in the specified folder
            all_files = os.listdir(folder_path)
            
            # Filter files that start with the specified prefix
            filtered_files = [file for file in all_files if file.startswith(prefix)]
            
            return filtered_files
        except FileNotFoundError:
            return f"The folder at {folder_path} does not exist."

    # Specify the path to your folder
    folder_path = os.path.join(directory, pajaro, dia)

    # Get all files starting with 's'
    files_starting_with_s = list_files_starting_with(folder_path, 's')
    files_starting_with_p = list_files_starting_with(folder_path, 'p')

    

    print(f"Files starting with 's' in {dia}: {len(files_starting_with_s)}")
    print(f"Files starting with 'p' in {dia}: {len(files_starting_with_p)}")
    
else:
    print(f"No folders found in {directory}")

#%%


sonidos = []
presiones = []
# Global variables
current_index = 0
sound = None

def update_plot():
    global sound
    
    # Stop previous sound
    stop_sound()
    
    # Update plot
    file_path = os.path.join(folder_path, files_starting_with_s[current_index])
    sound = parselmouth.Sound(file_path)
    
    file_path = os.path.join(folder_path, files_starting_with_p[current_index])
    pressure = parselmouth.Sound(file_path)
    
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()
    
    ax[0].plot(sound.xs(), sound.values.T)
    # ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Sound")
    ax[0].set_title(f"Waveform of {files_starting_with_s[current_index]},file: {current_index + 1}/{len(files_starting_with_s)}")
    
    ax[1].plot(pressure.xs(), pressure.values.T)
    # ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Pressure")
    
    spectrogram = sound.to_spectrogram()
    draw_spectrogram(ax[2], spectrogram)
    plt.tight_layout()

    
    canvas.draw()

# Function to draw spectrogram
def draw_spectrogram(ax, spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    mesh = ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    ax.set_ylim([spectrogram.ymin, spectrogram.ymax])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return mesh

# Function to handle next button
def next_file():
    global current_index
    current_index = (current_index + 1) % len(files_starting_with_s)
    update_plot()

# Function to handle previous button
def prev_file():
    global current_index
    current_index = (current_index - 1) % len(files_starting_with_s)
    update_plot()

# Function to stop sound playback
def stop_sound():
    sd.stop()
        
def play_sound():
    global sound
    if sound:
        sd.play(sound.values.T.flatten(), sound.sampling_frequency)
        
def add_sound():
    sonidos.append(files_starting_with_s[current_index])
    presiones.append(files_starting_with_p[current_index])

# Setup Tkinter window
root = tk.Tk()
root.title("Audio File Viewer")

# Create a figure and axes for plotting
fig, ax = plt.subplots(3, 1, figsize=(10,7) , sharex=True)
plt.ioff()  # Prevent matplotlib from opening a separate window

# Add the plot to the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Add control buttons
controls = ttk.Frame(root)
controls.pack()

prev_button = ttk.Button(controls, text="Previous", command=prev_file)
prev_button.pack(side=tk.LEFT)

next_button = ttk.Button(controls, text="Next", command=next_file)
next_button.pack(side=tk.LEFT)

play_button = ttk.Button(controls, text="Play", command=play_sound)
play_button.pack(side=tk.LEFT)

stop_button = ttk.Button(controls, text="Stop", command=stop_sound)
stop_button.pack(side=tk.LEFT)

add_button = ttk.Button(controls, text="Add", command=add_sound)
add_button.pack(side=tk.LEFT)

# Initialize the plot with the first file
update_plot()

# Start the Tkinter main loop
root.mainloop()
#%%

source_folder = os.path.join(directory, carpetas[0], dia) # La direccion de la carpeta Aviones
destination_folder = os.path.join(source_folder, 'Aviones y pajaros')

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
for src in presiones:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path)
    
for src in sonidos:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path) 
  