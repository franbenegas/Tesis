# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:23:55 2024

@author: beneg
"""
#%% New pratt

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, SpanSelector
import parselmouth
import os
import numpy as np
import sounddevice as sd

# Define the directory and load your data
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\Dia'
os.chdir(directory)
carpetas = os.listdir(directory)

pajaro = carpetas[1]

if carpetas:
    carpetas_1 = os.listdir(os.path.join(directory, pajaro))
    
    dia = carpetas_1[2]

    def list_files_starting_with(folder_path, prefix):
        try:
            all_files = os.listdir(folder_path)
            filtered_files = [file for file in all_files if file.startswith(prefix)]
            return filtered_files
        except FileNotFoundError:
            return f"The folder at {folder_path} does not exist."

    folder_path = os.path.join(directory, pajaro, dia)
    files_starting_with_s = list_files_starting_with(folder_path, 's')
    files_starting_with_p = list_files_starting_with(folder_path, 'pre')

sonidos = []
presiones = []

# Load the sound file
file_path = os.path.join(folder_path, files_starting_with_s[0])
sound = parselmouth.Sound(file_path)

file_path = os.path.join(folder_path, files_starting_with_p[0])
pressure = parselmouth.Sound(file_path)

def draw_spectrogram(ax, spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    ax.clear()  # Clear the previous spectrogram
    mesh = ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap='afmhot')
    ax.set_ylim([spectrogram.ymin, spectrogram.ymax])
    ax.set_xlabel("Time [s]",fontsize=12)
    ax.set_ylabel("Frequency [Hz]",fontsize=12)
    return mesh

# Get the sound data
x, y = sound.xs(), sound.values.T
xp, yp = pressure.xs(), pressure.values.T

# Plot the original sound data
fig, ax = plt.subplots(3, 1, figsize=(14, 7),sharex=True)
ax[0].set_title(f'{files_starting_with_s[0]}, file: {0 + 1}/{len(files_starting_with_s)}',fontsize=24)
ax[0].set_ylabel("Sound",fontsize=12)
ax[1].set_ylabel("Pressure",fontsize=12)
plt.tight_layout()

l, = ax[0].plot(x, y)
p, = ax[1].plot(xp, yp)

spectrogram = sound.to_spectrogram()
j = draw_spectrogram(ax[2], spectrogram)

class Index:
    ind = 0
    selection_start = None
    selection_end = None

    def update_plots(self):
        # Stop any currently playing sound
        sd.stop()
        
        i = self.ind % len(files_starting_with_s)
        
        ax[0].set_title(f'{files_starting_with_s[i]}, file: {i + 1}/{len(files_starting_with_s)}')
        
        # Update sound plot
        file_path = os.path.join(folder_path, files_starting_with_s[i])
        self.sound = parselmouth.Sound(file_path)
        ydata = self.sound.values.T
        l.set_ydata(ydata)
        ax[0].relim()
        ax[0].autoscale_view()

        # Update pressure plot
        file_path = os.path.join(folder_path, files_starting_with_p[i])
        self.pressure = parselmouth.Sound(file_path)
        ypdata = self.pressure.values.T
        p.set_ydata(ypdata)
        ax[1].relim()
        ax[1].autoscale_view()

        # Update spectrogram
        spectrogram = self.sound.to_spectrogram()
        draw_spectrogram(ax[2], spectrogram)

        # Reset selection
        self.selection_start = None
        self.selection_end = None
        span.extents = (0, 0)  # Clear the extents
        plt.draw()

    def next(self, event):
        self.ind += 1
        self.update_plots()

    def prev(self, event):
        self.ind -= 1
        self.update_plots()
        
    def play_sound(self, event):
        # Play the selected sound fragment
        if self.selection_start is not None and self.selection_end is not None:
            # Convert selection from time to sample indices
            start_index = int(self.selection_start * self.sound.sampling_frequency)
            end_index = int(self.selection_end * self.sound.sampling_frequency)
            
            # Extract the fragment and play it
            fragment = self.sound.values.T.flatten()[start_index:end_index]
            sd.play(fragment, self.sound.sampling_frequency)
        else:
            # Play the entire sound if no selection
            sd.play(self.sound.values.T.flatten(), self.sound.sampling_frequency)
    
    def stop_sound(self, event):
        # Stop any playing sound
        sd.stop()
    
    def add(self, event):
        sonidos.append(files_starting_with_s[self.ind])
        presiones.append(files_starting_with_p[self.ind])
    
    def onselect(self, vmin, vmax):
        # Update the selection range
        self.selection_start = vmin
        self.selection_end = vmax
        print(f'Selected range: {round(self.selection_start,2)} - {round(self.selection_end,2)}')

callback = Index()
callback.sound = sound  # Initialize the sound in the callback class

# Adjust the figure layout to create space for buttons below the plots
fig.subplots_adjust(bottom=0.25)

# Define the position of buttons (x, y, width, height)
button_width = 0.1
button_height = 0.075
button_spacing = 0.02  # Spacing between buttons

# Calculate the starting x position for centering the buttons
total_button_width = 5 * button_width + 4 * button_spacing  # 5 buttons, 4 spaces between them
start_x = (1 - total_button_width) / 2

axprev = fig.add_axes([start_x, 0.05, button_width, button_height])  # Previous button
axplay = fig.add_axes([start_x + (button_width + button_spacing) * 2, 0.05, button_width, button_height])  # Play button
axstop = fig.add_axes([start_x + (button_width + button_spacing) * 3, 0.05, button_width, button_height])  # Stop button
axnext = fig.add_axes([start_x + (button_width + button_spacing) * 4, 0.05, button_width, button_height])  # Next button
axadd  = fig.add_axes([start_x + (button_width + button_spacing) * 1, 0.05, button_width, button_height])  # Add button

# Create buttons and assign callbacks
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)
bplay = Button(axplay, 'Play')
bplay.on_clicked(callback.play_sound)
bstop = Button(axstop, 'Stop')
bstop.on_clicked(callback.stop_sound)
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
badd = Button(axadd, 'Add')
badd.on_clicked(callback.add)

span = SpanSelector(ax[0], callback.onselect, 'horizontal',
                    props=dict(facecolor='gray', alpha=0.5), useblit=True,interactive=True,button=1)

plt.show()


#%%
from shutil import copyfile

source_folder = os.path.join(directory, pajaro, dia) # La direccion de la carpeta Aviones
destination_folder = os.path.join(source_folder, 'Aviones')

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
#%%
