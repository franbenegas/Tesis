# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:04:17 2024

@author: beneg
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import find_peaks
from matplotlib.widgets import Button, SpanSelector
from scipy import signal
import pandas as pd
import pickle

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\2023-02-11-night\Aviones\Aviones y pajaros V2'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = []
Sonidos = []


for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
        Presiones.append(file)
    
def datos_normalizados_2(indice, ti, tf):
    
    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]
    
    fs,audio = wavfile.read(Sound)
    fs,pressure = wavfile.read(Pressure)
    
    pressure = pressure-np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)
    
    #funcion que normaliza al [-1, 1]
    def norm11_interval(x, ti, tf, fs):
      x_int = x[int(ti*fs):int(tf*fs)]
      return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1
        
    audio = audio-np.mean(audio)
    audio_norm = audio / np.max(audio)
    
    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs
    
    
# Find the longest continuous interval
def find_longest_interval(times, max_gap=1):
    longest_interval = []
    current_interval = [times[0]]

    for i in range(1, len(times)):
        if times[i] - times[i-1] <= max_gap:
            current_interval.append(times[i])
        else:
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]

    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval

    return longest_interval


class AnalisisApp:
    
    def __init__(self):
        
        self.columnas = ['Nombre','Tiempo inicial normalizacion','Tiempo final normalizacion','Tiempo inicial avion','Tiempo final avion']
        self.Datos = pd.DataFrame(columns=self.columnas)
        self.indice = 0
        self.ti = 0
        self.tf = 1

        # Create the figure and axes without using constrained_layout
        self.fig, self.axs = plt.subplots(2, 1, figsize=(14, 7))
        
        # Adjust the layout to leave space for buttons at the bottom
        self.fig.subplots_adjust(bottom=0.25)

        # Load initial data
        self.load_data()
        self.plot_data()

        # Create buttons and store references in a dictionary
        self.buttons = {}
        self.create_buttons()

        # Create SpanSelector for interactive selection
        self.span_selector = SpanSelector(self.axs[0], self.on_select, 'horizontal',
                                          props=dict(facecolor='gray', alpha=0.5), useblit=True)
        # Connect the click event to on_click method
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_data(self):
        self.audio, self.pressure, self.name, self.fs = datos_normalizados_2(self.indice, self.ti, self.tf)
        self.time = np.linspace(0, len(self.pressure) / self.fs, len(self.pressure))
        
        self.peaks_maximos, self.properties_maximos = find_peaks(self.pressure, prominence=1, height=0, distance=int(self.fs * 0.1))

        f, t, Sxx = signal.spectrogram(self.audio, self.fs)
        # Define the frequency cutoff
        frequency_cutoff = 1000  # Adjust as needed #1000 funciona piola
        threshold = -10  # dB threshold for detection   -10 se acerca mas

        # Convert the spectrogram to dB scale
        Sxx_dB =np.log(Sxx)

        # Find the indices where the frequency is above the cutoff
        freq_indices = np.where(f > frequency_cutoff)[0]

        # Identify times where the spectrogram surpasses the threshold
        time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
        time_above_threshold = t[time_indices]


        self.longest_interval = find_longest_interval(time_above_threshold)
        
    def plot_data(self):
        self.axs[0].cla()
        self.axs[1].cla()
        
        self.axs[0].plot(self.time, self.audio)
        self.axs[0].axvline(self.longest_interval[0],color='k')
        self.axs[0].axvline(self.longest_interval[-1],color='k')
        self.axs[1].plot(self.time, self.pressure)
        self.axs[1].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        self.axs[1].legend(fancybox=True, shadow=True, loc='upper right')
        self.axs[1].axvline(self.longest_interval[0],color='k')
        self.axs[1].axvline(self.longest_interval[-1],color='k')
        
        self.fig.suptitle(f'{self.name}:{self.indice + 1}/{len(Sonidos)}', fontsize=16)
        self.axs[0].set_ylabel("Audio (arb. u.)")
        self.axs[1].set_ylabel("Pressure (arb. u.)")
        self.axs[1].set_xlabel("Time (sec)")
        
        plt.draw()

    def create_buttons(self):
        # Adjust the buttons' position below the plot
        button_width = 0.1
        button_height = 0.075
        button_spacing = 0.02
        total_button_width = 4 * button_width + 3 * button_spacing
        start_x = (1 - total_button_width) / 2

        # Position the buttons below the plot by setting y-coordinate lower
        axprev = self.fig.add_axes([start_x, 0.02, button_width, button_height])
        axnext = self.fig.add_axes([start_x + (button_width + button_spacing), 0.02, button_width, button_height])
        axsave = self.fig.add_axes([start_x + 2 * (button_width + button_spacing), 0.02, button_width, button_height])
        axdatos = self.fig.add_axes([start_x + 3 * (button_width + button_spacing), 0.02, button_width, button_height])
        # Create buttons and store references in the dictionary
        self.buttons['prev'] = Button(axprev, 'Previous')
        self.buttons['next'] = Button(axnext, 'Next')
        self.buttons['save'] = Button(axsave, 'Save')
        self.buttons['add'] = Button(axdatos,'Add')
        
        # Assign callbacks
        self.buttons['prev'].on_clicked(self.prev)
        self.buttons['next'].on_clicked(self.next)
        self.buttons['save'].on_clicked(self.save_peaks)
        self.buttons['add'].on_clicked(self.save_dataframe)
        
        plt.draw()    
        
    def next(self, event):
        self.indice += 1
        self.update_plot()

    def prev(self, event):
        self.indice -= 1
        self.update_plot()

    def on_select(self, vmin, vmax):
        self.ti = vmin
        self.tf = vmax
        self.update_plot()

    def update_plot(self):
        self.load_data()
        self.plot_data()

    def on_click(self, event):
        if event.inaxes == self.axs[1]:
            x_click = event.xdata
            y_click = event.ydata
    
            if self.peaks_maximos.size > 0:
                distances = np.sqrt((self.time[self.peaks_maximos] - x_click)**2 + (self.pressure[self.peaks_maximos] - y_click)**2)
                closest_peak_index = np.argmin(distances)
                if distances[closest_peak_index] < 0.1:  # Adjust the threshold distance as needed
                    self.peaks_maximos = np.delete(self.peaks_maximos, closest_peak_index)
                    self.plot_data()
    
    def save_peaks(self, event=None):
        np.savetxt(f'{self.name}_maximos_prueba.txt', self.peaks_maximos, delimiter=',', newline='\n', fmt='%i')
        temp_df = pd.DataFrame([[self.name, self.ti, self.tf, self.longest_interval[0],self.longest_interval[-1]]], columns=self.columnas)

        self.Datos = pd.concat([self.Datos, temp_df], ignore_index=True)
        print('Saved maximos')
      
    # def save_dataframe(self,event=None):
    #     with open(f'Datos2xdxdxdxd {self.name[:-9]}.pkl', 'wb') as file:
    #         pickle.dump(self.Datos, file)
    #     print('Saved dataframe')
    def save_dataframe(self, event=None):
        # Save DataFrame as a JSON file
        json_file_name = 'Datos_lol.json'
        self.Datos.to_json(json_file_name, orient='records', lines=True)  # You can change 'orient' and 'lines' if needed
        print(f'Saved dataframe as {json_file_name}')
        
if __name__ == "__main__":
    app = AnalisisApp()
    plt.show()

#%% Agregamos rate aca
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-28-night\poster'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = []
Sonidos = []


for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
        Presiones.append(file)
    
def datos_normalizados_2(indice, ti, tf):
    
    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]
    
    fs,audio = wavfile.read(Sound)
    fs,pressure = wavfile.read(Pressure)
    
    pressure = pressure-np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)
    
    #funcion que normaliza al [-1, 1]
    def norm11_interval(x, ti, tf, fs):
      x_int = x[int(ti*fs):int(tf*fs)]
      return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1
        
    audio = audio-np.mean(audio)
    audio_norm = audio / np.max(audio)
    
    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs
    
    
# Find the longest continuous interval
def find_longest_interval(times, max_gap=1):
    longest_interval = []
    current_interval = [times[0]]

    for i in range(1, len(times)):
        if times[i] - times[i-1] <= max_gap:
            current_interval.append(times[i])
        else:
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]

    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval

    return longest_interval


class AnalisisApp:
    
    def __init__(self):
        
        self.columnas = ['Nombre','Tiempo inicial normalizacion','Tiempo final normalizacion','Tiempo inicial avion','Tiempo final avion']
        self.Datos = pd.DataFrame(columns=self.columnas)
        self.indice = 0
        self.ti = 0
        self.tf = 1

        # Create the figure and axes without using constrained_layout
        self.fig, self.axs = plt.subplots(3, 1, figsize=(17, 8))
        # Adjust the layout to leave space for buttons at the bottom
        self.fig.subplots_adjust(bottom=0.25)

        # Load initial data
        self.load_data()
        self.plot_data()

        # Create buttons and store references in a dictionary
        self.buttons = {}
        self.create_buttons()

        # Create SpanSelector for interactive selection
        self.span_selector = SpanSelector(self.axs[0], self.on_select, 'horizontal',
                                          props=dict(facecolor='gray', alpha=0.5), useblit=True)
        # Connect the click event to on_click method
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_data(self):
        self.audio, self.pressure, self.name, self.fs = datos_normalizados_2(self.indice, self.ti, self.tf)
        self.time = np.linspace(0, len(self.pressure) / self.fs, len(self.pressure))
        
        self.peaks_maximos, _ = find_peaks(self.pressure, prominence=1, height=0, distance=int(self.fs * 0.1))
        
        self.rate = 1/np.diff(self.time[self.peaks_maximos])

        
        self.f, self.t, self.Sxx = signal.spectrogram(self.audio, self.fs)
        # Define the frequency cutoff
        frequency_cutoff = 1000  # Adjust as needed #1000 funciona piola
        threshold = -10  # dB threshold for detection   -10 se acerca mas

        # Convert the spectrogram to dB scale
        Sxx_dB =np.log(self.Sxx)

        # Find the indices where the frequency is above the cutoff
        freq_indices = np.where(self.f > frequency_cutoff)[0]

        # Identify times where the spectrogram surpasses the threshold
        time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
        time_above_threshold = self.t[time_indices]


        self.longest_interval = find_longest_interval(time_above_threshold)
        
    def plot_data(self):
        self.axs[0].cla()
        self.axs[1].cla()
        self.axs[2].cla()
        
        self.axs[0].plot(self.time, self.audio)
        self.axs[0].axvline(self.longest_interval[0],color='k')
        self.axs[0].axvline(self.longest_interval[-1],color='k')
        self.axs[0].axhline(1,color='k', linestyle='dashed', alpha=0.5)
        self.axs[0].axhline(-1,color='k', linestyle='dashed', alpha=0.5)
        self.axs[1].plot(self.time, self.pressure)
        self.axs[1].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        self.axs[1].legend(fancybox=True, shadow=True, loc='upper right')
        self.axs[1].axvline(self.longest_interval[0],color='k')
        self.axs[1].axvline(self.longest_interval[-1],color='k')
        self.axs[1].axhline(1,color='k', linestyle='dashed', alpha=0.5)
        self.axs[1].axhline(-1,color='k', linestyle='dashed', alpha=0.5)
        self.axs[2].plot(self.time[self.peaks_maximos][1:],self.rate)
        
        
        self.fig.suptitle(f'{self.name}:{self.indice + 1}/{len(Sonidos)}', fontsize=16)
        self.axs[0].set_ylabel("Audio (arb. u.)")
        self.axs[1].set_ylabel("Pressure (arb. u.)")
        self.axs[2].set_ylabel("Rate (Hz)")
        self.axs[2].set_xlabel("Time (sec)")
        
        plt.draw()
        
    def create_buttons(self):
        # Adjust the buttons' position below the plot
        button_width = 0.1
        button_height = 0.075
        button_spacing = 0.02
        total_button_width = 4 * button_width + 3 * button_spacing
        start_x = (1 - total_button_width) / 2

        # Position the buttons below the plot by setting y-coordinate lower
        axprev = self.fig.add_axes([start_x, 0.02, button_width, button_height])
        axnext = self.fig.add_axes([start_x + (button_width + button_spacing), 0.02, button_width, button_height])
        axsave = self.fig.add_axes([start_x + 2 * (button_width + button_spacing), 0.02, button_width, button_height])
        axdatos = self.fig.add_axes([start_x + 3 * (button_width + button_spacing), 0.02, button_width, button_height])
        # Create buttons and store references in the dictionary
        self.buttons['prev'] = Button(axprev, 'Previous')
        self.buttons['next'] = Button(axnext, 'Next')
        self.buttons['save'] = Button(axsave, 'Save')
        self.buttons['Save df'] = Button(axdatos,'Save df')
        
        # Assign callbacks
        self.buttons['prev'].on_clicked(self.prev)
        self.buttons['next'].on_clicked(self.next)
        self.buttons['save'].on_clicked(self.save_peaks)
        self.buttons['Save df'].on_clicked(self.save_dataframe)
        
        plt.draw()    
        
    def next(self, event):
        self.indice += 1
        self.update_plot()

    def prev(self, event):
        self.indice -= 1
        self.update_plot()

    def on_select(self, vmin, vmax):
        self.ti = vmin
        self.tf = vmax
        self.update_plot()

    def update_plot(self):
        self.load_data()
        self.plot_data()

    def on_click(self, event):
        if event.inaxes == self.axs[1]:
            if self.peaks_maximos.size > 0:
                x_click = event.xdata
                y_click = event.ydata
    
                distances = np.sqrt((self.time[self.peaks_maximos] - x_click)**2 + (self.pressure[self.peaks_maximos] - y_click)**2)
                closest_peak_index = np.argmin(distances)
                
                if distances[closest_peak_index] < 0.1:  # Adjust threshold
                    self.peaks_maximos = np.delete(self.peaks_maximos, closest_peak_index)
                    self.rate = 1 / np.diff(self.time[self.peaks_maximos])
                    self.plot_data()
            else:
                print("No more peaks to delete.")
    
    def save_peaks(self, event=None):
        np.savetxt(f'{self.name}_maximos.txt', self.peaks_maximos, delimiter=',', newline='\n', fmt='%i')
        temp_df = pd.DataFrame([[self.name, self.ti, self.tf, self.longest_interval[0],self.longest_interval[-1]]], columns=self.columnas)

        self.Datos = pd.concat([self.Datos, temp_df], ignore_index=True)
        print(f'Saved maximos: {self.indice + 1}/{len(Sonidos)}')
      
    def save_dataframe(self,event=None):
        with open(f'Datos2xdxdxdxd {self.name[:-9]}.pkl', 'wb') as file:
            pickle.dump(self.Datos, file)
        print('Saved dataframe')
    # def save_dataframe(self, event=None):
    #     # Save DataFrame as a JSON file
    #     json_file_name = 'Datos_lol.json'
    #     self.Datos.to_json(json_file_name, orient='records', lines=True)  # You can change 'orient' and 'lines' if needed
    #     print(f'Saved dataframe as {json_file_name}')
        
if __name__ == "__main__":
    app = AnalisisApp()
    plt.show()
    
#%% aca quiero hacer lo mismo pero solo para datos de presion

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\Datos basal'

lugar = os.listdir(directory)
Presiones = [file for file in lugar if file.startswith('p')]

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\2023-02-11-night\Aviones'
lugar = os.listdir(directory)
Presiones2 = [file for file in lugar if file.startswith('p')]


# Get the last 12 characters from each string in Presiones2
last_12_chars_Presiones2 = {file[-12:] for file in Presiones2}

# Filter Presiones where the last 12 characters are not in Presiones2
filtered_presiones = [file for file in Presiones if file[-12:] not in last_12_chars_Presiones2]

Presiones = filtered_presiones[::2]

#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\Datos basal'

from shutil import copyfile

source_folder = directory# La direccion de la carpeta Aviones
destination_folder = os.path.join(source_folder, 'Basal')

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
for src in Presiones:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path)
#%%
# Define the working directory
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\Datos basal\Basal'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = [file for file in lugar if file.startswith('p')]

# Presiones = filtered_presiones[::2]

def datos_normalizados_2(indice, ti, tf):
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

class AnalisisApp:
    def __init__(self):
        self.columnas = ['Nombre', 'Tiempo inicial normalizacion', 'Tiempo final normalizacion', 'Tiempo inicial avion', 'Tiempo final avion']
        self.Datos = pd.DataFrame(columns=self.columnas)
        self.indice = 0
        self.ti = 0
        self.tf = 1

        # Create figure and axes
        self.fig, self.axs = plt.subplots(2, 1, figsize=(17, 8),sharex=True)
        self.fig.subplots_adjust(bottom=0.25)

        # Load data and plot
        self.load_data()
        self.plot_data()

        # Create buttons
        self.buttons = {}
        self.create_buttons()

        # SpanSelector
        self.span_selector = SpanSelector(self.axs[0], self.on_select, 'horizontal', props=dict(facecolor='gray', alpha=0.5),button=3, useblit=True)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_data(self):
        self.pressure, self.name, self.fs = datos_normalizados_2(self.indice, self.ti, self.tf)
        self.time = np.linspace(0, len(self.pressure) / self.fs, len(self.pressure))
        self.peaks_maximos, _ = find_peaks(self.pressure, prominence=1, height=0, distance=int(self.fs * 0.1))
        self.rate = 1 / np.diff(self.time[self.peaks_maximos])

    def plot_data(self):
        self.axs[0].cla()  # Clear previous plots on the first axis (pressure data)
        self.axs[1].cla()  # Clear previous plots on the second axis (rate data)
    
        # Plot the pressure data along with the identified peaks
        self.axs[0].plot(self.time, self.pressure)
        self.axs[0].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        self.axs[0].legend(fancybox=True, shadow=True, loc='upper right')
    
        # Add dashed horizontal lines for visualization purposes
        self.axs[0].axhline(1, color='k', linestyle='dashed', alpha=0.5)
        self.axs[0].axhline(-1, color='k', linestyle='dashed', alpha=0.5)
    
        # Plot rate data
        if len(self.peaks_maximos) > 1:
            self.axs[1].plot(self.time[self.peaks_maximos][1:], self.rate)  # Ensure the x-axis has the correct length
    
        # Update plot labels and title
        self.fig.suptitle(f'{self.name}: {self.indice + 1}/{len(Presiones)}', fontsize=16)
        self.axs[0].set_ylabel("Pressure (arb. u.)")
        self.axs[1].set_ylabel("Rate (Hz)")
        self.axs[1].set_xlabel("Time (sec)")
    
        # Refresh the plot
        plt.draw()


    def create_buttons(self):
        button_width = 0.1
        button_height = 0.075
        button_spacing = 0.02
        total_button_width = 4 * button_width + 3 * button_spacing
        start_x = (1 - total_button_width) / 2

        axprev = self.fig.add_axes([start_x, 0.02, button_width, button_height])
        axnext = self.fig.add_axes([start_x + (button_width + button_spacing), 0.02, button_width, button_height])
        axsave = self.fig.add_axes([start_x + 2 * (button_width + button_spacing), 0.02, button_width, button_height])
        axdatos = self.fig.add_axes([start_x + 3 * (button_width + button_spacing), 0.02, button_width, button_height])

        self.buttons['prev'] = Button(axprev, 'Previous')
        self.buttons['next'] = Button(axnext, 'Next')
        self.buttons['save'] = Button(axsave, 'Save')
        self.buttons['Save df'] = Button(axdatos, 'Save df')

        self.buttons['prev'].on_clicked(self.prev)
        self.buttons['next'].on_clicked(self.next)
        self.buttons['save'].on_clicked(self.save_peaks)
        self.buttons['Save df'].on_clicked(self.save_dataframe)

    def next(self, event):
        if self.indice < len(Presiones) - 1:
            self.indice += 1
            self.update_plot()

    def prev(self, event):
        if self.indice > 0:
            self.indice -= 1
            self.update_plot()

    def on_select(self, vmin, vmax):
        self.ti = vmin
        self.tf = vmax
        self.update_plot()

    def update_plot(self):
        self.load_data()
        self.plot_data()

    def on_click(self, event):
        if event.inaxes == self.axs[0]:
            if self.peaks_maximos.size > 0:
                x_click = event.xdata
                y_click = event.ydata
    
                distances = np.sqrt((self.time[self.peaks_maximos] - x_click)**2 + (self.pressure[self.peaks_maximos] - y_click)**2)
                closest_peak_index = np.argmin(distances)
                
                if distances[closest_peak_index] < 0.1:  # Adjust threshold
                    self.peaks_maximos = np.delete(self.peaks_maximos, closest_peak_index)
                    self.rate = 1 / np.diff(self.time[self.peaks_maximos])
                    self.plot_data()
            else:
                print("No more peaks to delete.")

    def save_peaks(self, event=None):
        # np.savetxt(f'{self.name}_maximos.txt', self.peaks_maximos, delimiter=',', newline='\n', fmt='%i')
        if len(self.peaks_maximos) > 0:
            first_peak = self.time[self.peaks_maximos[0]]
            last_peak = self.time[self.peaks_maximos[-1]]
        else:
            first_peak, last_peak = None, None
        temp_df = pd.DataFrame([[self.name, self.ti, self.tf, first_peak, last_peak]], columns=self.columnas)
        self.Datos = pd.concat([self.Datos, temp_df], ignore_index=True)
        print(f'Saved maximos: {self.indice + 1}/{len(Presiones)}')

    def save_dataframe(self, event=None):
        with open(f'Datos3xdxdxdxd_{self.name}.pkl', 'wb') as file:
            pickle.dump(self.Datos, file)
        print('Saved dataframe')


if __name__ == "__main__":
    app = AnalisisApp()
    plt.show()
