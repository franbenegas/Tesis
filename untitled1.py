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


directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\2023-01-15-night\Aviones'
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


class SpectrogramApp:
    
    def __init__(self):
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

    def plot_data(self):
        self.axs[0].cla()
        self.axs[1].cla()
        
        self.axs[0].plot(self.time, self.audio)
        self.axs[1].plot(self.time, self.pressure)
        self.axs[1].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        self.axs[1].legend(fancybox=True, shadow=True, loc='upper right')
        
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
        total_button_width = 3 * button_width + 2 * button_spacing
        start_x = (1 - total_button_width) / 2

        # Position the buttons below the plot by setting y-coordinate lower
        axprev = self.fig.add_axes([start_x, 0.02, button_width, button_height])
        axnext = self.fig.add_axes([start_x + (button_width + button_spacing), 0.02, button_width, button_height])
        axsave = self.fig.add_axes([start_x + 2 * (button_width + button_spacing), 0.02, button_width, button_height])

        # Create buttons and store references in the dictionary
        self.buttons['prev'] = Button(axprev, 'Previous')
        self.buttons['next'] = Button(axnext, 'Next')
        self.buttons['save'] = Button(axsave, 'Save')

        # Assign callbacks
        self.buttons['prev'].on_clicked(self.prev)
        self.buttons['next'].on_clicked(self.next)
        self.buttons['save'].on_clicked(self.save_peaks)
        
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

if __name__ == "__main__":
    app = SpectrogramApp()
    plt.show()