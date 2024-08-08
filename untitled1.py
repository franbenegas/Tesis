# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 18:04:17 2024

@author: beneg
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from scipy.io import wavfile
from tkinter import simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
from scipy.signal import find_peaks
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
from tkinter import simpledialog, filedialog


# directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF073-RoVio\2023-01-15-night\Aviones'
# os.chdir(directory)

# lugar = os.listdir(directory)
# Presiones = []
# Sonidos = []

# for file in lugar:
#     if file[0]=='s':
#         Sonidos.append(file)
#     elif file[0]=='p':
#         Presiones.append(file)
#%%
def datos_normalizados_2(indice, ti, tf, Sonidos, Presiones):
    Sound, Pressure = Sonidos[indice], Presiones[indice]
    name = Pressure[9:-4]
    
    fs, audio = wavfile.read(Sound)
    fs, pressure = wavfile.read(Pressure)
    
    pressure = pressure - np.mean(pressure)
    pressure_norm = pressure / np.max(pressure)
    
    def norm11_interval(x, ti, tf, fs):
        x_int = x[int(ti * fs):int(tf * fs)]
        return 2 * (x - np.min(x_int)) / (np.max(x_int) - np.min(x_int)) - 1
        
    audio = audio - np.mean(audio)
    audio_norm = audio / np.max(audio)
    
    pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
    audio_norm = norm11_interval(audio_norm, ti, tf, fs)

    return audio_norm, pressure_norm, name, fs

def find_longest_interval(times, max_gap=1):
    longest_interval = []
    current_interval = [times[0]]

    for i in range(1, len(times)):
        if times[i] - times[i - 1] <= max_gap:
            current_interval.append(times[i])
        else:
            if len(current_interval) > len(longest_interval):
                longest_interval = current_interval
            current_interval = [times[i]]

    if len(current_interval) > len(longest_interval):
        longest_interval = current_interval

    return longest_interval

class SpectrogramApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrogram Analysis")

        self.fig = plt.Figure(figsize=(12, 6))
        self.axs = self.fig.subplots(2, 1)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.indice = 0
        self.ti = 0
        self.tf = 1
        self.directory = ""
        self.Sonidos = []
        self.Presiones = []

        self.create_input_fields()

        self.canvas.mpl_connect("button_press_event", self.on_click)

    def load_data(self):
        self.audio, self.pressure, self.name, self.fs = datos_normalizados_2(self.indice, self.ti, self.tf, self.Sonidos, self.Presiones)
        self.time = np.linspace(0, len(self.pressure) / self.fs, len(self.pressure))

        self.f, self.t, self.Sxx = signal.spectrogram(self.audio, self.fs)
        self.Sxx_dB = np.log(self.Sxx)
        self.frequency_cutoff = 1000
        self.threshold = -10

        freq_indices = np.where(self.f > self.frequency_cutoff)[0]
        self.time_indices = np.any(self.Sxx_dB[freq_indices, :] > self.threshold, axis=0)
        self.time_above_threshold = self.t[self.time_indices]
        self.longest_interval = find_longest_interval(self.time_above_threshold)
        
        self.peaks_maximos, self.properties_maximos = find_peaks(self.pressure, prominence=1, height=0, distance=int(self.fs * 0.1))

    def plot_data(self):
        self.fig.suptitle(f'{self.name}')
        self.axs[0].cla()
        self.axs[1].cla()

        self.axs[0].plot(self.time, self.audio)
        self.axs[0].axvline(x=self.longest_interval[0], color='k', linestyle='-')
        self.axs[0].axvline(x=self.longest_interval[-1], color='k', linestyle='-', label='Avion')
        self.axs[0].axhline(y=1, color='k', linestyle='-')
        self.axs[0].axhline(y=-1, color='k', linestyle='-')
        self.axs[0].set_ylabel("Audio (arb. u.)")
        self.axs[0].legend(fancybox=True, shadow=True)

        self.axs[1].plot(self.time, self.pressure)
        self.axs[1].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        for i in range(len(self.peaks_maximos)):
            self.axs[1].text(self.time[self.peaks_maximos[i]], self.pressure[self.peaks_maximos[i]], str(i))
        self.axs[1].axvline(x=self.longest_interval[0], color='k', linestyle='-')
        self.axs[1].axvline(x=self.longest_interval[-1], color='k', linestyle='-', label='Avion')
        self.axs[1].axhline(y=1, color='k', linestyle='-')
        self.axs[1].axhline(y=-1, color='k', linestyle='-')
        self.axs[1].set_ylabel("Pressure (arb. u.)")
        self.axs[1].legend(fancybox=True, shadow=True)
        self.axs[1].set_xlabel("Time (sec)")
        self.fig.tight_layout()
        self.canvas.draw()

    def create_input_fields(self):
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.directory_button = tk.Button(self.input_frame, text="Choose Directory", command=self.choose_directory)
        self.directory_button.pack(side=tk.LEFT)

        self.indice_label = tk.Label(self.input_frame, text="Indice:")
        self.indice_label.pack(side=tk.LEFT)
        self.indice_entry = tk.Entry(self.input_frame)
        self.indice_entry.pack(side=tk.LEFT)

        self.ti_label = tk.Label(self.input_frame, text="ti:")
        self.ti_label.pack(side=tk.LEFT)
        self.ti_entry = tk.Entry(self.input_frame)
        self.ti_entry.pack(side=tk.LEFT)

        self.tf_label = tk.Label(self.input_frame, text="tf:")
        self.tf_label.pack(side=tk.LEFT)
        self.tf_entry = tk.Entry(self.input_frame)
        self.tf_entry.pack(side=tk.LEFT)

        self.update_button = tk.Button(self.input_frame, text="Update", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT)
        
        self.save_button = tk.Button(self.input_frame, text="Save", command=self.save_peaks)
        self.save_button.pack(side=tk.LEFT)

    def choose_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.directory = directory
            os.chdir(directory)
            
            lugar = os.listdir(directory)
            self.Presiones = []
            self.Sonidos = []

            for file in lugar:
                if file[0] == 's':
                    self.Sonidos.append(file)
                elif file[0] == 'p':
                    self.Presiones.append(file)
            
            self.update_plot()

    def update_plot(self):
        try:
            self.indice = int(self.indice_entry.get())
            self.ti = float(self.ti_entry.get())
            self.tf = float(self.tf_entry.get())
        except ValueError:
            tk.messagebox.showerror("Invalid input", "Please enter valid numerical values for indice, ti, and tf.")
            return

        self.load_data()
        self.plot_data()

    def on_click(self, event):
        if event.inaxes == self.axs[1]:
            x_click = event.xdata
            y_click = event.ydata

            if self.peaks_maximos.size > 0:
                distances = np.sqrt((self.time[self.peaks_maximos] - x_click)**2 + (self.pressure[self.peaks_maximos] - y_click)**2)
                closest_peak_index = np.argmin(distances)
                if distances[closest_peak_index] < 0.1:
                    self.peaks_maximos = np.delete(self.peaks_maximos, closest_peak_index)
                    self.plot_data()

    def save_peaks(self):
        np.savetxt(f'{self.name}_maximos2.txt', self.peaks_maximos, delimiter=',', newline='\n', fmt='%i')

def main():
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

#%%

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
    
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrogram Analysis")
    

    
            # Create a Figure object without setting it as the current figure
        self.fig = plt.Figure(figsize=(12, 6))
        # self.fig.suptitle('Audio vs Pressure')
        self.axs = self.fig.subplots(2, 1)  # Create subplots directly from the Figure object
        self.fig.tight_layout()
        # Create a canvas for the matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
        # Add the toolbar for interactivity
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
    
        self.indice = 0
        self.ti = 0
        self.tf = 1
    
        self.load_data()
        self.plot_data()
    
        self.create_input_fields()
    
        # Connect event handlers for interactive peak removal
        self.canvas.mpl_connect("button_press_event", self.on_click)
    
    def load_data(self):
        self.audio, self.pressure, self.name, self.fs = datos_normalizados_2(self.indice, self.ti, self.tf)
        self.time = np.linspace(0, len(self.pressure) / self.fs, len(self.pressure))
    
        self.f, self.t, self.Sxx = signal.spectrogram(self.audio, self.fs)
        self.Sxx_dB = np.log(self.Sxx)
        self.frequency_cutoff = 1000
        self.threshold = -10
    
        freq_indices = np.where(self.f > self.frequency_cutoff)[0]
        self.time_indices = np.any(self.Sxx_dB[freq_indices, :] > self.threshold, axis=0)
        self.time_above_threshold = self.t[self.time_indices]
        self.longest_interval = find_longest_interval(self.time_above_threshold)
        
        self.peaks_maximos, self.properties_maximos = signal.find_peaks(self.pressure, prominence=1, height=0, distance=int(self.fs * 0.1))
    
    def plot_data(self):
        self.fig.suptitle(f'{self.name}')
        self.axs[0].cla()
        self.axs[1].cla()
    
        self.axs[0].plot(self.time, self.audio)
        self.axs[0].axvline(x=self.longest_interval[0], color='k', linestyle='-')
        self.axs[0].axvline(x=self.longest_interval[-1], color='k', linestyle='-', label='Avion')
        self.axs[0].axhline(y=1, color='k', linestyle='-')
        self.axs[0].axhline(y=-1, color='k', linestyle='-')
        self.axs[0].set_ylabel("Audio (arb. u.)")
        self.axs[0].legend(fancybox=True, shadow=True)
    
        self.axs[1].plot(self.time, self.pressure)
        self.axs[1].plot(self.time[self.peaks_maximos], self.pressure[self.peaks_maximos], '.C1', label='Maximos', ms=10)
        for i in range(len(self.peaks_maximos)):
            self.axs[1].text(self.time[self.peaks_maximos[i]], self.pressure[self.peaks_maximos[i]], str(i))
        self.axs[1].axvline(x=self.longest_interval[0], color='k', linestyle='-')
        self.axs[1].axvline(x=self.longest_interval[-1], color='k', linestyle='-', label='Avion')
        self.axs[1].axhline(y=1, color='k', linestyle='-')
        self.axs[1].axhline(y=-1, color='k', linestyle='-')
        self.axs[1].set_ylabel("Pressure (arb. u.)")
        self.axs[1].legend(fancybox=True, shadow=True)
        self.axs[1].set_xlabel("Time (sec)")
        self.fig.tight_layout()
        self.canvas.draw()
    
    def create_input_fields(self):
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
        self.indice_label = tk.Label(self.input_frame, text="Indice:")
        self.indice_label.pack(side=tk.LEFT)
        self.indice_entry = tk.Entry(self.input_frame)
        self.indice_entry.pack(side=tk.LEFT)
    
        self.ti_label = tk.Label(self.input_frame, text="ti:")
        self.ti_label.pack(side=tk.LEFT)
        self.ti_entry = tk.Entry(self.input_frame)
        self.ti_entry.pack(side=tk.LEFT)
    
        self.tf_label = tk.Label(self.input_frame, text="tf:")
        self.tf_label.pack(side=tk.LEFT)
        self.tf_entry = tk.Entry(self.input_frame)
        self.tf_entry.pack(side=tk.LEFT)
    
        self.update_button = tk.Button(self.input_frame, text="Update", command=self.update_plot)
        self.update_button.pack(side=tk.LEFT)
        
        self.save_button = tk.Button(self.input_frame, text="Save", command=self.save_peaks)
        self.save_button.pack(side=tk.LEFT)
        
    def update_plot(self):
        try:
            self.indice = int(self.indice_entry.get())
            self.ti = float(self.ti_entry.get())
            self.tf = float(self.tf_entry.get())
        except ValueError:
            tk.messagebox.showerror("Invalid input", "Please enter valid numerical values for indice, ti, and tf.")
            return
    
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
    
    
    def save_peaks(self):
       np.savetxt(f'{self.name}_maximos2.txt', self.peaks_maximos, delimiter=',',newline='\n',fmt='%i') 
   


def main():
    root = tk.Tk()
    app = SpectrogramApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()     