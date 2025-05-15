# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 16:23:00 2024

@author: beneg
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
from plyer import notification
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from scipy import stats

# Set the font to 'STIX'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'


#%%

def process_night_data(directories):
    """
    Process data from multiple directories and return a combined time series list.

    Args:
        directories (list of str): List of directory paths.

    Returns:
        list of dict: Combined time series data from all directories.
    """
    def datos_normalizados(Sonidos,Presiones,indice, ti, tf):
        
        Sound, Pressure = Sonidos[indice], Presiones[indice]
        name = Pressure[9:-4]
        
        fs,audio = wavfile.read(Sound)
        fs,pressure = wavfile.read(Pressure)
       
        audio = audio-np.mean(audio)
        audio_norm = audio / np.max(audio)
        
        pressure = pressure-np.mean(pressure)
        pressure_norm = pressure / np.max(pressure)
        
        #funcion que normaliza al [-1, 1]
        def norm11_interval(x, ti, tf, fs):
          x_int = x[int(ti*fs):int(tf*fs)]
          return 2 * (x-np.min(x_int))/(np.max(x_int)-np.min(x_int)) - 1
            
        
        pressure_norm = norm11_interval(pressure_norm, ti, tf, fs)
        audio_norm = norm11_interval(audio_norm, ti, tf, fs)

        return audio_norm, pressure_norm, name, fs
    
        
    combined_time_series_list = []

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
            elif file[0] == 'D':
                datos.append(file)
            

        with open(datos[0], 'rb') as f:
            datos = pickle.load(f)
        
        for indice in range(datos.shape[0]):
            ti, tf = datos.loc[indice, 'Tiempo inicial normalizacion'], datos.loc[indice, 'Tiempo final normalizacion']
            tiempo_inicial = datos.loc[indice, 'Tiempo inicial avion']
            
            audio, pressure, name, fs = datos_normalizados(sonidos, presiones, indice, ti, tf)
            
            time = np.linspace(0, len(pressure)/fs, len(pressure))
            
            peaks_sonido, _ = find_peaks(audio, height=0, distance=int(fs*0.1), prominence=.001)
            # spline_amplitude_sound = UnivariateSpline(time[peaks_sonido], audio[peaks_sonido], s=0, k=3)
            
            # interpolado = spline_amplitude_sound(time)
            
            lugar_maximos = []
            maximos = np.loadtxt(f'{name}_maximos.txt')
            for i in maximos:
                lugar_maximos.append(int(i))

            periodo = np.diff(time[lugar_maximos])
            tiempo = time - tiempo_inicial
            combined_time_series_list.append({
                'time': tiempo[peaks_sonido],
                'sonido': audio[peaks_sonido], ### poner  ---> interpolado
                'time maximos': tiempo[lugar_maximos],
                'presion': pressure[lugar_maximos],
                'time rate': tiempo[lugar_maximos][1:],
                'rate': 1/periodo
            })

    return combined_time_series_list


def interpolate_single_data(time_series_list, data_key, time_key, common_time_length):
    """
    Interpolates a specific data series from the time series list.

    Args:
        time_series_list (list of dict): Combined time series data from all directories.
        data_key (str): The key to access the data to be interpolated (e.g., 'sonido', 'rate', 'presion').
        time_key (str): The key to access the corresponding time series (e.g., 'time', 'time rate', 'time maximos').
        common_time_length (int): Length of the common time base for interpolation.

    Returns:
        common_time_base (ndarray): The interpolated common time base.
        interpolated_data (ndarray): The interpolated data for the specified key.
    """
    # Calculate the common time base
    start_time = min(ts[time_key][0] for ts in time_series_list)
    end_time = max(ts[time_key][-1] for ts in time_series_list)
    common_time_base = np.linspace(start_time, end_time, common_time_length)
    
    # Interpolate the data onto the common time base
    interpolated_data = []
    for ts in time_series_list:
        interp_func = interp1d(ts[time_key], ts[data_key], bounds_error=False, fill_value=np.nan)
        interpolated_data.append(interp_func(common_time_base))
    
    interpolated_data = np.array(interpolated_data)
    
    return common_time_base, interpolated_data

def compute_average_and_std(data):
    average = np.nanmean(data, axis=0)
    count_non_nan = np.sum(~np.isnan(data), axis=0)
    std_error = np.nanstd(data, axis=0) / np.sqrt(count_non_nan)
    return average, std_error, count_non_nan


#%% Celda donde importo y proceso los datos

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
RoNe_noche = process_night_data(directories)

# Interpolate the sound data from RoNe_noche using 'sonido' as the data key and 'time' as the time key
# Specify a common time length of 44150 samples for sound
time_sonido_RoNe_noche, interpolated_sonido_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='sonido', time_key='time', common_time_length=44150
)

# Compute the average and standard deviation of the interpolated sound data
average_RoNe_noche_sonido, std_RoNe_noche_sonido, _ = compute_average_and_std(interpolated_sonido_RoNe_noche)

# Interpolate the pressure data from RoNe_noche using 'presion' as the data key and 'time maximos' as the time key
# Specify a common time length of 300 samples for pressure
time_maximos_RoNe_noche, interpolated_presion_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='presion', time_key='time maximos', common_time_length=300
)

# Compute the average and standard deviation of the interpolated pressure data
average_RoNe_noche_presion, std_RoNe_noche_presion, _ = compute_average_and_std(interpolated_presion_RoNe_noche)

# Interpolate the rate data from RoNe_noche using 'rate' as the data key and 'time rate' as the time key
# Specify a common time length of 300 samples for rate
time_rate_RoNe_noche, interpolated_rate_RoNe_noche = interpolate_single_data(
    RoNe_noche, data_key='rate', time_key='time rate', common_time_length=300
)

# Compute the average and standard deviation of the interpolated rate data
average_RoNe_noche_rate, std_RoNe_noche_rate, _ = compute_average_and_std(interpolated_rate_RoNe_noche)

# Notify the user when the program finishes execution
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # Optional: Path to an icon file
    timeout=10  # Notification will disappear after 10 seconds
)

#%%

from scipy.signal import butter, sosfiltfilt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

# Apply the band-pass filter


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_signal = sosfiltfilt(sos, data)
    return filtered_signal


lowcut = 0.01  # Low cutoff frequency in Hz
highcut = 20  # High cutoff frequency in Hz
fs = 44150  # Sampling frequency in Hz
order = 6  # Filter order

# Apply the band-pass filter
filtered_signal = butter_bandpass_filter(
    average_RoNe_noche_sonido, lowcut, highcut, fs, order=order)

time = time_sonido_RoNe_noche
dt = np.mean(np.diff(time))
Asp = np.concatenate(([0], np.diff(filtered_signal)))
#%%


from matplotlib.widgets import Slider
from scipy.stats import norm
from scipy.signal import find_peaks
from numba import njit
from scipy.signal import butter, sosfiltfilt

@njit
def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def sigm(x):
    return 1 / (1 + np.exp(-100 * x))

@njit
def f(v, t, pars):
    cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp = pars
    r, theta, mu, w = v
    
    # Equations
    drdt = mu * r - cr * r**3
    dthetadt = w - ci * r**2
    dmudt = (-mu + A_s0 * As)/ tau
    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) * sigm(Asp - 0.5)) * (A_s0 * As + w0 - w)
    
    return np.array([drdt, dthetadt, dmudt, dwdt])


@njit
def integrate_system(time, dt, pars, As_values, Asp_values):
    # Initial conditions
    r = np.zeros_like(time)
    theta = np.zeros_like(time)
    mu = np.zeros_like(time)
    w = np.zeros_like(time)
    r[0], theta[0], mu[0], w[0] = 0.25, 0, 0, 1

    for ix in range(len(time) - 1):
        pars[3] = As_values[ix]  # Modify As within the loop
        pars[-1] = Asp_values[ix]  # Modify Asp within the loop
        r[ix + 1], theta[ix + 1], mu[ix + 1], w[ix + 1] = rk4(f, np.array([r[ix], theta[ix], mu[ix], w[ix]]), time[ix], dt, pars)
    
    return r, theta, mu, w


fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)
plt.tight_layout()

# Grafico del promedio 
ax[0].errorbar(time_sonido_RoNe_noche, average_RoNe_noche_sonido- np.mean(average_RoNe_noche_sonido[0:100]), std_RoNe_noche_sonido, color='C0')
ax[1].errorbar(time_maximos_RoNe_noche, average_RoNe_noche_presion, std_RoNe_noche_presion, color='C0')
ax[2].errorbar(time_rate_RoNe_noche, average_RoNe_noche_rate, std_RoNe_noche_rate, color='C0')

plt.subplots_adjust(bottom=0.35)

p, = ax[1].plot([], [], 'r')
r_plot, = ax[2].plot([], [],'r')

slider_params = [
    (r'$c_r$', 0.1, 0.25, 0.3, 0.03, 0.0, 20.0, 3),
    (r'$\omega_0$', 0.1, 0.20, 0.3, 0.03, 0.0, 10.0, 2*np.pi),
    (r'$c_i$', 0.1, 0.15, 0.3, 0.03, -5.0, 6.0, 2*np.pi-1),
    (r'$\tau$', 0.1, 0.1, 0.3, 0.03, 0.0, 5.0, 1),
    (r'$A_{s0}$', 0.65, 0.25, 0.3, 0.03, 0.0, 5.0, 1),
    (r'$\tau_{up}$', 0.65, 0.20, 0.3, 0.03, 0.0, 7.0, 5),
    (r'$\tau_{down}$', 0.65, 0.15, 0.3, 0.03, 0.0, 20.0, 12),
]

sliders = {}
for name, left, bottom, width, height, min_val, max_val, init_val in slider_params:
    sliders[name] = Slider(plt.axes([left, bottom, width, height]), name, min_val, max_val, valinit=init_val)

for slider in sliders.values():
    slider.label.set_fontsize(14)
    
# Function to update plots
def update(val):
    # Convert all slider values to float32 to ensure consistency
    pars = np.array([
        sliders[r'$c_r$'].val, sliders[r'$\omega_0$'].val, sliders[r'$c_i$'].val, 1.0,
       sliders[r'$\tau$'].val, sliders[r'$A_{s0}$'].val, sliders[r'$\tau_{up}$'].val, sliders[r'$\tau_{down}$'].val, 1.0
       ], dtype=np.float64)
    
    r, theta, mu, w = integrate_system(time, dt, pars, average_RoNe_noche_sonido - np.mean(average_RoNe_noche_sonido[0:100]) , Asp)

    # peaks, _ = find_peaks(r * np.cos(theta), height=0, distance=int(0.1 / dt))
    
    # Update plots
    p.set_data(time, r )
    r_plot.set_data(time, w)
    ax[1].relim()
    ax[1].autoscale_view()
    ax[2].relim()
    ax[2].autoscale_view()
    fig.canvas.draw_idle()

# Connect sliders to the update function
for slider in sliders.values():
    slider.on_changed(update)

# Initial call to update the plot
update(None)