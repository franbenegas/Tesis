# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:03:15 2024

@author: beneg
"""

import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from IPython import get_ipython
from scipy.signal import butter, sosfiltfilt
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import seaborn as sns
import signal_envelope as se

get_ipython().run_line_magic('matplotlib', 'qt5')

# Specify the directory containing the files
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
os.chdir(directory)

Datos = pd.read_csv('adq-log.txt', delimiter='\t') 
#%%


def datos_normalizados(indice):
    
    #La funcion que normaliza. 
    def normalizar(x, mean, max):
      #x: el array con la señal que levantas en .wav
      #mean y max: el valor medio y el maximo de la señal. Esta info esta en el txt asociado a esa medicion 
      return (x / np.max(np.abs(x))) * max + mean
    
    Pressure = Datos.loc[indice,'pressure_fname']
    Sound = Datos.loc[indice,'sound_fname']

    fs,audio = wavfile.read(Sound)
    ps,pressure = wavfile.read(Pressure)
    
    #normalizamos aca
    #levantamos el archivo txt que tiene info de la normalizacion
    data_norm = np.loadtxt(Pressure[9:-4] + '.txt',delimiter=',',skiprows=1)
    name = Pressure[9:-4]
    #aca asignamos los valores medios y los maximos de sonido y presion (revisar si los indices estan bien puestos)
    mean_s, max_s = data_norm[0], data_norm[2]
    mean_p, max_p = data_norm[1], data_norm[3]

    #la info de los maximos y las medias se usa para normalizar:
    pressure_norm = normalizar(pressure, mean_p, max_p)
    audio_norm = normalizar(audio, mean_s, max_s)
    
    return audio_norm, pressure_norm, name, fs

def plot_sound_vs_pressure(indice:int,Datos:pd):
    
    audio, pressure, name,fs = datos_normalizados(indice)
    
    
    time = np.linspace(0, len(audio)/fs, len(audio))
    
    freq_audio, fft_result_audio = signal.periodogram(audio, fs)
    freq_pressure, fft_result_pressure = signal.periodogram(pressure, fs)
    
    plt.figure(figsize=(14,7))
    plt.suptitle(f'{indice} = {name}')
    
    
    plt.subplot(2,4,(1,2))
    plt.title('Sound')
    plt.plot(time, audio)
    plt.xlabel('Time [sec]')
    plt.grid(linestyle='dashed')
    
    plt.subplot(2,4,(3,4))
    plt.title('Pressure')
    plt.plot(time, pressure)
    plt.xlabel('Time [sec]')
    plt.grid(linestyle='dashed')
    
    
    # Plot PSD of audio signal
    plt.subplot(2, 4, (5,6))
    plt.plot(freq_audio[1:], fft_result_audio[1:])
    plt.title('fft Audio')
    plt.xlim(0,500)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.grid(linestyle='dashed')

    # Plot PSD of pressure signal
    plt.subplot(2, 4, (7,8))
    plt.plot(freq_pressure[1:], fft_result_pressure[1:])
    plt.title('fft Pressure')
    plt.xlim(0,10)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.grid(linestyle='dashed')
    
    plt.tight_layout()
    
# Design a band-pass filter
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

def find_zero_crossings(time, data):
    """
    Find indices where the data crosses zero using Bolzano's theorem.
    
    Parameters:
        time (array-like): 1D array of time values.
        data (array-like): 1D array of data values.
    
    Returns:
        list: Indices where the data crosses zero.
    """
    zero_crossings = []
    for i in range(1, len(data)):
        if (data[i] >= 0 and data[i-1] < 0) or (data[i] < 0 and data[i-1] >= 0):
            tiempo = (time[i-1] + time[i])/2
            zero_crossings.append(tiempo)
    return zero_crossings
#%%

indice = 268

plot_sound_vs_pressure(indice, Datos)
#%%

indice = 268


_, pressure, name, ps = datos_normalizados(indice)

#defino el tiempo
time = np.linspace(0, len(pressure)/ps, len(pressure))
#%%

#%%
def runge_kutta(f, t0, x0, alpha, h, tf):
    esp_temporal = np.arange(t0, tf + h, h)
    x_rungeado = np.zeros([len(esp_temporal), len(x0)])  # Adjusted to handle multiple variables

    x_rungeado[0] = x0

    for i, ti in enumerate(esp_temporal[:-1]):  # Adjusted loop range
        K1 = f(ti, x_rungeado[i], alpha) * h
        K2 = f(ti + h / 2, x_rungeado[i] + K1 / 2, alpha) * h
        K3 = f(ti + h / 2, x_rungeado[i] + K2 / 2, alpha) * h
        K4 = f(ti + h, x_rungeado[i] + K3, alpha) * h  # Corrected K4 calculation

        x_rungeado[i + 1] = x_rungeado[i] + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

    return x_rungeado, esp_temporal

def suavisador(time, pressure, alpha):
    # pressure = pressure[ps*time[0]:ps*time[-1]]
    dx_dt = -alpha * time + abs(pressure)  # Adjusted equation
    return dx_dt
#%%
t0 = 10
tf = 20#time[-1]  # Adjust final time as needed
x0 = np.array([pressure[ps*t0], 0])  # Adjust initial values as needed
alpha = 0.005  # Adjust damping coefficient as needed
h = 1/1000#ps  # Adjust time step size as needed

x_rungeado, esp_temporal = runge_kutta(suavisador, t0, x0, alpha, h, tf)

limpio,_ = np.transpose(x_rungeado)

plt.close("SUavisado")
plt.figure("SUavisado",figsize=(14,7))
plt.plot(time,abs(pressure))
plt.plot(esp_temporal,limpio, marker = '.')
#%%

# plt.close("SUavisado")
# plt.figure("SUavisado",figsize=(14,7))
# plt.plot(time,pressure)
all_x_rungeado = []
all_esp_temporal = []

initial_pressure = 0  # Initial pressure value
for i in range(60):
    t0 = i
    x0 = np.array([initial_pressure, 0])  # Initial pressure and velocity
    alpha = 0.003  # Adjust damping coefficient as needed
    h = 1 / 100#ps  # Adjust time step size as needed
    tf = i + 1  # Adjust final time as needed

    # Run the Runge-Kutta method
    x_rungeado, esp_temporal = runge_kutta(suavisador, t0, x0, alpha, h, tf)
    
    # Update the initial pressure for the next iteration
    initial_pressure = x_rungeado[-1, 0]  # Last pressure value from the current iteration

    # Store results of current iteration
    all_x_rungeado.append(x_rungeado)
    all_esp_temporal.append(esp_temporal)

# Concatenate results of all iterations
all_x_rungeado = np.concatenate(all_x_rungeado)
all_esp_temporal = np.concatenate(all_esp_temporal)
limpio,_ = np.transpose(all_x_rungeado)

plt.close("SUavisado")
plt.figure("SUavisado",figsize=(14,7))
plt.plot(time,pressure)
plt.plot(all_esp_temporal,limpio, marker = '.')
#%% Filtro ancho de bandas

lowcut = 0.7 # Low cutoff frequency in Hz
highcut = 3# High cutoff frequency in Hz
fs = ps  # Sampling frequency in Hz
order = 6  # Filter order

# Apply the band-pass filter
filtered_signal = butter_bandpass_filter(pressure, lowcut, highcut, fs, order=order)
plt.close("Butter filter")
plt.figure("Butter filter",figsize=(14,7))
plt.title(f'{highcut} - {lowcut} Hz')
plt.plot(time,pressure,label='Señal',alpha=0.5)
plt.plot(time,filtered_signal,label='Filtrado')
plt.legend()
plt.grid(linestyle='dashed')

#%% Busco maximos del filtrado con promiencia mayor a 0.1 y mayor al cero
peaks,properties = find_peaks(filtered_signal,prominence=0.1,height=0)


plt.close(fig="Filtros y maximos")
plt.figure("Filtros y maximos",figsize=(14,7))
plt.plot(time,filtered_signal,label='Filtrado')
plt.plot(time[peaks],filtered_signal[peaks],'.g')
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.title('Filtros y maximos')

#%% Hago el filtro savgol

window_length = 101  # Window length (odd number)
polyorder = 6  # Polynomial order
filtered_data = savgol_filter(pressure, window_length, polyorder)

plt.close("Filtro savgol")
plt.figure("Filtro savgol",figsize=(14,7))
plt.plot(time,pressure,label='Presion')
plt.plot(time,filtered_data,label='Filtro savgol')
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.legend()

#%% Comparacion de filtros
plt.close("Comparacion de filtros")
plt.figure("Comparacion de filtros",figsize=(14,7))
plt.plot(time,filtered_signal,label='Filtro_butter')
plt.plot(time,filtered_data,label='Filtro savgol',alpha=0.5)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.legend()

#%% Comparo la transformada de la señal con la del filstro savgol
freq_pressure, fft_result_pressure = signal.periodogram(pressure, fs)
freq_pressure_filtrada, fft_result_pressure_filtrada = signal.periodogram(filtered_data, fs)

plt.close("FFT datos y savgol")
plt.figure("FFT datos y savgol",figsize=(14,7))
plt.subplot(2,2,(1,2))
plt.plot(freq_pressure, fft_result_pressure)
plt.title('fft Pressure')
plt.xlim(0,10)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.grid(linestyle='dashed')
plt.subplot(2,2,(3,4))
plt.plot(freq_pressure_filtrada, fft_result_pressure_filtrada)
plt.title('fft Filtrada')
plt.xlim(0,10)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.grid(linestyle='dashed')
plt.tight_layout()

#%% Busco maximos en la presion posta

peaks_maximos,properties_maximos = find_peaks(pressure,prominence=0.1,height=0, 
                                              distance=8000)

plt.close(fig="Maximos orginales")
plt.figure("Maximos originales",figsize=(14,7))
plt.plot(time,pressure,label='Presion')
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.title('Filtros y maximos')
plt.legend()

#%% Aca voy a ver los minimios

peaks_minimos,properties_minimos = find_peaks((-1)*pressure,prominence=0.1,height=0)

plt.close(fig="Minimos orginales")
plt.figure("Minimos originales",figsize=(14,7))
plt.plot(time,pressure,label='Presion')
plt.plot(time[peaks_minimos],pressure[peaks_minimos],'.C1',label='Minimos')
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.title('Filtros y minimos')
plt.legend()

#%% Plot de los maximos y minimos en funcion del tiempo
t_max,p_max = time[peaks_maximos],pressure[peaks_maximos]
t_min,p_min = time[peaks_minimos],pressure[peaks_minimos]


plt.close(fig="Max y min vs tiempo")
plt.figure("Max y min vs tiempo",figsize=(14,7))

plt.subplot(2,2,(1,2))
# ploteo de los maximos
plt.plot(t_max,np.arange(len(t_max)),'.C1',label = 'Maximos')


#Ploteo de los minimos
plt.plot(t_min,np.arange(len(t_min)),'.C2',label='Minimos')

plt.xlabel('Time [sec]')
plt.ylabel('Numero de maximo')
plt.grid(linestyle='dashed')

plt.legend()

plt.subplot(2,2,(3,4))
# ploteo de los maximos
plt.plot(t_max,p_max,'.C1',label = 'Maximos')

#Ploteo de los minimos
plt.plot(t_min,p_min,'.C2',label='Minimos')
plt.xlabel('Time [sec]')
plt.ylabel('Numero de maximo')
plt.grid(linestyle='dashed')
plt.legend()
#%% Grafico la funcion con solo max y min

plt.close(fig="Datos y extremos")
plt.figure("Datos y extremos",figsize=(14,7))
plt.plot(time,pressure,label='Presion',alpha=0.5)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos')
plt.plot(time[peaks_minimos],pressure[peaks_minimos],'.C2',label='Minimos')
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')

plt.legend()

#%%
plt.figure()
# plt.subplot(1,1,1)
# sns.jointplot(data = None,x= t_max,y=  np.arange(len(t_max)),marginal_kws=dict(kde=True))
# sns.jointplot(data = None,x=t_min,y=np.arange(len(t_min)),marginal_kws=dict(kde=True))
sns.histplot(x= t_max,bins=10, kde=True,label = 'Maximos')
sns.histplot(x= t_min,bins=10, kde=True,label = 'Minimos')
plt.xlabel('Time [sec]')
plt.legend()
#%%
plt.figure()
bins = np.histogram_bin_edges(t_max, bins='auto')
sns.histplot(x= t_max,bins=bins, kde=True,label = 'Maximos')
x= plt.get_height(bins)
#%% Busco los ceros de los datos

# Find zero crossings in the data
zero_crossings = find_zero_crossings(time, filtered_data)


fig, ax1 = plt.subplots(figsize=(14,7))
ax1.plot(t_max,np.arange(len(t_max)),'.C1',label = 'Maximos')
ax1.set_xlabel('Time [sec]')
ax1.set_ylabel('Numero de Maximo y Minimos')
ax1.grid(linestyle='dashed')
ax1.plot(t_min,np.arange(len(t_min)),'.C2',label='Minimos')
ax1.legend()
ax2 = ax1.twinx()
ax2.plot(zero_crossings,np.arange(len(zero_crossings)),'.C0',label='Zeros')
ax2.set_ylabel('Numeros de zeros')
ax2.legend(loc='lower right')

#%% Grafico los extremos con los ceros de los datos

plt.close(fig="Extremos y ceros vs tiempo")
plt.figure("Extremos y ceros vs tiempo",figsize=(14,7))
# plt.plot(time,pressure,label='Presion',alpha=0.5)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos')
plt.plot(time[peaks_minimos],pressure[peaks_minimos],'.C2',label='Minimos')
plt.plot(zero_crossings,np.zeros(len(zero_crossings)),'.C0',label='Zeros')
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')

plt.legend()



#%%

def densidad_de_puntos(tiempo: np.ndarray, datos: np.ndarray, longitud_intervalo:int):
    
    interval_length = longitud_intervalo
    intervalos = np.arange(np.min(tiempo), np.max(tiempo) + interval_length, interval_length)
    
    densidad = np.zeros(len(intervalos) - 1)
    
    for i in range(len(intervalos) - 1):
        # Find data points within the current interval
        points_in_interval = datos[(tiempo >= intervalos[i]) & (tiempo < intervalos[i + 1])]
        # Calculate the density of points within the interval
        densidad[i] = len(points_in_interval) / interval_length
    
    return intervalos, densidad

longitud_intervalo = 3

intervalos_max, densidad_max = densidad_de_puntos(t_max, p_max, longitud_intervalo)
intervalos_min, densidad_min = densidad_de_puntos(t_min, p_min, longitud_intervalo)
intervalos_zeros, densidad_zeros = densidad_de_puntos(zero_crossings,np.zeros(len(zero_crossings)), longitud_intervalo)


plt.close(fig="Densidades")
plt.figure("Densidades",figsize=(14,7))
plt.title(f'longitud del intervalo = {longitud_intervalo}')
plt.plot(intervalos_max[:-1],densidad_max, marker = '.', linestyle='--', label = 'Maximos')
plt.plot(intervalos_min[:-1],densidad_min,marker = '.', linestyle='--',label = 'Minimos')
# plt.plot(intervalos_zeros[:-1],densidad_zeros,marker = '.', linestyle='--', label = 'Zeros')
plt.grid(linestyle='dashed')
plt.ylabel('Densidad')
plt.xlabel('Tiempo')
plt.legend()

#%% Calculo las derivadas
peaks,properties = find_peaks(densidad_max)
derivadas = np.concatenate(([densidad_max[0]],np.diff(densidad_max)))
peaks_2,properties_2 = find_peaks(derivadas)
plt.close(fig="Derivadas y maximos")
plt.figure("Derivadas y maximos",figsize=(14,7))
plt.plot(intervalos_max[:-1],densidad_max, marker = '.', linestyle='--', label = 'Maximos')
plt.plot(intervalos_max[:-1],derivadas, marker = '.', linestyle='--', label = 'Derivadas')
plt.plot(intervalos_max[peaks],densidad_max[peaks],marker = '.', linestyle='--',label='Picos')
plt.plot(intervalos_max[peaks_2],derivadas[peaks_2],marker = '.', linestyle='--',label='Picos')
plt.grid(linestyle='dashed')
plt.ylabel('Densidad')
plt.xlabel('Tiempo')
plt.legend()

#%% Este me agarra los intervalos pero con un tiempo puesto a mano

plt.close(fig="Intervalos puestos a mano")
plt.figure("Intervalos puestos a mano",figsize=(14,7))
plt.plot(time,filtered_data)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.ylabel('Presion')
for i in peaks:
    derecha = ps*int(intervalos_max[i] + 5)
    izquierda = ps*int(intervalos_max[i])
    
    maximo = max(pressure[izquierda:derecha])
    indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
    lugar_del_maximo = izquierda + indices_del_maximo 
    # print(lugar_del_maximo,maximo)
    
    plt.plot(time[izquierda:derecha],pressure[izquierda:derecha])
    plt.plot(time[lugar_del_maximo],pressure[lugar_del_maximo],'.k')


#%% Este me agarra todos los intervalos entre los maximos

inter = intervalos_max[peaks]
plt.close(fig="Intervalos entre maximos")
plt.figure("Intervalos entre maximos",figsize=(14,7))
plt.plot(time,pressure,label='Presion',alpha=0.5)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.ylabel('Presion')
for i in range(len(inter)-1):
    derecha = ps*int(inter[i+1])
    izquierda = ps*int(inter[i])
    
    
    maximo = max(pressure[izquierda:derecha])
    indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
    lugar_del_maximo = izquierda + indices_del_maximo 
    
    
    plt.plot(time[izquierda:derecha],pressure[izquierda:derecha])
    plt.plot(time[lugar_del_maximo],pressure[lugar_del_maximo],'.k')
    
#%%

plt.close(fig="Lol")
plt.figure("Lol",figsize=(14,7))
plt.plot(time,pressure,label='Presion',alpha=0.5)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.ylabel('Presion')
for i in peaks:
    
    inter_derecha = intervalos_max[i + 1]
    inter_izquierda = intervalos_max[i - 1]
    
    
    derecha = ps*int(inter_derecha)
    izquierda = ps*int(inter_izquierda)
    
    maximo = max(pressure[izquierda:derecha])
    indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
    lugar_del_maximo = izquierda + indices_del_maximo 
    
    
    plt.plot(time[izquierda:derecha],pressure[izquierda:derecha])
    plt.plot(time[lugar_del_maximo],pressure[lugar_del_maximo],'.k')
    
#%% Este me agarra segun las derivadas
plt.close(fig="Intervalos de interes")
plt.figure("Intervalos de interes",figsize=(14,7))
plt.plot(time,pressure,label='Presion',alpha=0.5)
plt.grid(linestyle='dashed')
plt.xlabel('Time [sec]')
plt.ylabel('Presion')
for i in peaks_2:

    inter_derecha = intervalos_max[i + 1]
    inter_izquierda = intervalos_max[i - 1]
    
    
    derecha = ps*int(inter_derecha)
    izquierda = ps*int(inter_izquierda)
    
    maximo = max(pressure[izquierda:derecha])
    indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
    lugar_del_maximo = izquierda + indices_del_maximo 
    
    
    plt.plot(time[izquierda:derecha],pressure[izquierda:derecha])
    plt.plot(time[lugar_del_maximo],pressure[lugar_del_maximo],'.k')
    
#%% Variacion del slicing

Metodo_1_lugares = []
Metodo_1 = []

Metodo_2_lugares = []
Metodo_2 = []

Metodo_3_lugares = []
Metodo_3 = []

Metodo_4_lugares = []
Metodo_4 = []

for i in range(1,len(t_max)):
    
    longitud_intervalo = i
    
    intervalos_max, densidad_max = densidad_de_puntos(t_max, p_max, longitud_intervalo)
    
    peaks,properties = find_peaks(densidad_max)
    derivadas = np.concatenate(([densidad_max[0]],np.diff(densidad_max)))
    peaks_2,properties_2 = find_peaks(derivadas)
    
    
    Lugar_A_mano = []
    A_mano = []
    
    Lugar_Entre_inter = []
    Entre_inter = []
    
    Lugar_Antes_y_despues_pico = []
    Antes_y_despues_pico = []
    
    Lugar_Antes_y_despues_derivada = []
    Antes_y_despues_derivada = []
    
    
    for i in peaks:
        
        #A mano
        derecha = ps*int(intervalos_max[i] + 5)
        izquierda = ps*int(intervalos_max[i])
        
        maximo = max(pressure[izquierda:derecha])
        indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
        lugar_del_maximo = izquierda + indices_del_maximo 
        
        Lugar_A_mano.append(lugar_del_maximo)
        A_mano.append(maximo)
        
        
        #Lugar antes y despues del pico
        
        inter_derecha = intervalos_max[i + 1]
        inter_izquierda = intervalos_max[i - 1]
        
        
        derecha = ps*int(inter_derecha)
        izquierda = ps*int(inter_izquierda)
        
        maximo = max(pressure[izquierda:derecha])
        indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
        lugar_del_maximo = izquierda + indices_del_maximo 
        
        Lugar_Antes_y_despues_pico.append(lugar_del_maximo)
        Antes_y_despues_pico.append(maximo)
    
    
    
    for i in peaks_2:
        
        inter_derecha = intervalos_max[i + 1]
        inter_izquierda = intervalos_max[i - 1]
        
        
        derecha = ps*int(inter_derecha)
        izquierda = ps*int(inter_izquierda)
        
        maximo = max(pressure[izquierda:derecha])
        indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
        lugar_del_maximo = izquierda + indices_del_maximo
        
        Lugar_Antes_y_despues_derivada.append(lugar_del_maximo)
        Antes_y_despues_derivada.append(maximo)
    
    
    inter = intervalos_max[peaks]
    
    for i in range(len(inter)-1):
        derecha = ps*int(inter[i+1])
        izquierda = ps*int(inter[i])
        
        
        maximo = max(pressure[izquierda:derecha])
        indices_del_maximo = np.where(pressure[izquierda:derecha] == maximo)[0][0]
        lugar_del_maximo = izquierda + indices_del_maximo 
        
        Lugar_Entre_inter.append(lugar_del_maximo)
        Entre_inter.append(maximo)
        
        
    Metodo_1.append(A_mano)
    Metodo_1_lugares.append(Lugar_A_mano)
    
    Metodo_2.append(Entre_inter)
    Metodo_2_lugares.append(Lugar_Entre_inter)
    
    Metodo_3.append(Antes_y_despues_pico)
    Metodo_3_lugares.append(Lugar_Antes_y_despues_pico)
    
    Metodo_4.append(Antes_y_despues_derivada)
    Metodo_4_lugares.append(Lugar_Antes_y_despues_derivada)

#%%
f, ((ax1,ax2),(ax3,ax4) )= plt.subplots(2, 2, figsize=(18,10))


for i in range(12):
    ax1.plot(time[Metodo_1_lugares[i]],Metodo_1[i],marker='.',linestyle='--',label=f'longitud del intervalo = {i+1}')
    ax2.plot(time[Metodo_2_lugares[i]],Metodo_2[i],marker='.',linestyle='--',label=f'longitud del intervalo = {i+1}')
    ax3.plot(time[Metodo_3_lugares[i]],Metodo_3[i],marker='.',linestyle='--',label=f'longitud del intervalo = {i+1}')
    ax4.plot(time[Metodo_4_lugares[i]],Metodo_4[i],marker='.',linestyle='--',label=f'longitud del intervalo = {i+1}')

# Create a single legend outside the subplots
# ax4.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, ncol=6)
ax1.legend(loc='upper right', fancybox=True, shadow=True)
ax2.legend(loc='upper right', fancybox=True, shadow=True)
ax3.legend(loc='upper right', fancybox=True, shadow=True)
ax4.legend(loc='upper right', fancybox=True, shadow=True)

# Set titles and labels for subplots
ax1.set_title('Metodo 1')
ax1.set_xlabel('Time [sec]')
ax1.grid(linestyle='dashed')

ax2.set_title('Metodo 2')
ax2.set_xlabel('Time [sec]')
ax2.grid(linestyle='dashed')

ax3.set_title('Metodo 3')
ax3.set_xlabel('Time [sec]')
ax3.grid(linestyle='dashed')

ax4.set_title('Metodo 4')
ax4.set_xlabel('Time [sec]')
ax4.grid(linestyle='dashed')

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()
    
#%%



def densidad_de_puntos(tiempo: np.ndarray, otro_tiempo: np.ndarray, longitud_intervalo: int, solapamiento: int = 1):
    """
    Calculate the density of points within each interval based on occurrences of otro_tiempo within tiempo intervals.

    Args:
        tiempo (np.ndarray): Array of time values.
        otro_tiempo (np.ndarray): Array of other time values to count occurrences of within tiempo intervals.
        longitud_intervalo (int): Length of each interval.
        solapamiento (int, optional): Number of elements by which consecutive intervals overlap. Defaults to 1.

    Returns:
        tuple: Tuple containing the time intervals and the density of points within each interval.
    """
    densidad = []

    # Generate slices with overlap
    for i in range(0, len(tiempo) - longitud_intervalo + 1, solapamiento):
        # Slice the tiempo array to define the interval
        intervalo_tiempo = tiempo[i:i+longitud_intervalo]
        # Count occurrences of otro_tiempo within the interval
        count_ocurrences = np.sum((otro_tiempo >= intervalo_tiempo[0]) & (otro_tiempo <= intervalo_tiempo[-1]))
        # Calculate the density based on the count of occurrences and the length of the interval
        densidad_intervalo = count_ocurrences / longitud_intervalo
        # Append the density to the list
        densidad.append(densidad_intervalo)

    return densidad


longitud_intervalo = 10
tiempo = np.arange(0, 60,0.1)
# Calculate density of points with overlap
densidad = densidad_de_puntos(tiempo, t_max, longitud_intervalo)

# print("Density of Points in Each Interval:", densidad)
plt.close(fig="nose_2")
plt.figure("nose_2",figsize=(14,7))
plt.plot(np.arange(len(densidad))/10,densidad)