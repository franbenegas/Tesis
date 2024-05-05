# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:46:15 2024

@author: beneg
"""

import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt5')

# Specify the directory containing the files
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
os.chdir(directory)

#%% Organizacion de archivos 

file_counts = {}

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        _, extension = os.path.splitext(filename)  
        
        Separate = _.split('-') # Me los separa en funcion del espaciador '-'
        
        # Esta parte me cuenta todos los archivos que tienen el mismo final, es decir el tiempo
        if Separate[-1].lower() in file_counts:
            file_counts[Separate[-1].lower()] += filename,
        else:
            file_counts[Separate[-1].lower()] = filename,
txt,pressure,sound = file_counts['23.59.33']
print(pressure)

ps,pressure = wavfile.read(pressure)
plt.plot(pressure)
#%%
Datos = pd.read_csv('adq-log.txt', delimiter='\t') 



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
    
    return audio_norm, pressure_norm, name

def plot_sound_vs_pressure(indice:int):
    
    audio, pressure, name = datos_normalizados(indice)
    
    
    time = np.linspace(0, len(audio)/fs, len(audio))
    
    freq_audio, fft_result_audio = signal.periodogram(audio, fs)
    freq_pressure, fft_result_pressure = signal.periodogram(pressure, ps)
    
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
    print(f'{indice} = ', Sound)
    
# def plot_sound_vs_pressure(indice:int,Datos:pd):
    
#     Pressure = Datos.loc[indice,'pressure_fname']
#     Sound = Datos.loc[indice,'sound_fname']

#     fs,audio = wavfile.read(Sound)
#     ps,pressure = wavfile.read(Pressure)
    
#     time = np.linspace(0, len(audio)/fs, len(audio))
    
#     freq_audio, fft_result_audio = signal.periodogram(audio, fs)
#     freq_pressure, fft_result_pressure = signal.periodogram(pressure, ps)

#     frequencies_audio, times_audio, spectrogram_audio = signal.spectrogram(audio, fs)
#     frequencies_pressure, times_pressure, spectrogram_pressure = signal.spectrogram(pressure, ps)


#     plt.figure(figsize=(14,7))
#     plt.suptitle(f'Indice = {indice}')
    
#     plt.subplot(3,4,(1,2))
#     plt.title('Sound')
#     plt.plot(time, audio)
#     plt.xlabel('Time [sec]')
    
#     plt.subplot(3,4,(3,4))
#     plt.title('Pressure')
#     plt.plot(time, pressure)
#     plt.xlabel('Time [sec]')
    
#     # Plot PSD of audio signal
#     plt.subplot(3, 4, (5,6))
#     plt.plot(freq_audio[1:], fft_result_audio[1:])
#     plt.title('fft Audio')
#     plt.xlim(0,500)
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Power')

#     # Plot PSD of pressure signal
#     plt.subplot(3, 4, (7,8))
#     plt.plot(freq_pressure[1:], fft_result_pressure[1:])
#     plt.title('fft Pressure')
#     plt.xlim(0,10)
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Power')
    
#     # Plot spectrogram of audio signal
#     plt.subplot(3, 4, (9,10))
#     plt.pcolormesh(times_audio, frequencies_audio, np.log(spectrogram_audio))
#     plt.colorbar(label='Log Power')
#     plt.title('Spectrogram of Audio')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')

#     # Plot spectrogram of pressure signal
#     plt.subplot(3, 4, (11,12))
#     plt.pcolormesh(times_pressure, frequencies_pressure, np.log(spectrogram_pressure))
#     plt.colorbar(label='Log Power')
#     plt.title('Spectrogram of Pressure')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')

#     plt.tight_layout()
#     print(f'{indice} = ', Sound)
    
    
#%%
# plt.close('all')   
 
plot_sound_vs_pressure(347, Datos)
#%% Grafico de Maximos y minimos

plt.close('all')
plt.figure(figsize=(14,7))


sound_max = Datos['sound_max']
sound_min = Datos['sound_min']
pressure_max = Datos['pressure_max']
pressure_min = Datos['pressure_min']

plt.subplot(2,2,(1,2))
plt.title('Sound extremes')
plt.plot(sound_max,color ='r',marker='.',linestyle='dashed',label='maximos')
plt.plot(sound_min,'b',marker='.',linestyle='dashed',label='minimos')
plt.vlines(268,Datos.loc[268,'sound_min'],Datos.loc[268,'sound_max'],'k',linestyles='dashed',label='Avion')
plt.legend()
plt.subplot(2,2,(3,4))
plt.title('Pressure extremes')
plt.plot(pressure_max,'r',marker='.',linestyle='dashed',label='maximos')
plt.plot(pressure_min,'b',marker='.',linestyle='dashed',label='minimos')
plt.vlines(268,Datos.loc[268,'pressure_min'],Datos.loc[268,'pressure_max'],'k',linestyles='dashed',label='Avion')
plt.legend()


#%% Caluclo de los Zig-Zag en comun

diff_sound_max = np.concatenate(([sound_max[0]],np.diff(sound_max)))
diff_pressure_max = np.concatenate(([pressure_max[0]],np.diff(pressure_max)))
# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(14,7))
plt.title('Zig-Zag')

# Plot diff on the first y-axis
ax1.plot(diff_sound_max, marker='.', linestyle='dashed', color='b', label='max Diff')
ax1.vlines(268,ymin=-0.1,ymax=0.1,color='k',linestyles='dashed',label='Avion')
ax1.set_ylabel('Sound Diff', color='b')
ax1.legend()

# Create a secondary y-axis and plot diff2 on it
ax2 = ax1.twinx()
ax2.plot(diff_pressure_max, marker='.', linestyle='dashed', color='r', label='Pressure Diff')
ax2.set_ylabel('Pressure Diff', color='r')



Indices_sound = np.where(diff_sound_max == 0)
Indices_pressure = np.where(diff_pressure_max == 0)

Zig_Zag_Sound = []

for i in range(len(Indices_sound[0])):
    diff_extremos = diff_sound_max[Indices_sound[0][i]-1] - diff_sound_max[Indices_sound[0][i]+1]

    if diff_extremos > 0:
        Zig_Zag_Sound.append(Indices_sound[0][i]-1) #Sacable
        Zig_Zag_Sound.append(Indices_sound[0][i])

Zig_Zag_pressure = []

for i in range(len(Indices_pressure[0])):
    diff_extremos = diff_pressure_max[Indices_pressure[0][i]-1] - diff_pressure_max[Indices_pressure[0][i]+1]

    if diff_extremos > 0:
        Zig_Zag_pressure.append(Indices_pressure[0][i-1]) #Sacable
        Zig_Zag_pressure.append(Indices_pressure[0][i])

Valores_comunes = np.intersect1d(Indices_sound,Indices_pressure)

Valores_comunes_zig_zag = np.intersect1d(np.array(Zig_Zag_Sound), np.array(Zig_Zag_pressure))


#%% Grafico de los ancho altura mitad de uno en especifico

indice = 1130
# Pressure = Datos.loc[indice,'pressure_fname']
# Sound = Datos.loc[indice,'sound_fname']

# fs,audio = wavfile.read(Sound)
# ps,pressure = wavfile.read(Pressure)
fs = 44150
audio, pressure = datos_normalizados(indice)

time = np.linspace(0, len(audio)/fs, len(audio))

sound_max = max(audio)
pressure_max = max(pressure)

Indices_sound_FWH = np.where(audio >= sound_max/2)
Indices_pressure_FWH = np.where(pressure >= pressure_max/2)

plt.figure(figsize=(16, 8))
plt.suptitle(f'indice = {indice}')


plt.subplot(2, 2, (1,2))
plt.plot(time, audio)
plt.plot()
plt.hlines(sound_max/2,xmin=time[0],xmax=time[-1],color='k',linestyle='dashed',label = 'Media altura')
plt.vlines(time[Indices_sound_FWH[0][0]],ymin=min(audio),ymax=max(audio),color='k',linestyles='dashed',label='Ancho')
plt.vlines(time[Indices_sound_FWH[0][-1]],ymin=min(audio),ymax=max(audio),color='k',linestyles='dashed')
plt.title('Audio')

plt.xlabel('Time [sec]')
plt.ylabel('Amplitud')
plt.legend()

plt.subplot(2, 2, (3,4))
plt.plot(time,pressure)
plt.hlines(pressure_max/2,xmin=time[0],xmax=time[-1],color='k',linestyle='dashed',label = 'Media altura')
plt.vlines(time[Indices_pressure_FWH[0][0]],ymin=min(pressure),ymax=max(pressure),color='k',linestyles='dashed',label='Ancho')
plt.vlines(time[Indices_pressure_FWH[0][-1]],ymin=min(pressure),ymax=max(pressure),color='k',linestyles='dashed')
plt.title('Pressure')

plt.xlabel('Time [sec]')
plt.ylabel('Amplitud')
plt.legend()

plt.tight_layout()

#%% Calculo de los ancho altura mitad de los valores comunes

FWH_sound = []
FWH_pressure = []

for indice in Valores_comunes_zig_zag:
    
    # Pressure = Datos.loc[indice,'pressure_fname']
    # Sound = Datos.loc[indice,'sound_fname']

    # fs,audio = wavfile.read(Sound)
    # ps,pressure = wavfile.read(Pressure)
    fs = 44150
    audio, pressure = datos_normalizados(indice)

    time = np.linspace(0, len(audio)/fs, len(audio))

    sound_max = max(audio)
    pressure_max = max(pressure)

    Indices_sound_FWH = np.where(audio >= sound_max/2)
    Indices_pressure_FWH = np.where(pressure >= pressure_max/2)

    FWH_sound.append(time[Indices_sound_FWH[0][-1]] - time[Indices_pressure_FWH[0][0]])
    FWH_pressure.append(time[Indices_pressure_FWH[0][-1]] - time[Indices_pressure_FWH[0][0]])

# Los paso a array porque es mas facil trabajar con los mismos   
FWH_sound = np.array(FWH_sound)
FWH_pressure = np.array(FWH_pressure)

#Calculo las diferencias entre ambos

Dif_FWH = FWH_pressure - FWH_sound

#%% Grafico de los ancho altura mitad

plt.figure(figsize=(14,7))
plt.subplot(2,2,(1,2))
plt.title('Anchos altura mitad')
plt.plot(Valores_comunes_zig_zag,FWH_sound,'.',label = 'Sound')
plt.plot(Valores_comunes_zig_zag,FWH_pressure,'.',label='Pressure')
plt.hlines(y=20,xmin=0,xmax=Valores_comunes_zig_zag[-1],color = 'k',linestyle='dashed',label='Corte')
plt.legend()
plt.grid(linestyle='dashed')
plt.xlabel('Indice')
plt.ylabel('Time [sec]')

plt.subplot(2,2,(3,4))
plt.plot(Valores_comunes_zig_zag,Dif_FWH,'.g',label = 'Diferencia de anchos')
plt.grid(linestyle='dashed')
plt.xlabel('Indice')
plt.ylabel('Time [sec]')
plt.legend()

#%% Busco los indices que me gustan

Indices_interes = Valores_comunes_zig_zag[np.where(np.array(FWH_sound)<=20)]
Indices_interes_diff = Valores_comunes_zig_zag[np.where(abs(Dif_FWH)<=10)]

Indices_reducidos = np.intersect1d(Indices_interes, Indices_interes_diff)

#Busco los donde estan en el array de indices para poder graficar
Indices_grafico = np.where(np.isin(Valores_comunes_zig_zag,Indices_reducidos))


fig, ax1 = plt.subplots(figsize=(14,7))
ax1.set_title('Anchos interesantes')
ax1.plot(Indices_grafico[0],FWH_sound[Indices_grafico],'.',label='Sound') 
ax1.plot(Indices_grafico[0],FWH_pressure[Indices_grafico],'.',label='Pressure')
ax1.grid(linestyle='dashed')
ax1.legend()
ax1.set_xlabel('Indice')
ax1.set_ylabel('Time [sec]')

ax2 = ax1.twinx()
ax2.plot(Indices_grafico[0],Dif_FWH[Indices_grafico],'--g',alpha = 0.5)
ax2.set_ylabel('Diferencia de ancho')

    

#%%

plt.close('all')
for indice in Indices_reducidos[20:]:
    plot_sound_vs_pressure(indice)

#%%
from tqdm import tqdm


file_counts = []

for file in os.listdir(directory):
    if file[0] =="s":
        file_counts.append(file)
    
hola = []
with tqdm(total = len(file_counts)) as pbar_h:
    for file in file_counts:
        
        fs, audio = wavfile.read(file)
        
        freq_audio, fft_result_audio = signal.periodogram(audio, fs)
        if  np.log(fft_result_audio[4990:5010].any())>5:
            hola.append(file)
            
        pbar_h.update(1)
        

