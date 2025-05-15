#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:40:06 2019

@author: sebageli
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
# from numpy.fft import rfft
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from scipy.io import wavfile
# from scipy import signal

#%%###################### Importamos nombres de archivos #########################
# Carpeta = 'D:/Experimentos 2022/Emg abdominal y presion/CaFF028-RoNe/Experimento/CaFF028-RoNe/2022-12-15-night/'
Carpeta = "D:/Experimentos 2022/Emg abdominal y presion/CaFF028-RoNe/Experimento/CaFF028-RoNe/2023-01-28-night/"
Archivos=[]

import glob, os
os.chdir(Carpeta)
for file in glob.glob("*.wav"):
    if file[0]=='s' and 'BOS' not in file and 'REV' not in file and 'CON' not in file:
        Archivos.append(file)

# for file in glob.glob("*.WAV"):
#     Archivos.append(file)

Archivos = Archivos[::2]
print(Archivos)
print("El numero de archivos es: ", len(Archivos))
#import os
#for files in os.walk(Carpeta, topdown=False):
#    for name in files:
#        Archivos = name[1:len(name)]

#%%###############################################################################

#Promedio de las transformadas en intervalo de interes
Promedios = []

#Promedios en el intervalo de alta frecuencia
PromediosAltos = []

#Guardo el cociente entre las listas
Cocientes= []

#Guardo los que son probablemente ruido
Ruido = []

for i in range(len(Archivos)):
    #Levantamos el archivo de audio
    fs,audio = wavfile.read(Carpeta + '/'+Archivos[i])
    
    #Definimos parametros para calcular el fft
    N = len(audio) #Numero de puntos
    dt = 1/fs    #Paso temporal
    xf = np.linspace(0, 1.0/(2.0*dt), N//2) #Vector de frecuencias
    
    #Damos el intervalo de frecuencias en donde buscaremos canto
    f1 = int(10.0*N/fs)  #Posicion del valor 1.2kHz en el vector xf
    f2 = int(700.0*N/fs)  #Posicion  del valor 6kHz en el vector xf
    
    #Damos el intervalo de frecuencias donde buscaremos ruido base
    f3 = int(15200.0*N/fs)
    f4 = int(20000.0*N/fs)
    
    
    print (str(np.round(100*i/(len(Archivos)),2)) + "%") #Decimos que porcentaje de archivos llevamos analizados
    
    #Hacemos fft al archivo de audio
    audio_ff = fftpack.fft(audio)
    
    #Guardamos las cantidades que usaremos para determinar si es canto o no
    Promedios.append(np.mean(abs(audio_ff[f1:f2])))    
    PromediosAltos.append(np.mean(abs(audio_ff[f3:f4])))
    Cocientes.append(Promedios[i]/PromediosAltos[i])
    
    #Hacemos un espectrograma para ver el archivo
#    plt.figure(i,[17,3])
#    plt.title(Archivos[i]+"; Cociente="+str(Promedios[i]/PromediosAltos[i]))
#    plt.specgram(audio,Fs=fs,NFFT=1024,cmap='inferno');
#    plt.show()
    
#%% Para determinar qué valor de cociente es el umbral de canto o no canto vemos
# un histograma
plt.hist(Cocientes, 100)
plt.axvline(1.59, c='r', zorder=10)
    


#%%    
cantos = []
print ("Ruidos:")
for i in range(len(Cocientes)):
        if Cocientes[i] < 15: #Si el cociente es menor a 2.4 lo consideramos ruido
            print (Archivos[i], Cocientes[i], i)
            Ruido.append(Cocientes[i])
        else: 
            cantos.append(Archivos[i]) #sino lo consideramos canto

print("Los posibles cantos son:")
print(cantos)
print("LA cantidad son: ", len(cantos))
            
Archivos = np.array(Archivos)
Promedios = np.array(Promedios)
Cocientes = np.array(Cocientes)

#%%
save_folder = 'C:/Users/facuf/Desktop/Facu 2024/Tesis de Licenciatura - Aviones y sueño/Results/Encontrar aviones - 2025/'
# np.save(save_folder+"cocientes.npy", Cocientes)
# np.save(save_folder+"Archivos.npy", Archivos)
# np.save(save_folder+"posibles_aviones.npy", cantos)




#%% Copiamos los cantos a otra carpeta

# destino = Carpeta+'/Posibles aviones'

destino = 'C:/Users/facuf/Desktop/Facu 2024/Tesis de Licenciatura - Aviones y sueño/Datos/CaFF079-AzuVe/2023-02-13-night/'


from shutil import copyfile

for src in cantos:
    copyfile(Carpeta+'/'+src, destino+'/'+src)
    copyfile(Carpeta+'/pressure'+src[5:], destino+'/pressure'+src[5:])
    # copyfile(Carpeta+'/hall'+src[5:], destino+'/hall'+src[5:])
    copyfile(Carpeta+'/'+src[6:-3]+"txt", destino+'/'+src[6:-3]+"txt")


    
#%%

for src in cantos:
    copyfile(Carpeta+'/pressure'+src[5:], destino+'/pressure'+src[5:])
    copyfile(Carpeta+'/hall'+src[5:], destino+'/vs'+src[5:])
    # copyfile(Carpeta+'/'+src[6:-3]+"txt", destino+'/'+src[6:-3]+"txt")

    
#%%
    
    
    