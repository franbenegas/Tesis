
#%%
import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from scipy.io import wavfile
from tqdm import tqdm
from plyer import notification
from shutil import copyfile
import glob, os
# from scipy import signal
get_ipython().run_line_magic('matplotlib', 'qt5')

#%%

def datos_normalizados_2(indice):
    
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
    
    pressure_norm = norm11_interval(pressure_norm, 0, 5, fs)
    audio_norm = norm11_interval(audio_norm, 0, 5, fs)

    return audio_norm, pressure_norm, name, fs

#%%###################### Importamos nombres de archivos #########################

Carpeta = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\2023-02-11-night'
Archivos=[]

import glob, os
os.chdir(Carpeta)
for file in glob.glob("*.wav"):
    if file[0]=='s':
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

with tqdm(total=len(Archivos)) as pbar_h:

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
        
        
        # print (str(np.round(100*i/(len(Archivos)),2)) + "%") #Decimos que porcentaje de archivos llevamos analizados
        
        #Hacemos fft al archivo de audio
        audio_ff = fftpack.fft(audio)
        
        #Guardamos las cantidades que usaremos para determinar si es canto o no
        Promedios.append(np.mean(abs(audio_ff[f1:f2])))    
        PromediosAltos.append(np.mean(abs(audio_ff[f3:f4])))
        Cocientes.append(Promedios[i]/PromediosAltos[i])
        
        pbar_h.update(1)
    
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)

#%% Para determinar qué valor de cociente es el umbral de canto o no canto vemos
# un histograma
plt.hist(Cocientes, 100)
# plt.axvline(1.59, c='r', zorder=10)
    
#%%    
cantos = []
# print ("Ruidos:")
for i in range(len(Cocientes)):
        if Cocientes[i] < 10: #Si el cociente es menor a 2.4 lo consideramos ruido
            # print (Archivos[i], Cocientes[i], i)
            Ruido.append(Cocientes[i])
        else: 
            cantos.append(Archivos[i]) #sino lo consideramos canto
                
            
print("Los posibles cantos son:")
print(cantos)
            
Archivos = np.array(Archivos)
Promedios = np.array(Promedios)
Cocientes = np.array(Cocientes)


#%% Copiamos los cantos a otra carpeta

# Define source and destination directories
source_folder = Carpeta
destination_folder = os.path.join(source_folder, 'Posibles aviones')

# Ensure the destination directory exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# List of source files
# cantos = ['file1', 'file2', 'file3']  # Replace with your actual file names

for src in cantos:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path)

    # Copy corresponding 'pressure' file
    pressure_src = os.path.join(source_folder, 'pressure' + src[5:])
    pressure_dest = os.path.join(destination_folder, 'pressure' + src[5:])
    copyfile(pressure_src, pressure_dest)

    # Uncomment the following lines if you want to copy 'hall' files as well
    # hall_src = os.path.join(source_folder, 'hall' + src[5:])
    # hall_dest = os.path.join(destination_folder, 'hall' + src[5:])
    # copyfile(hall_src, hall_dest)

    # Copy corresponding .txt file
    txt_src = os.path.join(source_folder, src[6:-3] + "txt")
    txt_dest = os.path.join(destination_folder, src[6:-3] + "txt")
    copyfile(txt_src, txt_dest)

#%%                   A partir de aca tengo que trabajar si ya tengo los datos filtrados    ######################################
#%%

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\2023-02-13-night'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = []
Sonidos = []

for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
        Presiones.append(file)


#%%

current_index = 0
aceptados_sonido = []
aceptados_presion = []
# Function to update the plot
def update_plot(index):
    plt.clf()  # Clear the current figure
    # Load data from the file (replace with your actual data loading code)
    # data = load_data_from_file(Sonidos[index])  
    audio, pressure, name, fs = datos_normalizados_2(index)

    #defino el tiempo
    time = np.linspace(0, len(pressure)/fs, len(pressure))
    plt.subplot(211)
    plt.plot(time,audio)
    plt.suptitle(f"{index}: {Sonidos[index]}")
    plt.subplot(212)
    plt.plot(time,pressure)
    plt.draw()

# Function to be called when the space bar is pressed
def on_key(event):
    global current_index
    if event.key == 'a':
        aceptados_sonido.append(Sonidos[current_index])
        aceptados_presion.append(Presiones[current_index])
        current_index = (current_index + 1) % len(Sonidos)
        update_plot(current_index)
        
    if event.key == ' ':
        current_index = (current_index + 1) % len(Sonidos)
        update_plot(current_index)
    

# Initialize the plot
fig, ax = plt.subplots(figsize=(14,7))
update_plot(current_index)

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)
#%%

source_folder = directory
destination_folder = os.path.join(source_folder, 'Aviones')

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
for src in aceptados_presion:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path)
    
for src in aceptados_sonido:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path) 
  
#%%

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF909-NaRo\2023-02-13-night\Aviones'
os.chdir(directory)

lugar = os.listdir(directory)
Presiones = []
Sonidos = []

for file in lugar:
    if file[0]=='s':
        Sonidos.append(file)
    elif file[0]=='p':
        Presiones.append(file)
#%% Aca selecciono los que pasa el avion y veo cosas en la presion.
current_index = 0
aceptados_sonido = []
aceptados_presion = []
# Function to update the plot
def update_plot(index):
    plt.clf()  # Clear the current figure
    # Load data from the file (replace with your actual data loading code)
    # data = load_data_from_file(Sonidos[index])  
    audio, pressure, name, fs = datos_normalizados_2(index)

    #defino el tiempo
    time = np.linspace(0, len(pressure)/fs, len(pressure))
    plt.subplot(211)
    plt.plot(time,audio)
    plt.suptitle(f"{index}: {Sonidos[index]}")
    plt.subplot(212)
    plt.plot(time,pressure)
    plt.draw()

# Function to be called when the space bar is pressed
def on_key(event):
    global current_index
    if event.key == 'a':
        aceptados_sonido.append(Sonidos[current_index])
        aceptados_presion.append(Presiones[current_index])
        current_index = (current_index + 1) % len(Sonidos)
        update_plot(current_index)
        
    if event.key == ' ':
        current_index = (current_index + 1) % len(Sonidos)
        update_plot(current_index)
    

# Initialize the plot
fig, ax = plt.subplots(figsize=(14,7))
update_plot(current_index)

# Connect the key press event
fig.canvas.mpl_connect('key_press_event', on_key)
#%%

source_folder = directory # La direccion de la carpeta Aviones
destination_folder = os.path.join(source_folder, 'Aviones y pajaros')

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    
for src in aceptados_presion:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path)
    
for src in aceptados_sonido:
    # Define the full source and destination paths for the files to copy
    src_path = os.path.join(source_folder, src)
    dest_path = os.path.join(destination_folder, src)
    copyfile(src_path, dest_path) 
  

#%%

counts = []

for i in range(len(Sonidos)):
    if Sonidos[i][-4:]==Presiones[i][-4:]:
        counts.append(i)
