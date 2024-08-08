# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:29:41 2024

@author: beneg
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm
import pickle
import seaborn as sns
from plyer import notification
get_ipython().run_line_magic('matplotlib', 'qt5')

# # Specify the directory containing the files
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

# La forma de gabo
#integrador
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

def f_suav(x, t, pars):
    l, value = pars[0], pars[1]
    dxdt = - l * x + value
    return dxdt

#%% Selecciono el archivo
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
os.chdir(directory)

Datos = pd.read_csv('adq-log.txt', delimiter='\t') 

indice = 268

audio, pressure, name, ps = datos_normalizados(indice)

#defino el tiempo
time = np.linspace(0, len(pressure)/ps, len(pressure))

# time = time[int(ps*0):int(ps*20)]
# pressure = pressure[int(ps*0):int(ps*20)]
mean  = np.mean(audio)
std = np.std(audio)

audio_std = np.where(audio >= 2*std)
indice_menor = audio_std[0][0]
indice_mayor = audio_std[0][-1]

print(name)
plt.figure(figsize=(14,7))
plt.subplot(211)
plt.plot(time, audio)
plt.vlines(x=time[indice_menor],ymin=min(audio),ymax=max(audio),color='k',label=r"$2\sigma$")
plt.vlines(x=time[indice_mayor],ymin=min(audio),ymax=max(audio),color='k')
plt.ylabel("Audio (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.subplot(212)
plt.plot(time,pressure)
plt.vlines(x=time[indice_menor],ymin=min(pressure),ymax=max(pressure),color='k',label='Avion')
plt.vlines(x=time[indice_mayor],ymin=min(pressure),ymax=max(pressure),color='k')
plt.ylabel("Pressure (arb. u.)")
plt.legend(fancybox=True, shadow=True)
plt.xlabel("Time (s)")

#%% Calculo el suavizado
p_suav = np.zeros_like(time)
dt = np.mean(np.diff(time))

#esto regula cuanto suaviza
l = 1

for ix, tt in enumerate(time[:-1]):
    p_suav[ix+1] = rk4(f_suav, [p_suav[ix]], tt, dt, [l, pressure[ix]])  #<- aca vamos poniendo que copie a la presion


plt.figure(figsize=(14,7))
plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
plt.subplot(211)
plt.plot(time, pressure)
plt.ylabel("pressure (arb. u.)")
plt.subplot(212)
plt.plot(time, p_suav)
plt.ylabel("Smoothed pressure (arb. u.)")
plt.xlabel("Time (s)")

#%% Maximos de ambos

peaks_maximos,properties_maximos = find_peaks(pressure,prominence=0.1,height=0)
peaks_suav, properties_suav = find_peaks(p_suav,prominence=np.mean(p_suav))
#%% graficos de maximos

plt.close(fig="Maximos originales")
plt.figure("Maximos originales",figsize=(14,7))
plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
plt.subplot(211)
plt.plot(time,pressure)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
# plt.xlim(41.6,41.8)
plt.ylabel("pressure (arb. u.)")
plt.grid(linestyle='dashed')
plt.legend(fancybox=True, shadow=True)



plt.subplot(212)
plt.plot(time, p_suav)
plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
# plt.xlim(41.6,41.8)
plt.grid(linestyle='dashed')
plt.ylabel("Smoothed pressure (arb. u.)")
plt.xlabel('Time [sec]')
plt.legend(fancybox=True, shadow=True)

#%% Calculo maximos originales y creo el data frame de los suavizados
peaks_maximos,properties_maximos = find_peaks(pressure,prominence=0.1,height=0)

suavizados = pd.DataFrame()

#%% Claculo los suvaizados y hago zoom en zonas de interes
dt = np.mean(np.diff(time))
with tqdm(total=10) as pbar_h:
    
    for i in range(21,31):
       
        p_suav = np.zeros_like(time)
        
        #esto regula cuanto suaviza
        l = i

        for ix, tt in enumerate(time[:-1]):
            p_suav[ix+1] = rk4(f_suav, [p_suav[ix]], tt, dt, [l, pressure[ix]])  #<- aca vamos poniendo que copie a la presion

        #Busco maximos del suavizado
        peaks_suav, properties_suav = find_peaks(p_suav,prominence=np.mean(p_suav))
        
        suavizados[f'{l}'] = p_suav
        
        pbar_h.update(1)
        plt.close(f'lambda = {l}')
        plt.figure(f'lambda = {l}',figsize=(14,7))
        plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
        
        plt.subplot(221)
        plt.plot(time,pressure)
        plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
        plt.xlim(0,10)
        plt.ylabel("pressure (arb. u.)")
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(222)
        plt.plot(time,pressure)
        plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)
        plt.xlim(10,20)
        # plt.ylabel("pressure (arb. u.)")
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(223)
        plt.plot(time, p_suav)
        plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
        plt.xlim(0,10)
        plt.ylabel("Smoothed pressure (arb. u.)")
        plt.xlabel('Time [sec]')
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(224)
        plt.plot(time, p_suav)
        plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
        plt.xlim(10,20)
        # plt.ylabel("Smoothed pressure (arb. u.)")
        plt.xlabel('Time [sec]')
        plt.legend(fancybox=True, shadow=True)
        
#%% Guardo los suavizados

with open('Suavizados', 'wb') as file:
    pickle.dump(suavizados, file)
#%%  Cargo el archivo con todos los suavizados

directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Graficos'
os.chdir(directory)
# Open the pickle file for reading
with open('Suavizados', 'rb') as f:
    # Load the DataFrame from the pickle file
    suavizados = pickle.load(f)
    
#%% Calculo de maximos para suavizados

l = 7
peaks_maximos,properties_maximos = find_peaks(pressure,prominence=0.1,height=0, distance=1000)
peaks_suav, properties_suav = find_peaks(suavizados[f"{l}"],prominence=np.mean(suavizados[f"{l}"]))
#%%
plt.figure(figsize=(14,7))
# plt.subplot(211)
plt.plot(time,pressure)
plt.plot(time[peaks_maximos],pressure[peaks_maximos],'.C1',label='Maximos', ms=10)

for i in range(len(peaks_maximos)):
    plt.text( time[peaks_maximos[i]],pressure[peaks_maximos[i]], str(i) )

# plt.xlim(41.6,41.8)
plt.ylabel("pressure (arb. u.)")
plt.grid(linestyle='dashed')
plt.legend(fancybox=True, shadow=True)



# plt.subplot(212)
# plt.plot(time, suavizados[f"{l}"])
# plt.plot(time[peaks_suav],suavizados[f"{l}"][peaks_suav],'C1',label='Maximos', ms=10)
# # plt.xlim(41.6,41.8)
# plt.grid(linestyle='dashed')
# plt.ylabel("Smoothed pressure (arb. u.)")
# plt.xlabel('Time [sec]')
# plt.legend(fancybox=True, shadow=True)

#%%
sacar = [10, 13, 14, 16, 18, 19, 21, 23, 24, 25, 27, 29, 32, 33, 35, 
         39, 44, 45, 46, 49, 50, 51, 52, 53, 55]

picos_limpios = []

for i in range(len(peaks_maximos)):
    if i in sacar:
        None
    else:
        picos_limpios.append(peaks_maximos[i])

#%%
plt.figure(figsize=(14,7))
# plt.subplot(211)
plt.plot(time,pressure)
plt.plot(time[picos_limpios],pressure[picos_limpios],'.C1',label='Maximos', ms=10)

for i in range(len(picos_limpios)):
    plt.text( time[picos_limpios[i]],pressure[picos_limpios[i]], str(i) )

# plt.xlim(41.6,41.8)
plt.ylabel("pressure (arb. u.)")
plt.grid(linestyle='dashed')
plt.legend(fancybox=True, shadow=True)

#%%
periodo_original = np.diff(time[picos_limpios])
periodo_suav = np.diff(time[peaks_suav])

plt.plot(periodo_original)
plt.plot(periodo_suav)




#%% Graficos de las diferencias de maximos y frecuencias

diff_maximos_suav = np.diff(time[peaks_suav])
plt.close('Differencia entre maximos')
plt.figure('Differencia entre maximos',figsize=(14,7))
plt.subplot(211)
plt.title('Frecuencia en funcion del tiempo')
plt.plot(time[peaks_suav][1:],1/diff_maximos_suav)
plt.xlabel('Time [sec]')
plt.ylabel('Frec 1/[sec]')
plt.subplot(223)
plt.title('Tiempo entre maximos')
plt.hist(diff_maximos_suav,edgecolor='k')
plt.ylabel('Count')
plt.xlabel('Time [sec]')
plt.subplot(224)
plt.title('Inversa del tiempo entre maximos')
plt.hist(1/diff_maximos_suav,edgecolor='k')
plt.xlabel('Frec 1/[sec]')



#%% Histogramas de las frecuencias y las diferencias entre maximos
for l in range(1,21):
    
    p_suav = suavizados[f'{l}']
    media = np.mean(p_suav)
    
    peaks_suav, properties_suav = find_peaks(p_suav,prominence=media)
    
    diff_maximos_suav = np.diff(time[peaks_suav])
    
    
    plt.close(f'Diferencia entre maximos lambda = {l}')
    plt.figure(f'Diferencia entre maximos lambda = {l}',figsize=(14,7))
    plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
    plt.subplot(211)
    plt.title('Frecuencia en funcion del tiempo')
    plt.plot(time[peaks_suav][1:],1/diff_maximos_suav)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frec 1/[sec]')
    plt.subplot(223)
    plt.title('Tiempo entre maximos')
    plt.hist(diff_maximos_suav,edgecolor='k')
    plt.ylabel('Count')
    plt.xlabel('Time [sec]')
    plt.subplot(224)
    plt.title('Inversa del tiempo entre maximos')
    plt.hist(1/diff_maximos_suav,edgecolor='k')
    plt.xlabel('Frec 1/[sec]')
    
    plt.savefig(f'Diferencia entre maximos lambda = {l}.png')

#%% Transformadas de los suavizados

for l in range(1,21):
    
    p_suav = suavizados[f'{l}']
    
    freq_p_suav, fft_result_p_suav = signal.periodogram(p_suav, ps)
    
    plt.close(f'Transformada lambda = {l}')
    plt.figure(f'Transformada lambda = {l}',figsize=(14,7))
    plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
    plt.plot(freq_p_suav, fft_result_p_suav)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power')
    plt.xlim(0,5)
    
    # plt.savefig(f'Transformada lambda = {l}.png')



#%% Graficos de presion vs suavizado vs maximos de suavizados vs frec


for l in range(1,21):
    p_suav = suavizados[f'{l}']
    media = np.mean(p_suav)
    
    peaks_suav, properties_suav = find_peaks(p_suav,prominence=media)
    diff_maximos_suav = np.diff(time[peaks_suav])
    
    plt.close(f'maximos lambda = {l}')
    plt.figure(f'maximos lambda = {l}',figsize=(20,10))
    plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
    
    plt.subplot(411)
    plt.plot(time,pressure)
    plt.vlines(x=time[indice_menor],ymin=min(pressure),ymax=max(pressure),color='k',label='Avion')
    plt.vlines(x=time[indice_mayor],ymin=min(pressure),ymax=max(pressure),color='k')
    plt.ylabel("Pressure (arb. u.)")
    plt.legend(fancybox=True, shadow=True)
    
    plt.subplot(412)
    plt.plot(time,p_suav)
    plt.vlines(x=time[indice_menor],ymin=min(p_suav),ymax=max(p_suav),color='k',label='Avion')
    plt.vlines(x=time[indice_mayor],ymin=min(p_suav),ymax=max(p_suav),color='k')
    plt.ylabel("Smoothed pressure (arb. u.)")
    plt.legend(fancybox=True, shadow=True)
    
    plt.subplot(413)
    plt.plot(time[peaks_suav],p_suav[peaks_suav],color='C0',marker='.',linestyle='-')
    plt.vlines(x=time[indice_menor],ymin=min(p_suav),ymax=max(p_suav),color='k',label='Avion')
    plt.vlines(x=time[indice_mayor],ymin=min(p_suav),ymax=max(p_suav),color='k')
    plt.legend(fancybox=True, shadow=True)
    plt.ylabel("Maximums (arb. u.)")
    
    plt.subplot(414)
    plt.plot(time[peaks_suav][0],0,'C0')
    plt.plot(time[peaks_suav][1:],1/diff_maximos_suav,'C0')
    plt.vlines(x=time[indice_menor],ymin=min(1/diff_maximos_suav),ymax=max(1/diff_maximos_suav),color='k',label='Avion')
    plt.vlines(x=time[indice_mayor],ymin=min(1/diff_maximos_suav),ymax=max(1/diff_maximos_suav),color='k')
    plt.legend(fancybox=True, shadow=True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frec 1/[sec]')
    
    plt.savefig(f'Original vs Smoothed vs Maximums vs frec lambda = {l}.png')
    
# Display a notification when the program finishes
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
#%% Calculo de las aumentos de frecuencias

tiempos = []
frecuencias = []
medias = []

for l in range(1,21):
    
    p_suav = suavizados[f'{l}']
    media = np.mean(p_suav)
    
    peaks_suav, properties_suav = find_peaks(p_suav,prominence=media)
    diff_maximos_suav = np.diff(time[peaks_suav])
    
    frec = 1/diff_maximos_suav
    
    frec = np.concatenate(([frec[0]], frec))

    Valores_frec = np.where(frec >= np.mean(frec))
    tiempos_de_frec = time[peaks_suav[Valores_frec[0]]]
    Valores_de_frec = frec[Valores_frec[0]]
    
    tiempos.append(tiempos_de_frec)
    frecuencias.append(Valores_de_frec)
    medias.append(np.mean(frec))
    
    
    
    plt.figure(figsize=(14,7))
    plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
    
    
    
    plt.plot(time[peaks_suav],frec,'C0')
    plt.plot(tiempos_de_frec,Valores_de_frec,'.C1',label = 'Frecuencias de interes')
    plt.vlines(x=time[indice_menor],ymin=min(frec),ymax=max(frec),color='k',label='Avion')
    plt.vlines(x=time[indice_mayor],ymin=min(frec),ymax=max(frec),color='k')
    plt.hlines(y=np.mean(frec), xmin=0, xmax=60,color='k', label = 'Media')
    plt.legend(fancybox=True, shadow=True)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frec 1/[sec]')
    
#%% Histograma de la distribucion de frecuencias y sus tiempos
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.title('Tiempo de aumento de frecuencia')
plt.hist(tiempos)
plt.xlabel('Time [sec]')
plt.subplot(122)
plt.title('Frecuencias')
plt.hist(frecuencias)
plt.xlabel('Frec 1/[sec]')



#%%
with tqdm(total=suavizados.shape[1]) as pbar_h:
    for l in range(1,21):
    
        p_suav = suavizados[f'{l}']
        
        media = np.mean(p_suav)
        
        peaks_suav, properties_suav = find_peaks(p_suav,prominence=media)
        
        
        plt.close(f'lambda = {l}')
        plt.figure(f'lambda = {l}',figsize=(14,7))
        plt.suptitle(r'$\lambda\,$' f'= {l}',fontsize=14)
        
        plt.subplot(221)
        plt.plot(time,pressure)
        plt.plot(time[peaks_suav],pressure[peaks_suav],'.C1',label='Maximos suavizados', ms=10)
        plt.xlim(0,10)
        plt.ylabel("Pressure (arb. u.)")
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(222)
        plt.plot(time,pressure)
        plt.plot(time[peaks_suav],pressure[peaks_suav],'.C1',label='Maximos suavizados', ms=10)
        plt.xlim(10,20)
        # plt.ylabel("pressure (arb. u.)")
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(223)
        plt.plot(time, p_suav)
        plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
        plt.xlim(0,10)
        plt.ylabel("Smoothed pressure (arb. u.)")
        plt.xlabel('Time [sec]')
        plt.legend(fancybox=True, shadow=True)
        
        plt.subplot(224)
        plt.plot(time, p_suav)
        plt.plot(time[peaks_suav],p_suav[peaks_suav],'.C1',label='Maximos', ms=10)
        plt.xlim(10,20)
        # plt.ylabel("Smoothed pressure (arb. u.)")
        plt.xlabel('Time [sec]')
        plt.legend(fancybox=True, shadow=True)
        
        
        plt.savefig(f'Maximos relajado vs avion lambda = {l}.png')
        pbar_h.update(1)
        

# Display a notification when the program finishes
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)


#%%

# Function to check if the sorted extrema alternate
def check_alternation_and_count_violations(sorted_extremos, maximos_set, minimos_set):
    violations_count = 0
    violations = []
    for i in range(1, len(sorted_extremos)):
        current = sorted_extremos[i-1]
        next = sorted_extremos[i]

        if (current in maximos_set and next in maximos_set) or (current in minimos_set and next in minimos_set):
            violations_count += 1
            violations.append(current)
            violations.append(next)
    return violations_count,violations


#%%

no_intercambian = pd.DataFrame()

with tqdm(total=suavizados.shape[1]) as pbar_h:
    for l in range(7,8):


        p_suav = suavizados[f'{l}']
        
        media = np.mean(p_suav)
        maximos,_ = find_peaks(p_suav,prominence=media)
        minimos,_ = find_peaks((-1)*p_suav,prominence=media,height=media)
        
        plt.close(f'Maximos y minimos {l}')
        plt.figure(f'Maximos y minimos {l}',figsize=(14,7))
        plt.title(r'$\lambda$'f'={l}',fontsize=12)
        # plt.subplot(211)
        plt.plot(time,p_suav)
        plt.plot(time[maximos],p_suav[maximos],'.C1',label='Maximos')
        plt.plot(time[minimos],p_suav[minimos],'.C2',label='Minimos')
        plt.hlines(media, time[0], time[-1], 'k',label='Media')
        plt.ylabel("Smoothed pressure (arb. u.)")
        plt.xlabel('Time [sec]')
        plt.legend(fancybox=True,shadow=True)
        
        
        tiempos_extremos = np.concatenate((maximos,minimos))
        
        tiempos_extremos_sorted = np.sort(tiempos_extremos)
        
        
        maximos_set = set(maximos)
        minimos_set = set(minimos)
        
        
        # Check if the extrema alternate and count violations
        violations_count,violations = check_alternation_and_count_violations(tiempos_extremos_sorted, maximos_set, minimos_set)

        no_intercambian[f'{l}'] = [violations_count]

        if violations_count == 0:
            print(f"for l = {l} The extrema alternate between maximos and minimos.")
        else:
            print(f"for l = {l} The extrema do not alternate between maximos and minimos {violations_count} times.")
        plt.plot(time[violations],p_suav[violations],'.C3',label='No cumplen',ms=10)    
        plt.legend(fancybox=True,shadow=True)
        
        # plt.savefig(f'Extremos intercalados lambda = {l}.png')
        pbar_h.update(1)
            
# Display a notification when the program finishes
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
#%%

plt.figure()
plt.plot(time,p_suav)
plt.plot(time[tiempos_extremos_sorted],p_suav[tiempos_extremos_sorted],'.C1')

#%%

l_values = np.arange(1, 21)
plt.figure(figsize=(14,7))
plt.plot(l_values, no_intercambian.loc[0, l_values.astype(str)], marker='o')
plt.vlines(7,0,23,'k',label=r'$\lambda \, = \, 7$' )
plt.xlabel(r'$\lambda$',fontsize=24)
plt.ylabel('Error count',fontsize=12)
plt.legend(fancybox=True,shadow=True)

#%%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\CaFF028-RoNe\2023-01-30-night'
os.chdir(directory)

Datos = pd.read_csv('adq-log.txt', delimiter='\t') 

def funcion_suav(indice,l):


    audio, pressure, name, ps = datos_normalizados(indice)
    
    #defino el tiempo
    time = np.linspace(0, len(pressure)/ps, len(pressure))
    
    p_suav = np.zeros_like(time)
    dt = np.mean(np.diff(time))
    
    
    
    for ix, tt in enumerate(time[:-1]):
        p_suav[ix+1] = rk4(f_suav, [p_suav[ix]], tt, dt, [l, pressure[ix]])
      
    return audio,pressure,p_suav,name


#%%   


indice = 360
l = 7


audio,pressure,p_suav,name = funcion_suav(indice, l)
# Display a notification when the program finishes
notification.notify(
    title='Program Finished',
    message='Your Python program has finished running.',
    app_icon=None,  # e.g. 'C:\\icon_32x32.ico'
    timeout=10,  # seconds
)
#%%
plt.figure(figsize=(14,7))
plt.title(f'{name}')
plt.subplot(311)
plt.plot(time,audio)
plt.ylabel("Audio (arb. u.)")
plt.subplot(312)
plt.plot(time,pressure)
plt.ylabel("Pressure (arb. u.)")
plt.subplot(313)
plt.plot(time,p_suav)
plt.ylabel("Smoothed pressure (arb. u.)")
plt.xlabel('Time [sec]')

#%% Pruebas del intervalo
indice = 268

audio, pressure, name, ps = datos_normalizados(indice)
freq_audio_avion, fft_result_audio_avion = signal.periodogram(audio, ps)

indice = 203

audio, pressure, name, ps = datos_normalizados(indice)
freq_audio_ruido, fft_result_audio_ruido = signal.periodogram(audio, ps)

plt.figure()
plt.plot(freq_audio_avion,np.log(fft_result_audio_avion))
plt.plot(freq_audio_avion, np.log(fft_result_audio_ruido))
# plt.axvline(4000)
# plt.axvline(5000)
# plt.axhline(-21)
#%%
indice = 536

audio, pressure, name, ps = datos_normalizados(indice)

std = np.std(audio)

audio_std = np.where(audio >= 2*std)
indice_menor = audio_std[0][0]
indice_mayor = audio_std[0][-1]

f, t, Sxx = signal.spectrogram(audio, ps)
# Define the frequency cutoff
frequency_cutoff = 1000  # Adjust as needed #1000 funciona piola
threshold = -21  # dB threshold for detection   -21 funciona piola

# Convert the spectrogram to dB scale
Sxx_dB =np.log(Sxx)

# Find the indices where the frequency is above the cutoff
freq_indices = np.where(f > frequency_cutoff)[0]

# Identify times where the spectrogram surpasses the threshold
time_indices = np.any(Sxx_dB[freq_indices, :] > threshold, axis=0)
time_above_threshold = t[time_indices]

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

longest_interval = find_longest_interval(time_above_threshold)

# Plot the spectrogram with the detected interval
plt.close('Espectograma')
plt.figure('Espectograma',figsize=(10, 6))
plt.suptitle(f'{indice} = {name}')
plt.subplot(211)
plt.plot(time,audio)
plt.ylabel("Audio (arb. u.)")
plt.axvline(x=longest_interval[0], color='k', linestyle='--',label='Frecuencias')
plt.axvline(x=longest_interval[-1], color='k', linestyle='--')
plt.axvline(time[indice_menor],color='r', linestyle='--',label=r'$2\sigma$')
plt.axvline(time[indice_mayor],color='r', linestyle='--')
plt.legend(fancybox=True,shadow=True)
# norm = mcolors.LogNorm(vmin=Sxx.min() + 1e-10, vmax=Sxx.max())
plt.subplot(212)
plt.pcolormesh(t, f, np.log(Sxx), shading='gouraud')
plt.colorbar(label='Intensity [dB]')
plt.axvline(x=longest_interval[0], color='red', linestyle='--', label='Start of Interval')
plt.axvline(x=longest_interval[-1], color='blue', linestyle='--', label='End of Interval')
plt.axhline(y=frequency_cutoff,color='k', linestyle='--', label='cutoff')
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.tight_layout()
# plt.ylim(0, sample_rate / 2)  # Limit the y-axis to the Nyquist frequency
plt.legend()
plt.show()

print(f"Longest continuous interval where the signal surpasses the threshold: {longest_interval[0]} to {longest_interval[-1]} seconds")








