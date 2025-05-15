# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:49:23 2024

@author: beneg
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import norm
from scipy.signal import find_peaks
from numba import njit
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['mathtext.fontset'] = 'stix'

@njit
def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def sigm(x):
    return 1 / (1 + np.exp(-10* x))

@njit
def f(v, t, pars):
    cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp = pars
    r, theta, mu, w = v
    
    # Equations
    drdt = mu * r - cr * r**3
    dthetadt = w - ci * r**2
    dmudt = (-mu + A_s0 * As)/ tau
    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) * sigm(Asp - 0.5)) * (A_s0* As + w0 - w)
    
    return np.array([drdt, dthetadt, dmudt, dwdt])

@njit
def integrate_system(time, dt, pars, As_values, Asp_values):
    # Initial conditions
    r = np.zeros_like(time)
    theta = np.zeros_like(time)
    mu = np.zeros_like(time)
    w = np.zeros_like(time)
    r[0], theta[0], mu[0], w[0] = 0.25, 0, 0, 2*np.pi

    for ix in range(len(time) - 1):
        pars[3] = As_values[ix]  # Modify As within the loop
        pars[-1] = Asp_values[ix]  # Modify Asp within the loop
        r[ix + 1], theta[ix + 1], mu[ix + 1], w[ix + 1] = rk4(f, np.array([r[ix], theta[ix], mu[ix], w[ix]]), time[ix], dt, pars)
    
    return r, theta, mu, w

time = np.arange(0, 60, 0.001)
dt = np.mean(np.diff(time))

# Precompute As and Asp
As_values = norm.pdf(time, loc=30, scale=0.1)
As_values /= np.max(As_values)
Asp_values = np.concatenate(([0], np.diff(As_values)))
Asp_values /= np.max(Asp_values)

plt.close('all')
# Create subplot
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Pressure (arb. u.)", fontsize=14)
ax[2].set_ylabel("Rate (Hz)", fontsize=14)
ax[2].set_xlabel("Time (s)", fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)

ax[0].plot(time, As_values, label="sound envelope")
p, = ax[1].plot([], [], 'C0o')
r_plot, = ax[2].plot([], [])

# Slider positions
slider_params = [
    (r'$c_r$', 0.1, 0.25, 0.3, 0.03, 0.0, 20.0, 3),
    (r'$\omega_0$', 0.1, 0.20, 0.3, 0.03, 0.0, 10.0, 2*np.pi),
    (r'$c_i$', 0.1, 0.15, 0.3, 0.03, -5.0, 6.0, 2*np.pi-1),
    (r'$\tau$', 0.1, 0.1, 0.3, 0.03, 0.0, 5.0, 1),
    (r'$A_{s0}$', 0.65, 0.25, 0.3, 0.03, 0.0, 5.0, 1),
    (r'$\tau_{up}$', 0.65, 0.20, 0.3, 0.03, 0.0, 5.0, 1),
    (r'$\tau_{down}$', 0.65, 0.15, 0.3, 0.03, 0.0, 20.0, 15),
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
    
    r, theta, mu, w = integrate_system(time, dt, pars, As_values, Asp_values)

    # peaks, _ = find_peaks(r * np.cos(theta), height=0, distance=int(0.1 / dt))
    
    # Update plots
    p.set_data(time, r)
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

#%%
time = np.arange(0, 60, 0.001)
dt = np.mean(np.diff(time))

# Precompute As and Asp
As_values = norm.pdf(time, loc=30, scale=3)
As_values /= np.max(As_values)
Asp_values = np.concatenate(([0], np.diff(As_values)))
Asp_values /= np.max(Asp_values)

def sigm(x):
    return 1 / (1 + np.exp(-10 * x)) - 0.5

fig, ax = plt.subplots(4, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)", fontsize=14)
ax[1].set_ylabel("Derivada (arb. u.)", fontsize=14)
ax[2].set_ylabel("Derivada seg (arb. u.)", fontsize=14)
ax[3].set_xlabel("Time (s)", fontsize=14)

ax[0].plot(time, As_values, label="sound envelope")
derivada = np.diff(As_values,prepend=0)
derivada_seg = np.diff(As_values,n=2,prepend=[0,0])
ax[1].plot(time, derivada)
ax[1].plot(time,sigm(derivada))
ax[2].plot(time, derivada_seg)
ax[2].plot(time, sigm(derivada_seg))
# ax[3].plot(time, (sigm(derivada)>0)*(sigm(derivada_seg)>0)*As_values)
ax[3].plot(time, (sigm(derivada)>0)*(sigm(derivada_seg)>0))

#%%

data = np.loadtxt(r"C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-30-night\Aviones\Aviones y pajaros/average RoNe rate", delimiter=',')
time_data = data[0]
rate = data[1]
error = data[2]

data_sound = np.loadtxt(r"C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran\CaFF028-RoNe\2023-01-30-night\Aviones\Aviones y pajaros/average RoNe sonido", delimiter=',')
time_sound = data_sound[0]
sound = data_sound[1]
error_s = data_sound[2]


#%%

def f(t, a, b, t0, c):
    return a * np.exp( b * (t-t0)) + c

fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
ax.errorbar(time_data, rate, yerr=error)
ax.plot(time_data, f(time_data, 1, .75, 3, 1.25))
ax.plot(time_data, f(time_data, 1, -.05, 3, 1.5))
ax.set_xlabel("time (s)")
ax.set_ylabel("rate (Hz)")
ax.set_ylim([1,3])
fig.tight_layout()

def sigm(x):
    return 1 / (1 + np.exp(-10 * x)) - 0.5



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
    sound, lowcut, highcut, fs, order=order)

time = time_sound
dt = np.mean(np.diff(time))
Asp = np.concatenate(([0], np.diff(filtered_signal)))
# Asp = np.diff(filtered_signal,prepend=0)
Asp_2 = np.concatenate(([0],np.diff(Asp)))
# Asp_2 = np.diff(Asp, prepend=0)
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].errorbar(time_sound, sound, yerr=error_s)
ax[1].plot(time_sound,Asp)
ax[1].plot(time_sound,sigm(Asp))
ax[2].plot(time_sound,Asp_2)
ax[2].plot(time_sound,sigm(Asp_2))


#%%

# Assuming time_sound, Asp, and Asp_2 are already defined arrays
# Apply sigmoid function to Asp and Asp_2
sigm_Asp = sigm(Asp)
sigm_Asp_2 = sigm(Asp_2)

# Find where both sigm(Asp) > 0 and sigm(Asp_2) > 0
condition = (sigm_Asp > 0) * (sigm_Asp_2 > 0)

# Find the longest interval where the condition is satisfied
max_length = 0
current_length = 0
start_idx = 0
max_start_idx = 0

for i in range(len(condition)):
    if condition[i] != 0:  # Condition holds
        if current_length == 0:
            start_idx = i  # Start of a new interval
        current_length += 1
    else:
        if current_length > max_length:
            max_length = current_length
            max_start_idx = start_idx
        current_length = 0

# Final check for the last interval
if current_length > max_length:
    max_length = current_length
    max_start_idx = start_idx

# Now we have the start and end indices of the longest interval
end_idx = max_start_idx + max_length

# Create a copy of the original signal or time array
filtered_signal = np.zeros_like(condition)
filtered_signal_2 = np.zeros_like(condition)
# Retain the values in the longest interval, set everything else to 0
filtered_signal[(np.abs(time_sound- 0)).argmin():end_idx] = condition[(np.abs(time_sound- 0)).argmin():end_idx]
filtered_signal_2[max_start_idx:end_idx] = condition[max_start_idx:end_idx]
fig, ax = fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
ax.plot(time_sound,condition,label='Funcion signo')
ax.plot(time_sound, filtered_signal,label = 'Itervalo mas largo >0')
ax.plot(time_sound, filtered_signal_2, label = 'intervalo mas largo')
ax.legend(fancybox=True,shadow=True)

#%%

fig, ax = plt.subplots(4, 1, figsize=(14, 7), sharex=True)
ax[0].errorbar(time_sound, sound, yerr=error_s)
ax[1].plot(time_sound,Asp,label = r'$\dot{A}$')
ax[1].plot(time_sound,sigm(Asp)>0, label = r'sig($\dot{A}$)')
ax[2].plot(time_sound,Asp_2, label = r'$\ddot{A}$')
ax[2].plot(time_sound,sigm(Asp_2)>0,label = r'sig($\ddot{A}$)')
ax[3].errorbar(time_data, rate, yerr=error)
ax[3].plot(time_sound, filtered_signal + 1,label = 'fucion signo > 0')
ax[3].plot(time_sound, filtered_signal_2 + 1, label = 'funcion signo intevalo mas grande')
ax[1].legend(fancybox=True,shadow=True)
ax[2].legend(fancybox=True,shadow=True)
ax[3].legend(fancybox=True,shadow=True)
#%%
fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
ax.errorbar(time_sound, sound, yerr=error_s)
ax.errorbar(time_data, rate, yerr=error)
# # Define the function to compute the derivative
# @njit
# def f_prueba(w, t, pars):
#     tau = pars
#     dwdt = (1/tau) * w
#     return dwdt

# # Integrate the system using RK4
# @njit
# def integrate_system_2(time, dt, pars):
#     # Initial conditions
#     w = np.zeros_like(time)
#     w[0] = 1.25  # Initial condition

#     # Define the RK4 method inside since numba njit doesn't work with external Python functions
#     def rk4(dxdt, w, t, dt, pars):
#         k1 = dxdt(w, t, pars) * dt
#         k2 = dxdt(w + 0.5 * k1, t, pars) * dt
#         k3 = dxdt(w + 0.5 * k2, t, pars) * dt
#         k4 = dxdt(w + k3, t, pars) * dt
#         return w + (k1 + 2 * k2 + 2 * k3 + k4) / 6

#     # Time integration loop
#     for ix in range(len(time) - 1):
#         w[ix + 1] = rk4(f_prueba, w[ix], time[ix], dt, pars)
    
#     return w


# dt = np.mean(np.diff(time_data))
# pars = -0.75
# # Call the integration
# w = integrate_system_2(time_data, dt, pars)

# fig, ax = fig, ax = plt.subplots(1, 1, figsize=(14, 7), sharex=True)
# ax.errorbar(time_data, rate, yerr=error)
# ax.plot(time_data,w)
# ax.set_ylim([1,3])