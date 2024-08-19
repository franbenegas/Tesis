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

@njit
def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

@njit
def sigm(x):
    return 1 / (1 + np.exp(-10 * x))

@njit
def f(v, t, pars):
    cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp = pars
    r, theta, mu, w = v
    
    # Equations
    drdt = mu * r - cr * r**3
    dthetadt = w - ci * r**2
    dmudt = -mu / tau + A_s0 * As
    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1) * sigm(Asp - 0.5)) * (A_s0 * As + w0 - w)
    
    return np.array([drdt, dthetadt, dmudt, dwdt])

@njit
def integrate_system(time, dt, pars, As_values, Asp_values):
    # Initial conditions
    r = np.zeros_like(time)
    theta = np.zeros_like(time)
    mu = np.zeros_like(time)
    w = np.zeros_like(time)
    r[0], theta[0], mu[0], w[0] = 0.25, 0, 0, pars[1]

    for ix in range(len(time) - 1):
        pars[3] = As_values[ix]  # Modify As within the loop
        pars[-1] = Asp_values[ix]  # Modify Asp within the loop
        r[ix + 1], theta[ix + 1], mu[ix + 1], w[ix + 1] = rk4(f, np.array([r[ix], theta[ix], mu[ix], w[ix]]), time[ix], dt, pars)
    
    return r, theta, mu, w

time = np.arange(0, 60, 0.01)
dt = np.mean(np.diff(time))

# Precompute As and Asp
As_values = norm.pdf(time, loc=30, scale=3)
As_values /= np.max(As_values)
Asp_values = np.concatenate(([0], np.diff(As_values)))
Asp_values /= np.max(Asp_values)

# Create subplot
fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
ax[0].set_ylabel("Audio (arb. u.)")
ax[1].set_ylabel("Pressure (arb. u.)")
ax[2].set_ylabel("Rate (Hz)")
ax[2].set_xlabel("Time (sec)")

plt.tight_layout()
plt.subplots_adjust(bottom=0.35)

ax[0].plot(time, As_values, label="sound envelope")
p, = ax[1].plot([], [], 'ro')
r_plot, = ax[2].plot([], [])

# Slider positions
slider_params = [
    ("cr", 0.1, 0.25, 0.3, 0.03, 0.0, 20.0, 3),
    ("w0", 0.1, 0.20, 0.3, 0.03, 0.0, 10.0, 5),
    ("ci", 0.1, 0.15, 0.3, 0.03, 0.0, 5.0, 1),
    ("tau", 0.1, 0.1, 0.3, 0.03, 0.0, 5.0, 1),
    ("A_s0", 0.65, 0.25, 0.3, 0.03, 0.0, 5.0, 1),
    ("t_up", 0.65, 0.20, 0.3, 0.03, 0.0, 5.0, 1),
    ("t_down", 0.65, 0.15, 0.3, 0.03, 0.0, 5.0, 1),
]

sliders = {}
for name, left, bottom, width, height, min_val, max_val, init_val in slider_params:
    sliders[name] = Slider(plt.axes([left, bottom, width, height]), name, min_val, max_val, valinit=init_val)

# Function to update plots
def update(val):
    # Convert all slider values to float32 to ensure consistency
    pars = np.array([
        sliders['cr'].val, sliders['w0'].val, sliders['ci'].val, 1.0,
        sliders['tau'].val, sliders['A_s0'].val, sliders['t_up'].val, sliders['t_down'].val, 1.0
    ], dtype=np.float64)
    
    r, theta, mu, w = integrate_system(time, dt, pars, As_values, Asp_values)

    peaks, _ = find_peaks(r * np.cos(theta), height=0, distance=int(0.1 / dt))
    
    # Update plots
    p.set_data(time[peaks], (r * np.cos(theta))[peaks])
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
plt.show()
