# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:37:04 2024

@author: facuf
"""

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'qt5')
#RK4  
def rk4(dxdt, x, t, dt, *args, **kwargs):
    x = np.asarray(x)
    k1 = np.asarray(dxdt(x, t, *args, **kwargs))*dt
    k2 = np.asarray(dxdt(x + k1*0.5, t, *args, **kwargs))*dt
    k3 = np.asarray(dxdt(x + k2*0.5, t, *args, **kwargs))*dt
    k4 = np.asarray(dxdt(x + k3, t, *args, **kwargs))*dt
    return x + (k1 + 2*k2 + 2*k3 + k4)/6

def sigm(x):
    return 1 / (1 + np.exp(-10*x))


#Vector field
def f(v, t, pars):
      
    cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp = [par for par in pars] 
    r, theta, mu, w = v[0], v[1], v[2], v[3]
    
    
    #Ecuaciones
    drdt =  mu * r - cr * r**3
    
    dthetadt = w - ci * r**2 

    dmudt = - mu / tau + A_s0 * As
    
    dwdt = (t_down**-1 - (t_down**-1 - t_up**-1)*sigm(Asp-.5)) * (A_s0 * As + w0 - w)

    return [drdt, dthetadt, dmudt, dwdt]
#%%
time = np.arange(0, 60, 0.01)
from scipy.stats import norm

As = norm.pdf(time, loc=30, scale=3)
As = As / np.max(As)

Asp = np.concatenate(([0], np.diff(As)))
Asp = Asp / np.max(Asp)
plt.figure(figsize=(14, 7))
plt.plot(time, As, label="amplitud sonido")
plt.plot(time, Asp, label="derivada amplitud sonido")
plt.plot(time, sigm(Asp-.5), label="sigmoidea de As_punto")
plt.legend()

#%%
#parameters
cr = 1
w0 = 2*np.pi - 4
ci = - 5
tau = 0.1

A_s0 = 10

t_up = .05
t_down = 25

#      cr, w0, ci, As, tau, A_s0, t_up, t_down, Asp 
pars = [cr, w0, ci, 1, tau, A_s0, t_up, t_down, 1]

dt = np.mean(np.diff(time))
r = np.zeros_like(time)
theta = np.zeros_like(time)
mu = np.zeros_like(time)
w = np.zeros_like(time)

#initial condition
r_0, theta_0, mu_0 = .25, 0, 0
r[0] = r_0
theta[0] = theta_0
mu[0] = mu_0
w[0] = w0

for ix, tt in enumerate(time[:-1]):  
    pars[3] = As[ix] 
    pars[-1] = Asp[ix]
    r[ix+1], theta[ix+1], mu[ix+1], w[ix+1] = rk4(f, [r[ix], theta[ix], mu[ix], 
                                                      w[ix]], tt, dt, pars)


plt.figure(figsize=(14, 7))
plt.plot(time,r * np.cos(theta))
plt.plot(time, As)
#%%
from scipy.signal import find_peaks
peaks, _ = find_peaks(r*np.cos(theta), height=0, distance=int(0.1/dt))

plt.plot(time, (r*np.cos(theta)))
plt.plot(time[peaks], (r*np.cos(theta))[peaks], '.')

rate_model = np.diff(time[peaks])**-1
time_rate_model = .5 * ( time[peaks[1:]]+time[peaks[:-1]] ) 
#%%
# save_folder = 'C:/Users/facuf/Desktop/Facu 2024/Tesis de Licenciatura - Aviones y sueño/Results/Modelado - Hopf y seguidor/'

fig, ax = plt.subplots(4, 1, figsize=(20, 3*2.5))


# code_name = os.path.basename(__file__)

# fig.suptitle(filep[9:-4]+"; "+code_name, fontsize=14)


# ax[0].set_title(title, fontsize=16)

ax[0].plot(time, As, label="sound envelope")

ax[1].plot(time, r * np.cos(theta), label=r"$r \cos(\phi)$")
ax[1].plot(time[peaks], (r*np.cos(theta))[peaks], '.-')

ax[2].plot(time_rate_model, rate_model, '.-', label="rate model")
ax[3].plot(time,w)
# for k in range(3):
#     ax[k].legend(fontsize=15)
# fig.tight_layout()
# plt.savefig(save_folder+"model_29052024.pdf")

#%%
plt.figure()
plt.plot((r * np.cos(theta))[1000:2000],(r * np.sin(theta))[1000:2000],label='Antes')
plt.plot((r * np.cos(theta))[2000:2500],(r * np.sin(theta))[2000:2500],label='Despues')
plt.scatter((r * np.cos(theta))[2000:2001],(r * np.sin(theta))[2000:2001],color='r')
plt.legend(fancybox=True,shadow=True)
