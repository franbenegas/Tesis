# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 00:41:35 2024

@author: beneg
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import matplotlib.animation as animation
from scipy.stats import norm

# Differential equation function using PyTorch
def funcion(P, t, tau, A_0, As):
    dPdt = (-P + A_0 * As +1)/ tau
    return dPdt

# RK4 integration method using PyTorch
def rk4(f, y, t, dt, tau, A_0, As):
    k1 = dt * f(y, t, tau, A_0, As)
    k2 = dt * f(y + k1 / 2, t + dt / 2, tau, A_0, As)
    k3 = dt * f(y + k2 / 2, t + dt / 2, tau, A_0, As)
    k4 = dt * f(y + k3, t + dt, tau, A_0, As)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Integration function for the system using PyTorch
def integrate_system(time, dt, tau, A_0, As_values):
    # Initial conditions
    P = torch.zeros_like(time)
    P[0] = 1.0  # Set initial condition for P

    # Loop over time steps
    for ix in range(len(time) - 1):
        As = As_values[ix]  # Get As value at current time step
        P[ix + 1] = rk4(funcion, P[ix], time[ix], dt, tau, A_0, As)
    
    return P

# Main function to optimize
def optimize_parameters(time, As_values, Pressure_values, Pressure_error, initial_params, lr=0.01, epochs=1000):
    # Convert to torch tensors for autograd
    As_values_torch = torch.tensor(As_values, dtype=torch.float32)
    Pressure_values_torch = torch.tensor(Pressure_values, dtype=torch.float32)
    Pressure_values_error_torch = torch.tensor(Pressure_error, dtype=torch.float32)

    # Initialize parameters as torch tensors with gradients enabled
    tau = torch.tensor(initial_params[0], dtype=torch.float32, requires_grad=True)
    A_0 = torch.tensor(initial_params[1], dtype=torch.float32, requires_grad=True)

    # Optimizer (Gradient Descent or SGD)
    # optimizer = torch.optim.SGD([tau, A_0], lr=lr)
    
    #Adam option
    optimizer = torch.optim.Adam([tau, A_0], lr=lr)

    dt = torch.mean(torch.diff(time))  # Time step size

    
    fig, ax = plt.subplots(2,1,figsize=(14,7))
    ax[0].set_ylabel('MSE Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].set_ylabel('tau')
    ax[1].set_xlabel('A_0')
    
    
    with tqdm(total=epochs, desc ="Epochs") as pbar_h:
        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()  # Zero out gradients
    
            # Integrate system with current parameters
            P_result = integrate_system(time, dt, tau, A_0, As_values_torch)
    
            # # Compute loss (MSE between P_result and Pressure_values)
            # loss = torch.mean((P_result - Pressure_values_torch) ** 2)
            
            # # Backpropagation to compute gradients
            # loss.backward()
    
            # # Gradient descent step
            # optimizer.step()
            
            
            # Define weights based on the inverse of Pressure_values_error (use +1e-8 to avoid division by zero)
            weights = 1 / (Pressure_values_error_torch ** 2 + 1e-8)
            
            # Compute weighted mean squared error
            weighted_loss = torch.mean(weights * (P_result - Pressure_values_torch) ** 2)
                
            weighted_loss.backward()
    
            # Gradient descent step
            optimizer.step()
            # Update the plot every 100 epochs
            if epoch % 50 == 0:
                # ax[0].plot(epochs_list, loss_list, color='C0')
                # ax[1].plot(A0_list, tau_list)
                # ax[0].plot(epoch, loss.item(), 'o',color='C0')
                ax[0].plot(epoch, weighted_loss.item(), 'o',color='C0')
                ax[1].plot(A_0.item(), tau.item(),'o',color='C0')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)  # Allows plot to update in real-time
                
            pbar_h.update(1)
    
    # Display final plot
    # plt.show()

    # Return the optimized parameters
    return tau.item(), A_0.item()

#random search of grid parameters
# torch.set_num_threads(1)
def optimize_parameters_random(time, As_values, Pressure_values, initial_params, lr=0.01, epochs=1000):
    As_values_torch = torch.tensor(As_values, dtype=torch.float32)
    Pressure_values_torch = torch.tensor(Pressure_values, dtype=torch.float32)

    tau = torch.tensor(initial_params[0], dtype=torch.float32, requires_grad=True)
    A_0 = torch.tensor(initial_params[1], dtype=torch.float32, requires_grad=True)

    #SGD option
    # optimizer = torch.optim.SGD([tau, A_0], lr=lr)
    
    
    #Adam option
    optimizer = torch.optim.Adam([tau, A_0], lr=lr)
    
    dt = torch.mean(torch.diff(time))
    loss_list = []
    
    with tqdm(total=epochs, desc="Epochs") as pbar_h:
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Simulate system integration (dummy function, replace with real)
            P_result = integrate_system(time, dt, tau, A_0, As_values_torch)
            
            loss = torch.mean((P_result - Pressure_values_torch) ** 2)
            loss.backward()
            optimizer.step()
            pbar_h.update(1)
            loss_list.append(loss.item())

    return tau.item(), A_0.item(), loss_list[-1],loss_list[0]


# Grid search with random initialization and optimization
def run_random_grid_search(n, taus, A0s, time, As_values, Pressure_values):
    taus_grid, A0s_grid = np.meshgrid(taus, A0s)
    loss_grid = np.ones_like(taus_grid)

    taus_selec, A0s_selec = [], []
    loss_initial, loss_final = [], []
    tau_final, A0_final = [], []
    
    for _ in range(n):
        random_tau = random.choice(taus)
        random_A0 = random.choice(A0s)

        taus_selec.append(random_tau)
        A0s_selec.append(random_A0)

        initial_params = [random_tau, random_A0]
        tau_opt, A0_opt, final_loss, initial_loss = optimize_parameters_random(time, As_values, Pressure_values,
                                                                              initial_params, lr=0.1, epochs=500)

        # Locate grid index and store losses
        tau_idx = np.argmin(np.abs(taus - tau_opt))
        A0_idx = np.argmin(np.abs(A0s - A0_opt))
        loss_grid[tau_idx, A0_idx] = final_loss

        tau_final.append(taus[tau_idx])
        A0_final.append(A0s[A0_idx])
        loss_initial.append(initial_loss)
        loss_final.append(final_loss)

    return taus_selec, A0s_selec, tau_final, A0_final, loss_initial, loss_final

# Visualization function
def plot_results(taus_selec, A0s_selec, tau_final, A0_final, loss_initial, loss_final):
    fig = plt.figure(figsize=(14, 7), layout='constrained')
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(taus_selec, A0s_selec, loss_initial, color='C0', label='Initial')
    ax.scatter(tau_final, A0_final, loss_final, color='C1', label='Final')

    ax.set_xlabel('tau')
    ax.set_ylabel('A_0')
    ax.set_zlabel('loss')
    ax.legend()
    plt.show()
#%%
# Example usage
time = torch.linspace(0, 10, 100)  # Time points from 0 to 10 (now as a PyTorch tensor)
As_values = np.sin(time.numpy())   # Example input for As_values
Pressure_values = np.cos(time.numpy())  # Example target pressure data (data to fit)
initial_params = [12.66, -12.65]       # Initial guesses for tau and A_0

# Optimize parameters
tau_opt, A_0_opt = optimize_parameters(time, As_values, Pressure_values, initial_params)

# Plot the final result
P_final = integrate_system(time, dt=torch.mean(torch.diff(time)), tau=tau_opt, A_0=A_0_opt, As_values=torch.tensor(As_values))

plt.figure(figsize=(14,7))
plt.plot(time.numpy(), P_final.detach().numpy(), label='P_result (optimized)')
plt.plot(time.numpy(), Pressure_values, label='Pressure_values (target)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('P(t) and Pressure_values')
plt.legend()
# plt.show()

#%%




#%% ejemplo random
# Example usage
time = torch.linspace(0, 10, 100)  # Time points from 0 to 10 (now as a PyTorch tensor)
As_values = np.sin(time.numpy())   # Example input for As_values
Pressure_values = np.cos(time.numpy())  # Example target pressure data (data to fit)

taus = np.linspace(7, 7.2, 50)
A0s = np.linspace(-7.2, -7, 50)
n_random_starts = 5


taus_selec, A0s_selec, tau_final, A0_final, loss_initial, loss_final = run_random_grid_search(
    n_random_starts, taus, A0s, time, As_values, Pressure_values
)

plot_results(taus_selec, A0s_selec, tau_final, A0_final, loss_initial, loss_final)


#%%
# index_min = np.nanargmin(loss_final)

# initial_params = [tau_final[index_min], A0_final[index_min]] 
initial_params = [7.21, -7.18]       # Initial guesses for tau and A_0

# Optimize parameters
tau_opt, A_0_opt = optimize_parameters(time, As_values, Pressure_values, initial_params)


P_final = integrate_system(time, dt=torch.mean(torch.diff(time)), tau=tau_opt, A_0=A_0_opt, As_values=torch.tensor(As_values))

plt.figure(figsize=(14,7))
plt.plot(time.numpy(), P_final.detach().numpy(), label='P_result (optimized)')
plt.plot(time.numpy(), Pressure_values, label='Pressure_values (target)', linestyle='--')
plt.xlabel('Time')
plt.ylabel('P(t) and Pressure_values')
plt.legend()
#%% animaciones de la ecuacion


# Initialize plot
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(time.numpy(), Pressure_values, label='Pressure_values (target)', linestyle='--')
line3, = ax.plot(time.numpy(), P_final.detach().numpy(), label='P_result (optimized)')
line2, = ax.plot(time.numpy(), As_values, label='Forzado')

# Initialize the marker (red dot) that will move along with the frame
marker, = ax.plot([0], [1], 'C1o')  # 'ro' specifies a red circular marker

ax.set_xlabel('Time')
ax.set_ylabel('P(t) and Pressure_values')
ax.legend()

# Update function for animation
def update(frame):
    # Update the plot lines for the current frame
    line3.set_data(time.numpy()[:frame], P_final.detach().numpy()[:frame])
    line2.set_data(time.numpy()[:frame], As_values[:frame])
    
    # Update the position of the marker at the current frame
    marker.set_data([time.numpy()[frame-1]], [P_final.detach().numpy()[frame-1]])  # Use lists for single points
    
    return line3, line2, marker

# Create the animation
ani = animation.FuncAnimation(fig, func=update, frames=len(time), interval=50, blit=True)


#%%
directory = r'C:\Users\beneg\OneDrive\Escritorio\Tesis\Datos\Datos Fran'
os.chdir(directory)  # Change the working directory to the specified path
carpetas = os.listdir(directory)  # List all folders within the directory

## RoNe
pajaro = carpetas[0]  # Select the first folder (assumed to be related to 'RoNe')

subdirectory = os.path.join(directory, pajaro)

os.chdir(subdirectory)

Sound = np.loadtxt('average RoNe sonido 300', delimiter=',')
Pressure = np.loadtxt('average RoNe pressure', delimiter=',')
Rate = np.loadtxt('average RoNe rate', delimiter=',')

time_sound = Sound[0]
sound = Sound[1]
error_s = Sound[2]


time_pressure = Pressure[0]
pressure = Pressure[1]
error_pressure = Pressure[2]

time_rate = Rate[0]
rate = Rate[1]
error_rate = Rate[2]

dt = np.mean(np.diff(time_sound))
sound_derivative = np.gradient(sound,dt)



indice_sound = np.argmin(abs(time_sound - 0))
# # Example usage
time_sound = torch.from_numpy(time_sound)
sound = sound/np.mean(sound[:indice_sound])

def sigm(x):
    return 1 / (1 + np.exp(-x -.5))
plt.figure()
plt.plot(sound)
plt.plot(sound_derivative)
plt.plot((np.sign(sound_derivative-0.1)+1)/2)
# plt.plot(sigm(sound_derivative))

# As_values = sound
#%%
initial_params = [0.2, 0.42]       # Initial guesses for tau and A_0
# As_values=As_values-1
# Optimize parameters
tau_opt, A_0_opt = optimize_parameters(time_sound, As_values, pressure, error_pressure, initial_params)

# Plot the final result
P_final = integrate_system(time_sound, dt=torch.mean(torch.diff(time_sound)), tau=tau_opt, A_0=A_0_opt, As_values=As_values)


#%%
P_final = integrate_system(time_sound, dt=torch.mean(torch.diff(time_sound)), tau=0.2, A_0=0.42, As_values=sound-1)
fig, ax = plt.subplots(2,1,figsize=(14,7))
ax[0].plot(time_sound.numpy(),sound)
ax[1].plot(time_sound.numpy(), P_final, label='P_result (optimized)')
ax[1].errorbar(time_pressure, pressure, yerr =error_pressure, label='Pressure_values (target)', linestyle='--')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Pressure')
ax[0].set_ylabel('Sound')
ax[1].legend()

#%%
hola = (np.sign(sound_derivative-0.1)+1)/2

def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def funcion(P, t, pars):
    
    def sigm(x):
        return 1 / (1 + np.exp(-x))
    
    A_0, P0, t_up, t_down, As, Asp = pars
    dPdt = (t_down**-1 - (t_down**-1 - t_up**-1)*Asp)* (A_0 * As + P0 - P)
    return dPdt

# Integration function for the system using PyTorch
def integrate_system(time, dt, pars, sound, sound_derivative):
    # Initial conditions
    P = torch.zeros_like(time)
    P[0] = 1.0  # Set initial condition for P

    # Loop over time steps
    for ix in range(len(time) - 1):
        pars[-2] = sound[ix]  # Get As value at current time step
        pars[-1] = sound_derivative[ix]
        P[ix + 1] = rk4(funcion, P[ix], time[ix], dt, pars)
    
    return P
        #A_0, P0, t_up, t_down, sound, derivative 
pars = [0.45, 1, 0.2, 1, 1, 1]
P_final = integrate_system(time_sound, dt, pars, sound-1,hola)

# plt.close(fig)
fig, ax = plt.subplots(2,1,figsize=(14,7))
ax[0].plot(time_sound,sound-1)
ax[1].plot(time_sound, P_final, label='P_result')
ax[1].errorbar(time_pressure, pressure, yerr= error_pressure, label='Pressure_values (target)', linestyle='--')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Pressure')
ax[0].set_ylabel('Sound')
ax[1].legend()



#%%
import matplotlib.cm as cm  # To access colormap
def optimize_parameters(time, As_values, derivada, Pressure_values, Pressure_error, initial_params, lr=0.01, epochs=1000):
    # Convert to torch tensors for autograd
    time_torch = torch.tensor(time, dtype=torch.float32)
    As_values_torch = torch.tensor(As_values, dtype=torch.float32)
    Pressure_values_torch = torch.tensor(Pressure_values, dtype=torch.float32)
    Pressure_values_error_torch = torch.tensor(Pressure_error, dtype=torch.float32)
    derivada_torch = torch.tensor(derivada, dtype=torch.float32)
    # Set `P0` as a known constant
    P0 = initial_params[3]  # Define your known constant here

    # Initialize parameters as torch tensors with gradients enabled
    A_0 = torch.tensor(initial_params[0], dtype=torch.float32, requires_grad=True)
    t_up = torch.tensor(initial_params[1], dtype=torch.float32, requires_grad=True)
    t_down = torch.tensor(initial_params[2], dtype=torch.float32, requires_grad=True)
    
    # Adam optimizer
    optimizer = torch.optim.Adam([A_0, t_up, t_down], lr=lr)
    dt = torch.mean(torch.diff(time_torch))  # Time step size
    
    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    ax[0].set_ylabel('Weighted MSE Loss')
    ax[0].set_xlabel('Epoch')
    ax[1].set_ylabel('t_up')
    ax[1].set_xlabel('A_0')
    ax[2].set_ylabel('t_down')
    ax[2].set_xlabel('A_0')
    
    with tqdm(total=epochs, desc="Epochs") as pbar_h:
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Integrate the system using current parameters and known constant `P0`
            pars = [A_0, P0, t_up, t_down, 0, 0]  # Placeholders for `As` and `Asp`
            P_result = integrate_system(time_torch, dt, pars, As_values_torch, derivada_torch)
            
            # Calculate weights and weighted loss
            weights = 1 / (Pressure_values_error_torch ** 2 + 1e-8)
            weighted_loss = torch.mean(weights * (P_result - Pressure_values_torch) ** 2)
            
            # Backpropagate and optimize
            weighted_loss.backward()
            optimizer.step()
            
            # Generate a color based on the current epoch
            color = cm.viridis(epoch / epochs)  # Adjust the colormap as desired
            
            # Plot updating with color gradient
            if epoch % 50 == 0:
                ax[0].plot(epoch, weighted_loss.item(), 'o', color=color)
                ax[1].plot(A_0.item(), t_up.item(), 'o', color=color, label='t_up' if epoch == 0 else "")
                ax[2].plot(A_0.item(), t_down.item(), 'o', color=color, label='t_down' if epoch == 0 else "")
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
            pbar_h.update(1)
                
    return A_0.item(), t_up.item(), t_down.item()


initial_params = [0.1,0.2,1,1.0]


A_0_opt, t_up_opt, t_down_opt = optimize_parameters(time_sound, sound-1.0, hola, pressure, error_pressure, initial_params)


#%%
       #A_0, P0, t_up, t_down, sound, derivative 
pars = [0.5,1, 0.2, 2,1,1]
P_final = integrate_system(time_sound, dt, pars, sound-1,hola)

fig, ax = plt.subplots(2,1,figsize=(14,7))
ax[0].plot(time_sound,sound-1)
ax[1].plot(time_sound, P_final, label='P_result (optimized)')
ax[1].errorbar(time_pressure, pressure, yerr= error_pressure, label='Pressure_values (target)', linestyle='--')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Pressure')
ax[0].set_ylabel('Sound')
ax[1].legend()


#%%
def rk4(dxdt, x, t, dt, pars):
    k1 = dxdt(x, t, pars) * dt
    k2 = dxdt(x + k1 * 0.5, t, pars) * dt
    k3 = dxdt(x + k2 * 0.5, t, pars) * dt
    k4 = dxdt(x + k3, t, pars) * dt
    return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def funcion_patada(w,t,pars):
    w0, tau, A = pars
    dwdt = (-w + w0)/tau + A
    return dwdt
    


def integrate_system(time, dt, pars, indice_sound, amplitud_patada):
    # Initial conditions
    w = np.zeros_like(time)
    w[0] = pars[0]  # Set initial condition for P
    
    # Loop over time steps
    for ix in range(len(time) - 1):

        if ix == indice_sound:
            pars[-1] = amplitud_patada
            w[ix + 1] = rk4(funcion_patada, w[ix], time[ix], dt, pars)
        else:
            pars[-1] = 0.0
            w[ix + 1] = rk4(funcion_patada, w[ix], time[ix], dt, pars)
    
    return w

       #W0, tau, patada
pars = [1.25, 20, 0]
indice_rate = np.argmin(abs(time_rate - 0))
# indice_rate_maximo = np.argmin(abs(time_rate - max()))
amplitud_patada = 4

rate_simulado = integrate_system(time_rate, dt, pars, indice_rate, amplitud_patada)
  
fig, ax = plt.subplots(figsize=(14,7))
ax.set_title('Patada delta')
ax.errorbar(time_rate,rate,error_rate, label = 'Datos RoNe')
ax.plot(time_rate,rate_simulado, label = 'Modelo exitable')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()


#%%

fig, ax = plt.subplots(figsize=(14, 7))
ax.errorbar(time_rate,rate,error_rate, label = 'Datos RoNe')
line3, = ax.plot(time_rate, rate_simulado, label='Modelo exitable')


# Initialize the marker (red dot) that will move along with the frame
marker, = ax.plot([0], [1], 'C1o')  # 'ro' specifies a red circular marker

ax.set_xlabel('Time (s)')
ax.set_ylabel('Rate (Hz)')
ax.legend(fancybox=True,shadow=True)
plt.tight_layout()
# Update function for animation
def update(frame):
    # Update the plot lines for the current frame
    line3.set_data(time_rate[:frame], rate_simulado[:frame])
    
    # Update the position of the marker at the current frame
    marker.set_data([time_rate[frame-1]], [rate_simulado[frame-1]])  # Use lists for single points
    
    return line3, marker

# Create the animation
ani = animation.FuncAnimation(fig, func=update, frames=len(time_rate), interval=30, blit=True)

# ani.save(filename="ajuste_rate.gif", writer="pillow")







