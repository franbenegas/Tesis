# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:15:04 2023

@author: beneg
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import csv
import pandas as pd
from scipy.optimize import curve_fit
import math
get_ipython().run_line_magic('matplotlib', 'qt5')

os.chdir (r'C:\Users\beneg\OneDrive\Escritorio')

#%% Funciones que uso
def conservacion(M,sigma_1,sigma_2,i,j):
    M[i,j] = 2*(sigma_1 + sigma_2)*M[i,j]
    M[i + 1,j] = - sigma_1*M[i + 1,j]
    M[i - 1,j] = - sigma_1*M[i - 1,j]
    M[i,j + 1] = - sigma_2*M[i,j + 1]
    M[i,j - 1] = - sigma_2*M[i,j - 1]
#######################################################################################################
def matrix_posta(m,n):
    F = np.zeros((m*n, m*n)) # pre-allocate memory for F
    k = 0
    for i in range(1,m+1):
        for j in range(1,n+1):
            P = np.ones((m+2,n+2))
            conservacion(P, sigma_1, sigma_2, i, j) # modify P in place
            # matriz con todos ceros menos los 5 lugares
            mask = (P == 1)
            # Set the locations where the mask is True (i.e., where the matrix has 1's) to 0
            P[mask] = 0
            #Condiciones de contorno
            if i - 1 < 1:
                P[i,j] += P[i - 1,j]
            if i + 1 > m:
                  P[i,j] += P[i + 1,j]
            if j - 1 < 1:
                P[i,j] += P[i,j - 1]
            if j + 1 > n:
                P[i,j] += P[i,j + 1]
            F[k] = P[1:m+1, 1:n+1].flatten() # store values in F
            k += 1

    return F
##############################################################################################
def Jacobi(A, b, x0, tol=1e-6, n=1000):
    U= np.triu(A, k=1)
    L= np.tril(A,k=-1)
    D=np.diag(np.diag(A), k=0)
    x=x0
    for i in range(n):
        D_inv= np.linalg.inv(D)
        xk_1= x
        x= np.dot( D_inv, np.dot(-(L+U), xk_1) ) + np.diag(D_inv*b)
        error= np.linalg.norm(x-xk_1)
        
    if error<tol:
        return x
    
    return x

###########################################################################################
def contactos(M,i1, j1,i2,j2):
    contacto_i1_j1= np.zeros(m*n)
    contacto_i2_j2= contacto_i1_j1.copy()
    b= contacto_i2_j2.copy()
    
    contacto_i1_j1[((n*(i1-1)) + (j1-1))] = 1
    contacto_i2_j2[((n*(i2-1)) + (j2-1))] = 1
    b[((n*(i1-1)) + (j1-1))] = 1
    b[((n*(i2-1)) + (j2-1))]=-1
    
    
    M[((n*(i1-1)) + (j1-1)),:]= contacto_i1_j1
    M[((n*(i2-1)) + (j2-1)),:]= contacto_i2_j2
    
    
    return M,b
######################################################################################################
def corriente(solucion1, i, j):
    i = i - 1
    j = j - 1
    nrows, ncols = solucion1.shape
    
    # Calculate values of neighbors (assuming missing values are equal to central element)
    left = solucion1[i, j-1] if j > 0 else solucion1[i, j]
    right = solucion1[i, j+1] if j < ncols-1 else solucion1[i, j]
    up = solucion1[i-1, j] if i > 0 else solucion1[i, j]
    down = solucion1[i+1, j] if i < nrows-1 else solucion1[i, j]
    
    # Calculate i_0
    i_0 = -sigma_1*(left-solucion1[i,j] + right-solucion1[i,j]) - sigma_2*(up-solucion1[i,j] + down-solucion1[i,j])
    
    return i_0
################################################################################################
def Diferencia(solucion1,p1,q1,p2,q2):   
    return solucion1[p1-1,q1-1] - solucion1[p2-1,q2-1]

# ####################################################################################################
def equipotenciales(Solucion):
    xlist = np.linspace(0, n, n)
    ylist = np.linspace(0, m, m)
    X, Y = np.meshgrid(xlist, ylist)
    
    fig,ax=plt.subplots(figsize=(12, 6))
    ax.set_title('Potential Contour Plot')
    cp = ax.contourf(X, Y, Solucion,levels= 100, cmap= 'RdGy')
    ax.contour(X, Y, Solucion, levels=100, colors= 'black', alpha= 0.4)
    fig.colorbar(cp) 
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm) ')
    
    plt.show()

##########################################################################################################
def get_neighbors_positions(m, n, contacts, radius):
    matrix = np.ones((m,n))
    positions_list = []
    for i, j in contacts:
        i = i - 1
        j = j - 1
        positions = []
        nrows, ncols = matrix.shape

        for r in range(1, radius+1):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if abs(di) + abs(dj) <= r:
                        ii = i + di
                        jj = j + dj
                        if ii >= 0 and ii < nrows and jj >= 0 and jj < ncols:
                            positions.append((ii, jj))

        chosen_position = random.choice(positions)
        if not isinstance(chosen_position, tuple) or len(chosen_position) != 2:
            chosen_position = (i+1, j+1)
        else:
            chosen_position = (chosen_position[0]+1, chosen_position[1]+1)
        positions_list.append(chosen_position)
    return positions_list
#########################################################################################################
def plot_matrix_with_chosen_cells(m, n, contactos, radius):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle(f'Chosen Cells within Radius = {radius}')
    
    matrix = np.ones((m,n))
    
    for idx, (i, j) in enumerate(contactos):
        i = i - 1
        j = j - 1
        positions = []
        nrows, ncols = matrix.shape

        for r in range(1, radius+1):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if abs(di) + abs(dj) <= r:
                        ii = i + di
                        jj = j + dj
                        if ii >= 0 and ii < nrows and jj >= 0 and jj < ncols:
                            positions.append((ii, jj))

        chosen_position = random.choice(positions)
        if not isinstance(chosen_position, tuple) or len(chosen_position) != 2:
            chosen_position = (i+1, j+1)
        else:
            chosen_position = (chosen_position[0]+1, chosen_position[1]+1)

        axs[int(idx/2), idx%2].imshow(matrix, cmap='gray', vmin=0, vmax=1)
        axs[int(idx/2), idx%2].plot(j, i, 'ro', markersize=10, label='Original Cell') # Show original cell with red circle
        axs[int(idx/2), idx%2].plot(chosen_position[1]-1, chosen_position[0]-1, 'bo', markersize=10, label='New Cell') # Show new cell with blue circle

        for pos in positions:
            axs[int(idx/2), idx%2].plot(pos[1], pos[0], 'bo', alpha=0.3, markersize=5) # Show cells within radius with blue dots

        axs[int(idx/2), idx%2].set_title(f'Contacto {idx+1}')
        axs[int(idx/2), idx%2].legend()
        
    plt.show()
##############################################################################################
def plot_matrix_with_chosen_cells2(m, n, contactos, radius, chosen_pairs=None):
    num_plots = len(contactos)
    num_rows = math.ceil(num_plots / 2)  # Number of rows for subplots
    num_cols = min(num_plots, 2)  # Number of columns for subplots

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))
    fig.suptitle(f'Chosen Cells within Radius = {radius}')

    matrix = np.ones((m, n))

    for idx, (i, j) in enumerate(contactos):
        i = i - 1
        j = j - 1
        positions = []
        nrows, ncols = matrix.shape

        for r in range(1, radius + 1):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if abs(di) + abs(dj) <= r:
                        ii = i + di
                        jj = j + dj
                        if ii >= 0 and ii < nrows and jj >= 0 and jj < ncols:
                            positions.append((ii, jj))

        # Adjust the subplot indices
        row_idx = idx // num_cols
        col_idx = idx % num_cols

        axs[row_idx, col_idx].imshow(matrix, cmap='gray', vmin=0, vmax=1)
        axs[row_idx, col_idx].plot(j, i, 'ro', markersize=10, label='Original Cell')  # Show original cell with red circle

        for pos in positions:
            axs[row_idx, col_idx].scatter(pos[1], pos[0], marker='o', color='blue', alpha=0.3, s=5)  # Show cells within radius with blue dots

        axs[row_idx, col_idx].set_title(f'Contacto {idx + 1}')
        axs[row_idx, col_idx].legend()

        if chosen_pairs is not None and idx < len(chosen_pairs):
            chosen_positions = chosen_pairs[idx]
            for chosen_position in chosen_positions:
                if not isinstance(chosen_position, tuple) or len(chosen_position) != 2:
                    chosen_position = (i + 1, j + 1)
                else:
                    chosen_position = (chosen_position[0] + 1, chosen_position[1] + 1)

                axs[row_idx, col_idx].scatter(chosen_position[1] - 1, chosen_position[0] - 1, marker='o', color='green',
                                              s=100, label='Chosen Cell')  # Show chosen cell with green circle

    plt.show()
   
######################################################################################################################################################
def calculate_R1_mariano(Pares, m, n, sigma_1, sigma_2):
    i1, j1 = Pares[0]
    i2, j2 = Pares[1]
    p1, p2 = Pares[4]
    q1, q2 = Pares[5]
    
    matriz= np.asarray(matrix_posta(m, n))
    matriz, b1= contactos(matriz, i1, j1, i2, j2)
    x0= np.zeros_like(b1)
    
    solucion1 = Jacobi(matriz, b1, x0)
    
    solucion1= solucion1.reshape((m,n))
    
    solucion1 = solucion1/corriente(solucion1, i1, j1)
    
    dif = Diferencia(solucion1,p1, p2, q1, q2)
    
    return dif

def calculate_R2_mariano(Pares, m, n, sigma_1, sigma_2):
    i1, j1 = Pares[2]
    i2, j2 = Pares[3]
    p1, p2 = Pares[4]
    q1, q2 = Pares[5]

    matriz= np.asarray(matrix_posta(m, n))
    matriz, b1= contactos(matriz, i1, j1, i2, j2)
    x0= np.zeros_like(b1)
    
    solucion1 = Jacobi(matriz, b1, x0)
    
    solucion1= solucion1.reshape((m,n))
    
    solucion1 = solucion1/corriente(solucion1, i1, j1)
    
    dif = Diferencia(solucion1,p1, p2, q1, q2)
    
    return dif
###############################################################################################
def cubica1(x,a,b,c,d):
    y = a*x**3 + b*x**2 + c*x + d
    return y

#%% Nuevo metodo barriendo las posiciones del par de potencial para un par de sigmas

sigma_1, sigma_2 = 1, 1
m, n= 11, 55 # Dimension de la matriz

# Par de contacto R1
i_R1, j_R1 = 1, int((n + 1)/2)
k_R1, l_R1 = m, int((n + 1)/2)

# Par de contacto R2
i_R2, j_R2 = int((m + 1)/2), n
k_R2, l_R2 = int((m + 1)/2), 1


'''
Calculo de la matriz con los contactos de R1
'''

matriz= np.asarray(matrix_posta(m, n))
matriz, b1= contactos(matriz, i_R1, j_R1, k_R1, l_R1)
x0= np.zeros_like(b1)

solucion = Jacobi(matriz, b1, x0)

solucion= solucion.reshape((m,n))

solucion_posta1 = solucion/corriente(solucion, i_R1, j_R1)

'''
Calculo de la matriz con los contactos de R2
'''

matriz= np.asarray(matrix_posta(m, n))
matriz, b1= contactos(matriz, i_R2, j_R2, k_R2, l_R2)
x0= np.zeros_like(b1)

solucion = Jacobi(matriz, b1, x0)

solucion= solucion.reshape((m,n))

solucion_posta2 = solucion/corriente(solucion, i_R2, j_R2)

'''
Barrido sobre el par de potencial
'''

Xd = []
Xd2 = []
for t in range(1, int((n + 1)/2)):
    p1, q1 = 1, int((n + 1)/2) + t
    p2, q2 = m, int((n + 1)/2) - t
    
    
    dif = Diferencia(solucion_posta1, p1, q1, p2, q2)
    Xd.append(dif)
    dif = Diferencia(solucion_posta2, p1, q1, p2, q2)
    Xd2.append(dif)
    
    
Xd3 = (np.asarray(Xd2)/np.asarray(Xd))

#%% Grafico
x  = 2*np.arange(1,(n + 1)/2)/m
plt.figure()
plt.subplot(1,3,1)
plt.plot(x,Xd,'o')
plt.subplot(1,3,2)
plt.plot(x,Xd2,'o')
plt.subplot(1,3,3)
plt.semilogy(x,Xd3,'o')

#%% Barrido de la poscion del par de potencial para varios sigmas

sigmas = [(1,1), (2,1), (1,2)]

m, n = 11, 111 # Dimension de la matriz

lol1 = []
lol2 = []
dota = []

# Par de contacto R1
i_R1, j_R1 = 1, int((n + 1)/2)
k_R1, l_R1 = m, int((n + 1)/2)

# Par de contacto R2
i_R2, j_R2 = int((m + 1)/2), n
k_R2, l_R2 = int((m + 1)/2), 1

start_time = time.time()

for sigma_1, sigma_2 in sigmas:
        
        '''
        Calculo de la matriz con los contactos de R1
        '''

        matriz= np.asarray(matrix_posta(m, n))
        matriz, b1= contactos(matriz, i_R1, j_R1, k_R1, l_R1)
        x0= np.zeros_like(b1)

        solucion = Jacobi(matriz, b1, x0)

        solucion= solucion.reshape((m,n))

        solucion_posta1 = solucion/corriente(solucion, i_R1, j_R1)

        '''
        Calculo de la matriz con los contactos de R2
        '''

        matriz= np.asarray(matrix_posta(m, n))
        matriz, b1= contactos(matriz, i_R2, j_R2, k_R2, l_R2)
        x0= np.zeros_like(b1)

        solucion = Jacobi(matriz, b1, x0)

        solucion= solucion.reshape((m,n))

        solucion_posta2 = solucion/corriente(solucion, i_R2, j_R2)
        
        '''
        Barrido sobre el par de potencial
        '''

        Xd = []
        Xd2 = []
        for t in range(1, int((n + 1)/2)):
            p1, q1 = 1, int((n + 1)/2) + t
            p2, q2 = m, int((n + 1)/2) - t
            
            
            dif = Diferencia(solucion_posta1, p1, q1, p2, q2)
            Xd.append(dif)
            dif = Diferencia(solucion_posta2, p1, q1, p2, q2)
            Xd2.append(dif)
            
            
        Xd3 = (np.asarray(Xd2)/np.asarray(Xd))
        
        lol1.append(Xd)
        lol2.append(Xd2)
        dota.append(Xd3)

        
end_time = time.time()
total_time = end_time - start_time
if total_time >= 3600:
    total_time = total_time/3600
    print(f"Total time: {total_time:.5f} hours")
elif total_time >= 60:
    total_time = total_time/60
    print(f"Total time: {total_time:.5f} minutes")
else:
    print(f"Total time: {total_time:.5f} seconds")
        
        
#%%

'''
Esto calcula la anisotropia R1-R2/R1+R2
'''

Resta = []
for i in range(len(lol1)):
    sublist_result = []
    for j in range(len(lol1[i])):
        sublist_result.append(lol1[i][j] - lol2[i][j])
    Resta.append(sublist_result)

Suma = []
for i in range(len(lol1)):
    sublist_result = []
    for j in range(len(lol1[i])):
        sublist_result.append(lol1[i][j] + lol2[i][j])
    Suma.append(sublist_result)

result = []
for i in range(len(Resta)):
    row = []
    for j in range(len(Resta[i])):
        row.append(Resta[i][j]/Suma[i][j])
    result.append(row)
    
'''
Esto calcula la diferencia de la anisotropia con los valores para el caso isotropico
'''

subtracted_lists = [[b - a for a, b in zip(result[0], lst)] for lst in result]

#%% Grafico

Labels = ['Sigma_2/Sigma_1 = 1', 'Sigma_2/Sigma_1 = 2', 'Sigma_2/Sigma_1 = 0.5']

x  = 2*np.arange(1,int((n + 1)/2))/m

plt.figure()

plt.subplot(1,3,1)
plt.title('Anisotropia')
for l in result:

    plt.plot(x,l,'o-')
    
plt.subplot(1,3,2)
plt.title('R')
for R in dota:
    plt.semilogy(x,R,'o')
    

plt.subplot(1,3,3)
plt.title('Diferencia de anisotropia')
for v in subtracted_lists:
    plt.plot(x,v,'o-')
    
plt.legend(Labels)

#%% Vario el largo y barro sobre los puntos de medicon con los dos metodos, dejando los 
# contactos en los bordes y tambien en el medio

sigmas = [(1,1), (2,1), (1,2)]

m = 11 # Dejo fijo el ancho

ns = [55,67,77,89,99,111] # Vario el largo

#Esta basofia es lo que quiero graficar 
R_bordes = []
Anisotropia_borde = []
Dif_anisotropia_borde = []

R_fijo = []
Anisotropia_fijo = []
Dif_anisotropia_fijo = []


with tqdm(total=len(sigmas)*len(ns)) as pbar_h:
    start_time = time.time()
    
    
    for n in ns:
        
        # Primer par de contacto que va a variar segun el largo pero se 
        # quedan en el medio
        i_R1, j_R1 = 1, int((n + 1)/2)
        k_R1, l_R1 = m, int((n + 1)/2)
        
        # Segundo par de contactos que siempre quedan fijos
        i_R2, j_R2 = int((m + 1)/2), n
        k_R2, l_R2 = int((m + 1)/2), 1
        

        # Segundo para de contactos que se moveran al centro 
        i_R2_2, j_R2_2 = int((m + 1)/2), int((n + 1)/2) + 27 #Me lo dejara fijo en la
        k_R2_2, l_R2_2 = int((m + 1)/2), int((n + 1)/2) - 27 #Matriz de 55
        
        R1_contactos_borde_dif_sigma= []
        R2_contactos_borde_dif_sigma = []
        R_contactos_borde_dif_sigma = []
        
        R1_contactos_fijo_dif_sigma= []
        R2_contactos_fijo_dif_sigma = []
        R_contactos_fijo_dif_sigma = []
        
        for sigma_1, sigma_2 in sigmas:
            
                     
            '''
            Calculo de la matriz con los contactos de R1
            '''
    
            matriz= np.asarray(matrix_posta(m, n))
            matriz, b1= contactos(matriz, i_R1, j_R1, k_R1, l_R1)
            x0= np.zeros_like(b1)
    
            solucion = Jacobi(matriz, b1, x0)
    
            solucion= solucion.reshape((m,n))
    
            solucion_posta1 = solucion/corriente(solucion, i_R1, j_R1)

            '''
            Calculo de la matriz con los contactos de R2
            '''

            matriz= np.asarray(matrix_posta(m, n))
            matriz, b1= contactos(matriz, i_R2, j_R2, k_R2, l_R2)
            x0= np.zeros_like(b1)

            solucion = Jacobi(matriz, b1, x0)

            solucion= solucion.reshape((m,n))

            solucion_posta2 = solucion/corriente(solucion, i_R2, j_R2)
            
            

            '''
            Calculo de la matriz con los contactos de R2 para matriz fija en el centro
            '''

            matriz= np.asarray(matrix_posta(m, n))
            matriz, b1= contactos(matriz, i_R2_2, j_R2_2, k_R2_2, l_R2_2)
            x0= np.zeros_like(b1)

            solucion = Jacobi(matriz, b1, x0)

            solucion= solucion.reshape((m,n))

            solucion_posta3 = solucion/corriente(solucion, i_R2_2, j_R2_2)
            
            '''
            Variacion del lugar de los puntos de medicion 
            '''
            #Caso borde
            R1_contactos_borde = []
            R2_contactos_borde = []
            R_contactos_borde = []
            
            #Caso fijo
            R2_contactos_fijo = []
            R_contactos_fijo = []
            
            for t in range(1, int((n + 1)/2)):
                p1, q1 = 1, int((n + 1)/2) + t
                p2, q2 = m, int((n + 1)/2) - t
                
                dif = Diferencia(solucion_posta1, p1, q1, p2, q2)
                R1_contactos_borde.append(dif)
                dif = Diferencia(solucion_posta2, p1, q1, p2, q2)
                R2_contactos_borde.append(dif)
                dif = Diferencia(solucion_posta3, p1, q1, p2, q2)
                R2_contactos_fijo.append(dif)
                
            pbar_h.update(1)
                
            R_contactos_borde.append(np.asarray(R2_contactos_borde)/np.asarray(R1_contactos_borde))
            
            R_contactos_fijo.append(np.asarray(R2_contactos_fijo)/np.asarray(R1_contactos_borde))

            
            R1_contactos_borde_dif_sigma.append(R1_contactos_borde)
            R2_contactos_borde_dif_sigma.append(R2_contactos_borde)
            R_contactos_borde_dif_sigma.append(R_contactos_borde)
            
            R1_contactos_fijo_dif_sigma.append(R1_contactos_borde)
            R2_contactos_fijo_dif_sigma.append(R2_contactos_fijo)
            R_contactos_fijo_dif_sigma.append(R_contactos_fijo)
            
        
        R_bordes.append(R_contactos_borde_dif_sigma)
        R_fijo.append(R_contactos_fijo_dif_sigma)
        
        '''
        Calculo de la anisotroipia para la de los borde
        '''
        Resta = []
        for i in range(len(R1_contactos_borde_dif_sigma)):
            sublist_result = []
            for j in range(len(R1_contactos_borde_dif_sigma[i])):
                sublist_result.append(R1_contactos_borde_dif_sigma[i][j] - R2_contactos_borde_dif_sigma[i][j])
            Resta.append(sublist_result)

        Suma = []
        for i in range(len(R1_contactos_borde_dif_sigma)):
            sublist_result = []
            for j in range(len(R1_contactos_borde_dif_sigma[i])):
                sublist_result.append(R1_contactos_borde_dif_sigma[i][j] + R2_contactos_borde_dif_sigma[i][j])
            Suma.append(sublist_result)

        Result = []
        for i in range(len(Resta)):
            row = []
            for j in range(len(Resta[i])):
                row.append( Resta[i][j]/ Suma[i][j])
            Result.append(row)
            
        subtracted_lists = [[b - a for a, b in zip(Result[0], lst)] for lst in Result]
        
        Anisotropia_borde.append(Result)
        Dif_anisotropia_borde.append(subtracted_lists)
        
        '''
        Calculo de la anisotroipia para la fija
        '''
        Resta = []
        for i in range(len(R1_contactos_borde_dif_sigma)):
            sublist_result = []
            for j in range(len(R1_contactos_fijo_dif_sigma[i])):
                sublist_result.append(R1_contactos_fijo_dif_sigma[i][j] - R2_contactos_fijo_dif_sigma[i][j])
            Resta.append(sublist_result)

        Suma = []
        for i in range(len(R1_contactos_fijo_dif_sigma)):
            sublist_result = []
            for j in range(len(R1_contactos_fijo_dif_sigma[i])):
                sublist_result.append(R1_contactos_fijo_dif_sigma[i][j] + R2_contactos_fijo_dif_sigma[i][j])
            Suma.append(sublist_result)

        Result = []
        for i in range(len(Resta)):
            row = []
            for j in range(len(Resta[i])):
                row.append( Resta[i][j]/ Suma[i][j])
            Result.append(row)
            
        subtracted_lists = [[b - a for a, b in zip(Result[0], lst)] for lst in Result]
        
        Anisotropia_fijo.append(Result)
        Dif_anisotropia_fijo.append(subtracted_lists)
        
    end_time = time.time()
    total_time = end_time - start_time
    if total_time >= 3600:
        total_time = total_time/3600
        print(f"Total time: {total_time:.5f} hours")
    elif total_time >= 60:
        total_time = total_time/60
        print(f"Total time: {total_time:.5f} minutes")
    else:
        print(f"Total time: {total_time:.5f} seconds")
        
#%% Grafico

'''
Dif_anisotropia_borde is a list of 6 lists, where each of the 6 lists contains 3 lists.
So you have a nested data structure where each of the 6 outer lists represents a different
value of n, and each of those 6 lists contains 3 inner lists, which represent 
the 3 different values of sigma2/sigma1.
'''

colors = ['red', 'green', 'blue']
markers = ['o', 's', '^', '*', 'D', 'x']

plt.close('Bordes')
plt.figure('Bordes')
# Assuming 'data' is your list of 6 lists
for i in range(len(Dif_anisotropia_borde)):
    sublist = Dif_anisotropia_borde[i]
    list_label = f'm, n = {m}, {ns[i]}'  # Label for the list
    for j in range(len(sublist)):
        x  = 2*np.arange(1,(ns[i] + 1)/2)/m
        plt.plot(x,sublist[j], color=colors[j], marker=markers[i], linestyle='-', label=f'{colors[j]}' if i==0 else None, markersize=5)
    plt.plot([], [], label=list_label, color='k', marker=markers[i], linestyle='', markersize=5)  # Add a dummy plot to the legend for the list label

plt.legend()
plt.show()

plt.close('Fijo')
plt.figure('Fijo')
# Assuming 'data' is your list of 6 lists
for i in range(len(Dif_anisotropia_fijo)):
    sublist = Dif_anisotropia_fijo[i]
    list_label = f'm, n = {m}, {ns[i]}'  # Label for the list
    for j in range(len(sublist)):
        x  = 2*np.arange(1,(ns[i] + 1)/2)/m
        plt.plot(x,sublist[j], color=colors[j], marker=markers[i], linestyle='-', label=f'{colors[j]}' if i==0 else None, markersize=5)
    plt.plot([], [], label=list_label, color='k', marker=markers[i], linestyle='', markersize=5)  # Add a dummy plot to the legend for the list label



plt.legend()
plt.show()

#%% Guardo los datos simulados

sigma_ratios = [1.0, 0.5, 2.0]

# Create a 2D list to store the data
data = [[0] * len(sigma_ratios) for i in range(len(ns))]

# Fill in the data
for i in range(len(ns)):
    for j in range(len(sigma_ratios)):
        sublist = Dif_anisotropia_borde[i]
        data[i][j] = sublist[j]

# Name the columns with sigma_2/sigma_1 values
column_names = [f'sigma2/sigma1 = {ratio:.1f}' for ratio in sigma_ratios]

# Name the rows with n values
row_names = [f'n = {n}' for n in ns]

filename = "Dif_anisotropia_borde.csv"

# Save the data to a CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([''] + column_names)  # Write column names
    for i in range(len(data)):
        writer.writerow([row_names[i]] + data[i])  # Write row names and data


'''
Guardo para el fijo
'''
        
# Create a 2D list to store the data
data = [[0] * len(sigma_ratios) for i in range(len(ns))]

# Fill in the data
for i in range(len(ns)):
    for j in range(len(sigma_ratios)):
        sublist = Dif_anisotropia_fijo[i]
        data[i][j] = sublist[j]

# Name the columns with sigma_2/sigma_1 values
column_names = [f'sigma2/sigma1 = {ratio:.1f}' for ratio in sigma_ratios]

# Name the rows with n values
row_names = [f'n = {n}' for n in ns]

filename = "Dif_anisotropia_fijo_separacion_55.csv"

# Save the data to a CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([''] + column_names)  # Write column names
    for i in range(len(data)):
        writer.writerow([row_names[i]] + data[i])  # Write row names and data


#%% Aca voy variar para una muestra fija, la posicion de los contactos y buscar la sens

sigmas = [(1,1), (2,1), (1,2)]

# ns = [55,67,77,89,99,111]

m,n = 11, 99

k = 6 #Cantidad de puntos que voy a queres barrer

# Primer par de contacto que va a variar segun el largo pero se 
# quedan en el medio
i_R1, j_R1 = 1, int((n + 1)/2)
k_R1, l_R1 = m, int((n + 1)/2)

R1s = []
R2s = []

with tqdm(total=len(sigmas)*k) as pbar_h:
    start_time = time.time()

    for sigma_1, sigma_2 in sigmas:
        
                 
        '''
        Calculo de la matriz con los contactos de R1
        '''
    
        matriz= np.asarray(matrix_posta(m, n))
        matriz, b1= contactos(matriz, i_R1, j_R1, k_R1, l_R1)
        x0= np.zeros_like(b1)
    
        solucion = Jacobi(matriz, b1, x0)
    
        solucion= solucion.reshape((m,n))
    
        solucion_posta1 = solucion/corriente(solucion, i_R1, j_R1)
        
        '''
        Variacion del lugar de los puntos de medicion 
        '''
        
        R1 = []
        
        for t in range(1, int((n + 1)/2)):
            p1, q1 = 1, int((n + 1)/2) + t
            p2, q2 = m, int((n + 1)/2) - t
            
            dif = Diferencia(solucion_posta1, p1, q1, p2, q2)
            R1.append(dif)
        '''
        Aca vario la posicion de los contactos desde el medio hasta afuera
        '''
        
        R2_contactos_expandiendo = []
        
        for i in range(1,k + 1):
            # Segundo para de contactos que se moveran al centro 
            i_R2_2, j_R2_2 = int((m + 1)/2), int((n + 1)/2) + i*int(((n + 1)/2)/k)#Me lo dejara fijo en la
            k_R2_2, l_R2_2 = int((m + 1)/2), int((n + 1)/2) - i*int(((n + 1)/2)/k) #Matriz de 55
            
            '''
            Calculo de la matriz con los contactos de R2 para matriz fija en el centro
            '''
    
            matriz= np.asarray(matrix_posta(m, n))
            matriz, b1= contactos(matriz, i_R2_2, j_R2_2, k_R2_2, l_R2_2)
            x0= np.zeros_like(b1)
    
            solucion = Jacobi(matriz, b1, x0)
    
            solucion= solucion.reshape((m,n))
    
            solucion_posta3 = solucion/corriente(solucion, i_R2_2, j_R2_2)
            
            R2_ponele = []
            
            for t in range(1, int((n + 1)/2)):
                p1, q1 = 1, int((n + 1)/2) + t
                p2, q2 = m, int((n + 1)/2) - t
                
                dif = Diferencia(solucion_posta3, p1, q1, p2, q2)
                R2_ponele.append(dif)
            
            R2_contactos_expandiendo.append(R2_ponele)
            
            pbar_h.update(1)
            
        R1s.append(R1)
        R2s.append(R2_contactos_expandiendo)


    end_time = time.time()
    total_time = end_time - start_time
    if total_time >= 60:
        total_time = total_time/60
        print(f"Total time: {total_time:.5f} minutes")
    else:
        print(f"Total time: {total_time:.5f} seconds")
           
Anisotropia = []
for i in range(len(R1s)):
    Resta = []
    for j in range(len(R2s[i])):
        sublist_result = []
        for k in range(len(R1s[i])):
            sublist_result.append(R1s[i][k] - R2s[i][j][k])
        Resta.append(sublist_result)
        
    Suma = []
    for j in range(len(R2s[i])):
        sublist_result = []
        for k in range(len(R1s[i])):
            sublist_result.append(R1s[i][k] + R2s[i][j][k])
        Suma.append(sublist_result)
        
    Result = []
    for j in range(len(R2s[i])):
        sublist_result = []
        for k in range(len(R1s[i])):
            sublist_result.append(Resta[j][k] / Suma[j][k])
        Result.append(sublist_result)
    Anisotropia.append(Result)
    
differences = []
for i in range(len(Anisotropia)):
    diff_sublist = []
    for j in range(len(Anisotropia[i])):
        if i == 0:
            diff_sublist.append([0]*len(Anisotropia[i][j]))
        else:
            diff_sublist.append([Anisotropia[i][j][k] - Anisotropia[0][j][k] for k in range(len(Anisotropia[i][j]))])
    differences.append(diff_sublist)



#%% Grafico
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^', '*', 'D', 'x']

x  = 2*np.arange(1,(n + 1)/2)/m

plt.close('Differences')
plt.figure('Differences')

# Assuming 'differences' is your list of lists
for i, diff_sublist in enumerate(differences):
    for j, diff_subsublist in enumerate(diff_sublist):
        if j >= len(markers):
            marker = markers[j % len(markers)]
        else:
            marker = markers[j]
        if i >= len(colors):
            color = colors[i % len(colors)]
        else:
            color = colors[i]
        list_label = f'Position {int((n + 1)/2) + i*int(((n + 1)/2)/k)}'  # Label for the list
        plt.plot(x, diff_subsublist, color=color, marker=marker, linestyle='-', label=list_label if i==0 else None, markersize=5)

plt.legend()
plt.show()

#%% Guardo los datos

df = pd.DataFrame(differences).T
n = 99
k = 6
L = []
for i in range(1,k + 1):
    L.append(int((n + 1)/2) + i*int(((n + 1)/2)/k))
    

row_names = [f'L = {n}' for n in L]

df.insert(0, 'Posicion', row_names)

# Assuming 'data' is your list of 3 lists, each containing 5 sublists
df.columns = ['Posicion','sigma_2/sigma_1 = 1.0', 'sigma_2/sigma_1 = 0.5', 'sigma_2/sigma_1 = 2.0']

df.to_csv('Anisotropia 11x99 al variar R2.csv', index=False)


#%% Defino los sigmas y los contactos

m, n = 11,111
i = 1
k = 6
t = int(m/2)

Sigmas = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]

R1 = []
R2 = []
Psi = []

i_R1, j_R1 = 1, int((n + 1)/2)

k_R1, l_R1 = m, int((n + 1)/2)

i_R2, j_R2 = int((m + 1)/2), int((n + 1)/2) + i*int(((n + 1)/2)/k)

k_R2, l_R2 = int((m + 1)/2), int((n + 1)/2) - i*int(((n + 1)/2)/k)

p1, q1 = 1, int((n + 1)/2) + t

p2, q2 = m, int((n + 1)/2) - t


Pares = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2),(p1, q1),(p2, q2)]

#%% Busco anisotropia en funcion de psi

with tqdm(total=len(Sigmas)) as pbar_h:
    start_time = time.time()

    for sigma_1, sigma_2 in Sigmas:
        
        Psi.append((sigma_1 - sigma_2)/(sigma_1 + sigma_2))
        
        R1.append(calculate_R1_mariano(Pares, m, n, sigma_1, sigma_2))
        
        R2.append(calculate_R2_mariano(Pares, m, n, sigma_1, sigma_2))

        pbar_h.update(1)


    end_time = time.time()
    total_time = end_time - start_time
    if total_time >= 3600:
        total_time = total_time/3600
        print(f"Total time: {total_time:.5f} hours")
    elif total_time >= 60:
        total_time = total_time/60
        print(f"Total time: {total_time:.5f} minutes")
    else:
        print(f"Total time: {total_time:.5f} seconds")


#%% Creo el data frame

# Zip the lists together
zipped_list = list(zip(Sigmas,Psi, R1, R2))

# Create a DataFrame from the zipped list
df = pd.DataFrame(zipped_list)

# Set the column names of the DataFrame
df.columns = ["Sigmas","Psi", "R1", "R2"]

df['Anisotropia'] = (df['R1'] - df['R2'])/(df['R1'] + df['R2'])

df.to_csv('Anisotropia 11x89 eb funcion de Psi.csv', index=False)
#%% Grafico

plt.figure()
plt.plot(df['Psi'],df['Anisotropia'],'o')

for i, label in enumerate(df['Sigmas']):
    plt.text(df['Psi'][i], df['Anisotropia'][i], label)
plt.xlabel('Psi')
plt.ylabel('Anisotropia')

#%% Ajuste de la anisotropia

popt, pcov = curve_fit(cubica1,df['Psi'],df['Anisotropia'])

x= np.linspace(-0.7,0.7,100)
plt.plot(x,cubica1(x,*popt),label='Ajuste')


#%% Aca busco la anisotropia para condiciones random de contactos de corriente

df = pd.read_csv('Anisotropia 11x89 eb funcion de Psi.csv')
df_contactos = pd.read_csv('Contactos.csv')

m, n = 11,89
i = 1
k = 6
t = int(m/2)
radius = 3

Sigmas = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]


i_R1, j_R1 = 1, int((n + 1)/2)

k_R1, l_R1 = m, int((n + 1)/2)

i_R2, j_R2 = int((m + 1)/2), int((n + 1)/2) + i*int(((n + 1)/2)/k)

k_R2, l_R2 = int((m + 1)/2), int((n + 1)/2) - i*int(((n + 1)/2)/k)

p1, q1 = 1, int((n + 1)/2) + t

p2, q2 = m, int((n + 1)/2) - t

Pares = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2),(p1, q1),(p2, q2)]
contacts = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2)]


# df_contactos['Pares'] = Pares
#%%

with tqdm(total=len(Sigmas)*7) as pbar_h:
    start_time = time.time()
    
    for o in range(3,11):
    
        Contactos_random = get_neighbors_positions(m, n, contacts, radius)
        Contactos_random.append(Pares[4])
        Contactos_random.append(Pares[5])
        
      
        # Assign the Series to the DataFrame column
        df_contactos[f'contactos_{o}'] = Contactos_random
        
        R1 = []
        R2 = []
        # Psi = []
        
        for sigma_1, sigma_2 in Sigmas:
            
            # Psi.append((sigma_1 - sigma_2)/(sigma_1 + sigma_2))
            
            R1.append(calculate_R1_mariano(Contactos_random, m, n, sigma_1, sigma_2))
            
            R2.append(calculate_R2_mariano(Contactos_random, m, n, sigma_1, sigma_2))
    
            pbar_h.update(1)

        
        df[f'R1_{o}']= R1
        df[f'R2_{o}']= R2
        df[f'Anisotropia_{o}'] = (df[f'R1_{o}'] - df[f'R2_{o}'])/(df[f'R1_{o}'] + df[f'R2_{o}'])
    
    end_time = time.time()
    total_time = end_time - start_time
    if total_time >= 3600:
        total_time = total_time/3600
        print(f"Total time: {total_time:.5f} hours")
    elif total_time >= 60:
        total_time = total_time/60
        print(f"Total time: {total_time:.5f} minutes")
    else:
        print(f"Total time: {total_time:.5f} seconds")


#%% Grafico la anisotripia moviendo todos los graficos al 0
plt.figure()
plt.plot(df['Psi'],df['Anisotropia']-df.loc[0,'Anisotropia'],'o', label='Ideal')

for i in range(3):
    plt.plot(df['Psi'],df[f'Anisotropia_{i}']- df.loc[0,f'Anisotropia_{i}'],'o',label=f'Iteracion {i}')
    
plt.legend()

#%% Guardo los contactos

df.to_csv('Anisotropia 11x89 eb funcion de Psi.csv', index=False,mode='a',header=False)


df_contactos.to_csv('Contactos.csv', index=False,mode='a',header=False)

#%% Grafico de los contactos

plt.figure()
# Extract x and y coordinates
x = [point[1] for point in Pares]
y = [point[0] for point in Pares]

# Plot the scatter plot
plt.scatter(x, y,label= 'Ideal')

for i in range(3):
    x = [point[1] for point in df_contactos[f'contactos_{i}']]
    y = [point[0] for point in df_contactos[f'contactos_{i}']]
    
    plt.scatter(x, y,label= f'iteracion {i}')
    
plt.legend()


