# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:08:36 2023

@author: Yo
"""
import os
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import csv

get_ipython().run_line_magic('matplotlib', 'qt5')

os.chdir (r'C:\Users\beneg\OneDrive\Escritorio\Labo 6 y 7')
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

###################################################################################################################################################
def calculate_R1(Pares, m, n, sigma_1, sigma_2):
    i1, j1 = Pares[0]
    i2, j2 = Pares[1]
    p1, p2 = Pares[2]
    q1, q2 = Pares[3]
    
    matriz= np.asarray(matrix_posta(m, n))
    matriz, b1= contactos(matriz, i1, j1, i2, j2)
    x0= np.zeros_like(b1)
    
    solucion1 = Jacobi(matriz, b1, x0)
    
    solucion1= solucion1.reshape((m,n))
    
    solucion1 = solucion1/corriente(solucion1, i1, j1)
    
    dif = Diferencia(solucion1,p1, p2, q1, q2)
    
    return dif

def calculate_R2(Pares, m, n, sigma_1, sigma_2):
    i1, j1 = Pares[0]
    i2, j2 = Pares[2]
    p1, p2 = Pares[1]
    q1, q2 = Pares[3]

    matriz= np.asarray(matrix_posta(m, n))
    matriz, b1= contactos(matriz, i1, j1, i2, j2)
    x0= np.zeros_like(b1)
    
    solucion1 = Jacobi(matriz, b1, x0)
    
    solucion1= solucion1.reshape((m,n))
    
    solucion1 = solucion1/corriente(solucion1, i1, j1)
    
    dif = Diferencia(solucion1,p1, p2, q1, q2)
    
    return dif

def calculate_R(R1, R2):
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    return R2/R1 
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



#%% #Plot de la grilla y sus potenciales
sigma_1, sigma_2 = 1, 1
m, n= 10, 50


# Primer contacto
i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

# Segundo contacto
i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

# Primer punto de medicion
p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

# Segundo punto de medicion
q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

matriz= np.asarray(matrix_posta(m, n))
matriz, b1= contactos(matriz, i1, j1, i2, j2)
x0= np.zeros_like(b1)

solucion = Jacobi(matriz, b1, x0)

solucion= solucion.reshape((m,n))

solucion_posta = solucion/corriente(solucion, i1, j1)

# contour plot
equipotenciales(solucion_posta)

#%% Grafico de la matriz con contactos del montgomery y los random

m ,n = 10, 20
radius = 3

# Primer contacto
i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

# Segundo contacto
i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

# Primer punto de medicion
p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

# Segundo punto de medicion
q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

Contactos = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]

plot_matrix_with_chosen_cells(m, n, Contactos, radius)

#%% Variaciones del Montgomery para muestra cuadrada

R1_mont = []
R2_mont = []
R_mont = []

# Antes eran listas con las 3 listas correspondientes a cada sigma variando la dimension
# Ahora van a ser listas con listas de 3 puntos, uno para cada sigma

sigmas = [(1,1), (2,1), (1,2)]

m = 10
n = 10
radius = 2


'''
Esto es para el montgomery cuadrado normal
'''

R1 = []
R2 = []

# Primer contacto
i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

# Segundo contacto
i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

# Primer punto de medicion
p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

# Segundo punto de medicion
q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
    
for sigma_1, sigma_2 in sigmas:
    
    '''
    R1
    '''
        
    R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2))
    
    '''
    R2
    '''
    
    R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2))
    
R1_mont.append(R1)
R2_mont.append(R2)


R_mont.append(calculate_R(R1, R2))

'''
Ahora hago las condiciones random
'''
for h in range(20):
    # Calculo de la matriz con la que voy a generar mis condiciones random
    

    # Primer contacto
    i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

    # Segundo contacto
    i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

    # Primer punto de medicion
    p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

    # Segundo punto de medicion
    q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

    Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
     
    random_positions = get_neighbors_positions(m, n, Pares, radius)
    
    
    R1 = []
    R2 = []
    
    for sigma_1, sigma_2 in sigmas:
        
        '''
        R1
        '''
        
        R1.append(calculate_R1(random_positions, m, n, sigma_1, sigma_2))
        
        '''
        R2
        '''
        
        R2.append(calculate_R2(random_positions, m, n, sigma_1, sigma_2))
        
    R1_mont.append(R1)
    R2_mont.append(R2)
    
    R_mont.append(calculate_R(R1, R2))

#%% Grafico

plt.close('XD')
plt.figure('XD')

plt.subplot(1,3,1)
# Create separate lists for each element
first_elem = []
second_elem = []
third_elem = []

for inner_list in R1_mont:
    first_elem.append(inner_list[0])
    second_elem.append(inner_list[1])
    third_elem.append(inner_list[2])

plt.title('R1')
plt.plot(first_elem[0], 'kx',label= 'Montgomery')
plt.plot(first_elem[1:], 'bo', label='Sigma_2/Sigma_1 = 1')
plt.plot(second_elem[0], 'kx')
plt.plot(second_elem[1:], 'ro', label='Sigma_2/Sigma_1 = 2')
plt.plot(third_elem[0], 'kx')
plt.plot(third_elem[1:], 'go', label='Sigma_2/Sigma_1 = 0.5')


plt.subplot(1,3,2)
# Create separate lists for each element
first_elem = []
second_elem = []
third_elem = []

for inner_list in R2_mont:
    first_elem.append(inner_list[0])
    second_elem.append(inner_list[1])
    third_elem.append(inner_list[2])

plt.plot(first_elem[0], 'kx',label= 'Montgomery')
plt.plot(first_elem[1:], 'bo', label='Sigma_2/Sigma_1 = 1')
plt.plot(second_elem[0], 'kx')
plt.plot(second_elem[1:], 'ro', label='Sigma_2/Sigma_1 = 2')
plt.plot(third_elem[0], 'kx')
plt.plot(third_elem[1:], 'go', label='Sigma_2/Sigma_1 = 0.5')

plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3)

plt.subplot(1,3,3)
# Create separate lists for each element
first_elem = []
second_elem = []
third_elem = []

for inner_list in R_mont:
    first_elem.append(inner_list[0])
    second_elem.append(inner_list[1])
    third_elem.append(inner_list[2])

plt.title('R2/R1')
plt.plot(first_elem[0], 'kx',label= 'Montgomery')
plt.plot(first_elem[1:], 'bo', label='Sigma_2/Sigma_1 = 1')
plt.plot(second_elem[0], 'kx')
plt.plot(second_elem[1:], 'ro', label='Sigma_2/Sigma_1 = 2')
plt.plot(third_elem[0], 'kx')
plt.plot(third_elem[1:], 'go', label='Sigma_2/Sigma_1 = 0.5')   
#%% barrido del montgomery dejando fijo los contactos y expandiendo para un lado

R1_mont_fijo = []
R2_mont_fijo = []
R_mont_fijo = []

sigmas = [(1,1), (2,1), (1,2)]

# Dimension de la grilla
m = 10 #nro de filas 

ns = [10,20,30,40,50,60,70,80,90,100] # y nro columnas


with tqdm(total=len(sigmas)*len(ns)) as pbar:

    for sigma_1, sigma_2 in tqdm(sigmas):
        print(f"sigma_1 = {sigma_1}, sigma_2 = {sigma_2}")
        start_time = time.time()
        R1 = []
        R2 = []
        for n in ns:
            
            # Primer contacto
            i1, j1 = 1, 1 #fila y columna del primer contacto
    
            # Segundo contacto
            i2, j2 = 10, 1 #fila y columna del segundo contacto
    
            # Primer punto de medicion
            p1,p2 = 1, 10 #fila y columna del primer punto de medicion
    
            # Segundo punto de medicion
            q1, q2 = 10, 10 #fila y columna del segundo punto de medicion
            
            Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
            
            '''
            Calculo R1
            '''

            R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2)) 
               
            
            '''
            Calculo R2
            '''
            R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2))
            
            pbar.update(1)
            
        R1 = [x/R1[0] for x in R1]
        R1_mont_fijo.append(R1)
        R2 = [x/R2[0] for x in R2]
        R2_mont_fijo.append(R2)
        
        R_mont_fijo.append(calculate_R(R1, R2))
        
        end_time = time.time()
        total_time = end_time - start_time
        if total_time >= 60:
            total_time = total_time/60
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} minutes")
        else:
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} seconds")
#%% Grafico
x = np.asarray(ns)/m

Labels = ['Sigma_2/Sigma_1 = 1', 'Sigma_2/Sigma_1 = 2', 'Sigma_2/Sigma_1 = 0.5']

plt.figure('Expansion para un lado')
for R1 in R1_mont_fijo:
    plt.subplot(1, 3, 1)
    plt.title('R1')
    plt.plot(x, R1,'-o')
for R2 in R2_mont_fijo:
    plt.subplot(1, 3, 2)
    plt.title('R2')
    plt.plot(x, R2,'-o')
for Rs in R_mont_fijo:
    plt.subplot(1, 3, 3)
    plt.title('R2/R1')
    plt.plot(x, Rs,'-o')
    

plt.legend(Labels)


#%% Grafico del barrido del montgomery dejando fijo los contactos y expandiendo para un lado

x = np.asarray(ns)/m

Labels = ['Sigma_2/Sigma_1 = 1', 'Sigma_2/Sigma_1 = 2', 'Sigma_2/Sigma_1 = 0.5']

plt.figure('Expansion para un lado')
for R1 in R1_mont_fijo:
    plt.subplot(1, 3, 1)
    plt.title('R1')
    plt.plot(x, R1,'-o')
for R2 in R2_mont_fijo:
    plt.subplot(1, 3, 2)
    plt.title('R2')
    plt.plot(x, R2,'-o')
for Rs in R_mont_fijo:
    plt.subplot(1, 3, 3)
    plt.title('R2/R1')
    plt.plot(x, Rs,'-o')
    

plt.legend(Labels)

#%% Barrido del montgomery dejando fijo los contactos y expandiendo para ambos lados

R1_lib = []
R2_lib = []
R_lib = []

sigmas = [(1,1), (2,1), (1,2)]

# Dimension de la grilla
m = 10 #nro de filas y nro columnas

ns = [10,50,60,70,80,90,100]


with tqdm(total=len(sigmas)*len(ns)) as pbar:

    for sigma_1, sigma_2 in tqdm(sigmas):
        print(f"sigma_1 = {sigma_1}, sigma_2 = {sigma_2}")
        start_time = time.time()
        R1 = []
        R2 = []
        for n in ns:
            
            # Primer contacto
            i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto
    
            # Segundo contacto
            i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto
    
            # Primer punto de medicion
            p1,p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion
    
            # Segundo punto de medicion
            q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion
            
            Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
            
            '''
            Calculo R1
            '''

            R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2)) 


            '''
            Calculo R2
            '''
            
            R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2)) 
            
            pbar.update(1)
            
        R1 = [x/R1[0] for x in R1]
        R1_lib.append(R1)
        R2 = [x/R2[0] for x in R2]
        R2_lib.append(R2)
            
        R_lib.append(calculate_R(R1, R2))
        
        end_time = time.time()
        total_time = end_time - start_time
        if total_time >= 60:
            total_time = total_time/60
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} minutes")
        else:
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} seconds")
        
        
#%% Aca lo grafico

x = np.asarray(ns)/m

Labels = ['Sigma_2/Sigma_1 = 1', 'Sigma_2/Sigma_1 = 2', 'Sigma_2/Sigma_1 = 0.5']

plt.figure('Expansion para ambos lados')
for R1 in R1_lib:
    plt.subplot(1, 3, 1)
    plt.title('R1')
    plt.plot(x, R1,'o-')
for R2 in R2_lib:
    plt.subplot(1, 3, 2)
    plt.title('R2')
    plt.plot(x, R2,'o-')    
for Rs in R_lib:
    plt.subplot(1, 3, 3)
    plt.title('R2/R1')
    plt.plot(x, Rs,'o-')
    

plt.legend(Labels)


#%% Barrido a ambos lados con condiciones random

m, n = 10, 20
radius = 2

# Primer contacto
i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

# Segundo contacto
i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

# Primer punto de medicion
p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

# Segundo punto de medicion
q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

Contactos = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]

random_positions = get_neighbors_positions(m, n, Contactos, radius)
deltas = []
for c, r in zip(Contactos, random_positions):
    diff = (r[0]-c[0], r[1]-c[1])
    deltas.append(diff)

# Dimension de la grilla
m = 10 #nro de filas y nro columnas

ns = [50,60,70,80,90,100]

R1sb = []
R2sb = []
Rb = []

sigmas = [(1,1), (2,1), (1,2)]

with tqdm(total=len(sigmas)*len(ns)) as pbar:
    
    for sigma_1, sigma_2 in tqdm(sigmas):

        print(f"sigma_1 = {sigma_1}, sigma_2 = {sigma_2}")
        start_time = time.time()
        R1 = []
        R2 = []
        n = 10
        
        # Primer contacto
        i1, j1 = 1, 1 #fila y columna del primer contacto
        
        # Segundo contacto
        i2, j2 = 10, 1 #fila y columna del segundo contacto
        
        # Primer punto de medicion
        p1, p2 = 1, 10 #fila y columna del primer punto de medicion
        
        # Segundo punto de medicion
        q1, q2 = 10, 10 #fila y columna del segundo punto de medicion
        
        Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
        
        '''
        Calculo R1 para el montgomery cuadrado 10x10
        '''
  
        R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2)) 
        
        '''
        Calculo R2 para el montgomery cuadrado en 10x10
        '''
        
        R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2)) 
        
        for n in ns:
            
            # Primer contacto
            i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto
        
            # Segundo contacto
            i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto
        
            # Primer punto de medicion
            p1,p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion
        
            # Segundo punto de medicion
            q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion
            
            Contactos = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
            
            Pares = []
            for c, r in zip(Contactos, deltas):
                diff = (r[0]+c[0], r[1]+c[1])
                Pares.append(diff) 
          
            '''
            Calculo R1
            '''

            R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2))
            
            '''
            Calculo R2
            '''
            
            R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2)) 
            
            pbar.update(1)
            
            
        R1 = [x/R1[0] for x in R1]
        R1sb.append(R1)
        R2 = [x/R2[0] for x in R2]
        R2sb.append(R2)
            
        Rb.append(calculate_R(R1, R2))
    
        end_time = time.time()
        total_time = end_time - start_time
        if total_time >= 60:
            total_time = total_time/60
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} minutes")
        else:
            print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} seconds")

#%% Grafico
ns = [10,50,60,70,80,90,100]
x = np.asarray(ns)/m

Labels = ['Sigma_2/Sigma_1 = 1', 'Sigma_2/Sigma_1 = 2', 'Sigma_2/Sigma_1 = 0.5']

plt.figure('Expansion random para ambos lados')
for R1 in R1sb:
    plt.subplot(1, 3, 1)
    plt.title('R1')
    plt.plot(x, R1,'o-')
for R2 in R2sb:
    plt.subplot(1, 3, 2)
    plt.title('R2')
    plt.plot(x, R2,'o-')    
for Rs in Rb:
    plt.subplot(1, 3, 3)
    plt.title('R2/R1')
    plt.plot(x, Rs,'o-')

#%% Barrido sobre varias condiciones random

R1_rand = []
R2_rand = []
R_rand = []

with tqdm(total=10*len(sigmas)*len(ns)) as pbar_h:
    start_time = time.time()
    for h in range(10):
        m, n = 10, 20
        radius = 2

        # Primer contacto
        i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto

        # Segundo contacto
        i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto

        # Primer punto de medicion
        p1, p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion

        # Segundo punto de medicion
        q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion

        Contactos = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]

        random_positions = get_neighbors_positions(m, n, Contactos, radius)
        deltas = []
        for c, r in zip(Contactos, random_positions):
            diff = (r[0]-c[0], r[1]-c[1])
            deltas.append(diff)

        # Dimension de la grilla
        m = 10 #nro de filas y nro columnas

        ns = [50,60,70,80,90,100]

        R1sb = []
        R2sb = []
        Rb = []

        # with tqdm(total=len(sigmas)*len(ns), desc=f"Iteration {h}") as pbar:
        for sigma_1, sigma_2 in sigmas:
            # pbar.set_description(f"Iteration {h}: sigma_1 = {sigma_1}, sigma_2 = {sigma_2}")
            # start_time = time.time()
            R1 = []
            R2 = []
            n = 10

            # Primer contacto
            i1, j1 = 1, 1 #fila y columna del primer contacto

            # Segundo contacto
            i2, j2 = 10, 1 #fila y columna del segundo contacto

            # Primer punto de medicion
            p1, p2 = 1, 10 #fila y columna del primer punto de medicion

            # Segundo punto de medicion
            q1, q2 = 10, 10 #fila y columna del segundo punto de medicion

            Pares = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]

            '''
            Calculo R1 para el montgomery cuadrado 10x10
            '''

            R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2)) 

            '''
            Calculo R2 para el montgomery cuadrado en 10x10
            '''

            R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2)) 

            for n in ns:
                
                # Primer contacto
                i1, j1 = 1, int(n/2) - 4 #fila y columna del primer contacto
            
                # Segundo contacto
                i2, j2 = 10, int(n/2) - 4 #fila y columna del segundo contacto
            
                # Primer punto de medicion
                p1,p2 = 1, int(n/2) + 5 #fila y columna del primer punto de medicion
            
                # Segundo punto de medicion
                q1, q2 = 10, int(n/2) + 5 #fila y columna del segundo punto de medicion
                
                Contactos = [(i1, j1),(i2, j2),(p1, p2),(q1, q2)]
                
                Pares = []
                for c, r in zip(Contactos, deltas):
                    diff = (r[0]+c[0], r[1]+c[1])
                    Pares.append(diff) 
              
                '''
                Calculo R1
                '''
    
                R1.append(calculate_R1(Pares, m, n, sigma_1, sigma_2))
                
                '''
                Calculo R2
                '''
                
                R2.append(calculate_R2(Pares, m, n, sigma_1, sigma_2)) 
                
                pbar_h.update(1)
                
                
            R1 = [x/R1[0] for x in R1]
            R1sb.append(R1)
            R2 = [x/R2[0] for x in R2]
            R2sb.append(R2)
                
            Rb.append(calculate_R(R1, R2))
        
            # end_time = time.time()
            # total_time = end_time - start_time
            # if total_time >= 60:
            #     total_time = total_time/60
            #     print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} minutes")
            # else:
            #     print(f"Total time for sigma ({sigma_1},{sigma_2}): {total_time:.5f} seconds")
    
    
    R1_rand.append(R1sb)
    R2_rand.append(R2sb)
    R_rand.append(Rb)
    
    end_time = time.time()
    total_time = end_time - start_time
    if total_time >= 60:
        total_time = total_time/60
        print(f"Total time: {total_time:.5f} minutes")
    else:
        print(f"Total time: {total_time:.5f} seconds")





