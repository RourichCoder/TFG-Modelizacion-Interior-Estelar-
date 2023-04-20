import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose,identity,array
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

'''Resuelve la estructura de una estrella primero sin variar R_total y L_total y después hallando el R_total y L_total que optimizan el error

    Args:
        -Parametros característicos de la estrella
        -Valores iniciales
        -Parámetros del mallado
    Return:
        -Gráficas y tablas con los resultados
'''s
#PARÁMETROS DEL MODELO --------------------------------------------------------------------------------------------------------------------
X=0.75; Y=0.20; M_total=5 #Parámetros característicos de la estrella
R_total=12; L_total=40.0; T_central=1.5; #Valores iniciales
capas=100;it1=10;aux1=10;iteraciones=20  #Parametros del mallado

#MODELO TFG, sin optimizar y optimizado
modelo_sin_optimizar_R_y_L(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1=True,plot2=True) #Printea todas las tablas con los resultados del procedimiento de integración y representa el error total en función de la temperatura central y las magnitudes normalizas (en la misma figura) en función del radio.
modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=True,representacion2=True,representacion3=True,representacion4=True,representacion5=True) #Printea el modelo óptimo y todas las gráficas sobre el proceso y los resultados


