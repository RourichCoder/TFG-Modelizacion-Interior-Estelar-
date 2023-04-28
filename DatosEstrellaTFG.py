import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose,identity,array
from tabulate import tabulate   
import matplotlib.pyplot as plt

'''Resuelve la estructura de una estrella primero sin variar R_total y L_total y después hallando el R_total y L_total que optimizan el error

    Args:
        -Parametros característicos de la estrella
        -Valores iniciales
        -Parámetros del mallado
    Return:
        -Gráficas y tablas con los resultados
'''
#PARÁMETROS DEL MODELO --------------------------------------------------------------------------------------------------------------------
X=0.75; Y=0.20; M_total=5 #Parámetros característicos de la estrella
R_total=12; L_total=40.0; T_central=1.5; #Valores iniciales
capas=100;it1=12;aux1=20;iteraciones=10;deltaR=0.1;deltaL=1;profundidad=16 #Parametros del mallado
representacion1=True;representacion2=True;representacion3=True;representacion4=True;representacion5=True

#MODELO TFG, sin optimizar y optimizado
modelo_sin_optimizar_R_y_L(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1=True,plot2=True) #Printea todas las tablas con los resultados del procedimiento de integración y representa el error total en función de la temperatura central y las magnitudes normalizas (en la misma figura) en función del radio.
(R_total,M_total,L_total,T_central,r_down,P,T,L,M,rho,M_frontera)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1,representacion2,representacion3,representacion4,representacion5,deltaR,deltaL,profundidad) #Printea el modelo óptimo y todas las gráficas sobre el proceso y los resultados


