import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose,identity,array
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

#PARÁMETROS DEL MODELO --------------------------------------------------------------------------------------------------------------------
X=0.75; Y=0.20; M_total=5 #PARÁMETROS CONSTANTES
R_total=12; L_total=40.0; T_central=1.5;  #VALORES INICIALES
capas=100;it1=10;aux1=10;it2=6;aux2=8 #Parametros del mallado

#plot1= Plot del error total en función de la temperatura central para un R y L totales fijos
#Representación1= Plot del mallado del error total en función de las luminosidad y radios totales 
#Representación2= Plot del mallado de la temperatura central óptima en función de las luminosidad y radios totales 
#Representación3= Tablas del modelo que optimiza el error
#Representación4= Plot de las magnitudes física en función del radio (un plot para cada magnitud)
#Representación5= Plot de todas las magnitudes físicas normalizadas (en un único plot)
#datos= Devuelve los valores R_total,M_total,L_total,T_central,r_down,M,L,T,rho


#MODELO TFG, sin optimizar y optimizado
modelo_sin_optimizar_R_y_L(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1=True) #Printea todas las tablas con los resultados del procedimiento de integración
iteraciones=20 #20
modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=True,representacion2=True,representacion3=True,representacion4=True,representacion5=True)


