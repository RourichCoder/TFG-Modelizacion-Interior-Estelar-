import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M
import SolucionRtotYLtotFijos
from SolucionRtotYLtotFijos import epsilon1_nu, ritmo_generacion_energia

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose, identity, array, log, log10, float64, isfinite
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

'''Resuelve la estructura de dos estrellas con la misma masas, pero distintas composición química.

    Args:
        -Parametros característicos de cada estrella
        -Valores iniciales de cada estrella
        -Parámetros del mallado
    Return:
        Devuelve una tabla con la T_ef, R_total y L_toal en función de la composición química
'''

#PARÁMETROS DEL MODELO --------------------------------------------------------------------------------------------------------------------
M_total=5 #Masa de la estrella
capas=100;it1=12;aux1=20;iteraciones=2  #Parametros del mallado

#Estrella 1 (PROPUESTA TFG)
X1=0.75; Y1=0.20; Z1=1-X1-Y1 #Composición química
R_total=12; L_total=40.0; T_central=1.5; #Valores iniciales
(R_total1,M_total1,L_total1,T_central1,r_down1,P1,T1,L1,M1,rho1,M_frontera1)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X1,Y1,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=True,deltaR=0.1,deltaL=1,profundidad=10)
Tef1=5777*((6.957/array(R_total1))**2*(array(L_total1)/3.828))**(1/4)

#Estrella 2 
X2=0.75; Y2=0.22; Z2=1-X2-Y2 #Composición química
R_total=11.5; L_total=70.0; T_central=2; #Valores iniciales
(R_total2,M_total2,L_total2,T_central2,r_down2,P2,T2,L2,M2,rho2,M_frontera2)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X2,Y2,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=True,deltaR=0.1,deltaL=1,profundidad=10)
Tef2=5777*((6.957/array(R_total2))**2*(array(L_total2)/3.828))**(1/4)


#Para masas no muy altas al disminuir la metalicidad, disminuye la opacidad b-f entonces las zonas convectivas
#son menores, la estrella es más transparente y, por lo tanto, más calientes (en Tef), más pequeña y algo más luminosas

#La abundancia de Helio, Y, afecta cambiando la opacidad y el peso molecular medio. Al aumentar Y,
#disminuye la opacidad y aumenta el ritmo de generación de energía, haciendo a las estrellas más luminosas y más calientes.

A=[['Y','Z','T_{ef}','R/R_sun','L/L_sun']]
A+=[[Y1,Z1,Tef1,R_total1/6.957,L_total1/3.828]]
A+=[[Y2,Z2,Tef2,R_total2/6.957,L_total2/3.828]]
print(tabulate(A, headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.2f'))
# print(tabulate(A, headers='firstrow', tablefmt='latex',stralign='center',floatfmt='.2f'))
