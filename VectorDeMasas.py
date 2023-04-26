import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose, identity, array, log10
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

'''Resuelve la estructura de 20 estrellas con la misma composición química, pero distintas masas.

    Args:
        -Parametros característicos de cada estrella
        -Valores iniciales de cada estrella
        -Parámetros del mallado
    Return:
        Devuelve distintas gráficas comparando sus estructuras:
            -Relación L_total-M_total
            -Relación R_total-M_total
            -Relación M_total-T_central
            -Relación M_total-Densidad_central
            -Relación Presión-Temperatura
            -Diagrama HR (Luminosidad-Temperatura efectiva)
'''

#VARIACIÓN DE LA MASA TOTAL-------------------------------------------------------------------------
X=0.75;Y=0.20;Z=1-X-Y;mu=1/(2*X+0.75*Y+0.5*Z) #Mantenemos constante la composición química
capas=100;it1=10;aux1=10;iteraciones=4 #Parametros del mallado

vector_Masa_total=[13,12.5,12,11.5,11,10.5,10,9.5,9,8.5,8,7.5,7,6.5,6,5.5,5,4.5,4,3.5]
vector_Valores_iniciales=[[23.3,6590,2.48],[20.8,4750,2.4],[20.5,3900,2.4],[20,3020,2.4],[19.8,2425,2.32],[19,1890,2.3],[18.5,1480,2.3],[17.5,1150,2.2],[17,840,2.2],[16,600,2.1],[15.5,460,2.1],[14.5,325,2.1],[14,230,2.1],[14,155,2.2],[14,95.0,2],[14,70.0,1.8],[12,40.0,1.5],[12, 40.0, 1.5],[10,30,1.5],[9.5, 22, 1.5]]

# iteraciones=4
# vector_Masa_total=[4.5,4,3.5]
# vector_Valores_iniciales=[[12, 40.0, 1.5],[10,30,1.5],[9.5, 22, 1.5]]

R_total=[None]*len(vector_Masa_total);M_total=[None]*len(vector_Masa_total);L_total=[None]*len(vector_Masa_total);T_central=[None]*len(vector_Masa_total);M_frontera=[None]*len(vector_Masa_total);
r_down=[None]*len(vector_Masa_total);P=[None]*len(vector_Masa_total);M=[None]*len(vector_Masa_total);L=[None]*len(vector_Masa_total);T=[None]*len(vector_Masa_total);rho=[None]*len(vector_Masa_total)

for i in range(len(vector_Masa_total)):
    print('Masa total =',vector_Masa_total[i])
    (R_total[i],M_total[i],L_total[i],T_central[i],r_down[i],P[i],T[i],L[i],M[i],rho[i],M_frontera[i])=modelo_optimizado_para_X_Y_y_M(vector_Masa_total[i],vector_Valores_iniciales[i][0],vector_Valores_iniciales[i][1],vector_Valores_iniciales[i][2],X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=False)

#%%  
#CÁLCULO MAGNITUDES--------------------------------------------------------------------------------------
Tef=5777*((6.957/array(R_total))**2*(array(L_total)/3.828))**(1/4) #Temperatura efectiva. Stefan-Boltzman en unidades solares
Tc=[];rhoc=[]
for i in range(len(T)):
    Tc+=[T[i][-1]]
for i in range(len(rho)):
    rhoc+=[rho[i][-1]]


#TABLA CON LOS RESULTADOS-----------------------------------------------------------------------------------
A=[['M/M_sun','log(L/L_sun)','log(T_ef)','R/R_sun','log(Tc)']]
for i in range(len(M_total)):
    A+=[[M_total[i]/1.989,log10(L_total[i]/3.828),log10(Tef[i]),R_total[i]/6.957,log10(Tc[i]*10**7)]]
print(tabulate(A, headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))


#REPRESENTACIÓN------------------------------------------------------------------------------------------
#Relación Luminosidad total - Masa total (proporcionalidad k=, ajustada a mano)
L_total_tabulada=10**array([3.43,3.2,2.99,2.89,2.77,2.57,2.48,2.19,1.86,1.8,1.58,1.49,1.38,1.23,1.13,1.09,1.05,1,0.96,0.92])
M_total_tabulada_M_sun=[7.3,6.1,5.4,5.1,4.7,4.3,3.92,3.38,2.75,2.68,2.18,2.05,1.98,1.86,1.93,1.88,1.83,1.77,1.81,1.75]
#NO logarítmica
# plt.figure()
# plt.title('Relación masa total - luminosidad total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
# plt.plot(M_total,L_total,color='blue',label='Resultado modelo')
# plt.plot(M_total,17*mu**4*array(M_total)**3,color='red',label='Relación homóloga') 
# plt.ylabel('L_total')
# plt.xlabel('M_total')
# plt.legend()

#Logarítmica
plt.figure()
plt.title('Relación masa total - luminosidad total (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(L_total)/3.828),color='blue',label='Resultado modelo')
plt.plot(log10(array(M_total)/1.989),log10((2*mu**4*array(M_total)**3)/3.828),color='red',label='Relación homóloga') 
plt.plot(log10(M_total_tabulada_M_sun),log10(L_total_tabulada/3.828),label='Datos tabulados')
plt.ylabel('log(L_total/L_sun)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Radio total - Masa total (falta añadir proporcionalidad relación homóloga)
R_total_tabulado_R_sun=[4.06,3.89,3.61,3.46,3.36,3.27,2.94,2.86,2.49,2.45,2.193,2.136,2.117,1.861,1.794,1.785,1.775,1.75,1.747,1.747]
#NO logarítmica
# plt.figure()
# plt.title('Relación masa total - radio total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
# plt.plot(M_total,R_total,color='blue',label='Resultado modelo')
# plt.plot(M_total,mu**0.67*array(M_total)**0.81,color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
# plt.plot(M_total,array(M_total)**0.43,color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
# plt.ylabel('R_total')
# plt.xlabel('M_total')
# plt.legend()

#Logarítmica
plt.figure()
plt.title('Relación masa total - radio total (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(R_total)/6.957),color='blue',label='Resultado modelo')
plt.plot(log10(array(M_total)/1.989),log10(4.3*(mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
plt.plot(log10(array(M_total)/1.989),log10(5.45*array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.plot(log10(M_total_tabulada_M_sun),log10(array(R_total_tabulado_R_sun)),label='Datos tabulados')
plt.ylabel('log(R_total/R_sun)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Masa total - Temperatura central (falta añadir las relaciones homólogas)
plt.figure()
plt.title('Relación masa total - temperatura central (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(Tc)*10**7),color='blue',label='Resultado modelo')
# plt.plot(logM,log((mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
# plt.plot(logM,log(array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('log(T_c)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Masa total - Densidad central
plt.figure()
plt.title('Relación masa total - densidad central (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(rhoc)*10**7),color='blue',label='Resultado modelo')
# plt.plot(logM,log((mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
# plt.plot(logM,log(array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('log(rho_c)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Diagrama HR (Luminosidad total - Temperatura efectiva) (Falta añadir relaciones homólogas)
Tef_tabulada=[20600,18500,17000,16400,15700,14500,14000,12300,10700,10400,9700,9300,8800,8600,8250,8100,7910,7760,7590,7400,7220,7020,6820,6750,6670,6550,6350,6280,6180,6050]
L_total_tabulada=10**array([3.43,3.2,2.99,2.89,2.77,2.57,2.48,2.19,1.86,1.8,1.58,1.49,1.38,1.23,1.13,1.09,1.05,1,0.96,0.92,0.86,0.79,0.71,0.67,0.62,0.56,0.43,0.39,0.29,0.22,])

#NO logarítmica
# plt.figure()
# plt.title('Diagrama HR',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
# plt.plot(Tef,array(L_total)/3.828, label='Datos del modelo')
# plt.plot(Tef_tabulada,L_total_tabulada,label='Datos tabulados, composición química solar')
# plt.ylabel('L_total/L_sun')
# plt.xlabel('T_ef')
# plt.gca().invert_xaxis() #El eje x en el diagrama HR está invertido
# plt.legend()

#Logarítmica
plt.figure()
plt.title('Diagrama HR (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(Tef),log10(array(L_total)/3.828),color='blue',label='Resultado modelo')
plt.plot(log10(Tef_tabulada),log10(L_total_tabulada),label='Datos tabulados, composición química solar')
plt.plot(log10(Tef),5.6*log10(Tef)-20.8,color='red',label='Observacionalmente: M baja, cadena p-p') #Hemos cogido la constante=-20.8
# plt.plot(log10(Tef),8.7*log10(Tef)-34,color='orange',label='M alta, ciclo CNO') #Hemos cogido la constante=-34.8 (no sale la misma pendiente porque ajusta bien para masa mayores de 10M_sun y lo máximo que estamos cogiendo es 6.5M_sun)
plt.ylabel('log(L_total/L_sun)')
plt.xlabel('log(T_ef)')
plt.gca().invert_xaxis() #El eje x en el diagrama HR está invertido
plt.legend()
