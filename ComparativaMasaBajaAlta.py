import RadioLuminosidadOptimosYRepresentacion
from RadioLuminosidadOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M
import SolucionRtotYLtotFijos
from SolucionRtotYLtotFijos import epsilon1_nu, ritmo_generacion_energia

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose, identity, array, log, log10, float64
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

#PARÁMETROS DEL MODELO TFG--------------------------------------------------------------------------------------------------------------------
X=0.75;Y=0.20;Z=1-X-Y; #Mantenemos constante la composición química
capas=100;it1=10;aux1=10;it2=6;aux2=8 #Parametros del mallado
mu=1/(2*X+0.75*Y+0.5*Z);

#plot1= Plot del error total en función de la temperatura central para un R y L totales fijos
#Representación1= Plot del mallado del error total en función de las luminosidad y radios totales 
#Representación2= Plot del mallado de la temperatura central óptima en función de las luminosidad y radios totales 
#Representación3= Tablas del modelo que optimiza el error
#Representación4= Plot de las magnitudes física en función del radio (un plot para cada magnitud)
#Representación5= Plot de todas las magnitudes físicas normalizadas (en un único plot)
#datos= Devuelve los valores R_total,M_total,L_total,T_central,r_down,M,L,T,rho

iteraciones=4

#MODELO MASA alta (11.3 M_sun)
M_total=22.5
R_total=33; L_total=102000 ; T_central=2.9
(R_total_alta,M_total_alta,L_total_alta,T_central_alta,r_down_alta,P_alta,T_alta,L_alta,M_alta,rho_alta)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=True,representacion4=False,representacion5=True)

# #MODELO MASA bastante baja 
# M_total=4.5
# R_total=12; L_total=40; T_central=1.5
# (R_total_p,M_total_p,L_total_p,T_central_p,r_down_p,M_p,L_p,T_p,rho_p)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=True,representacion4=False,representacion5=True)

#MODELO MASA baja  (1.75 M_sun)
M_total=3.5
R_total=9.5; L_total=22; T_central=1.5;  #VALORES INICIALES
(R_total_baja,M_total_baja,L_total_baja,T_central_baja,r_down_baja,P_baja,T_baja,L_baja,M_baja,rho_baja)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=True)

r_down_norm_alta=r_down_alta/R_total_alta
r_down_norm_baja=r_down_baja/R_total_baja
P_norm_alta=P_alta/amax(P_alta);T_norm_alta=T_alta/amax(T_alta); L_norm_alta=L_alta/amax(L_alta); M_norm_alta=M_alta/amax(M_alta);rho_norm_alta=rho_alta/amax(rho_alta)
P_norm_baja=P_baja/amax(P_baja);T_norm_baja=T_baja/amax(T_baja); L_norm_baja=L_baja/amax(L_baja); M_norm_baja=M_baja/amax(M_baja);rho_norm_baja=rho_baja/amax(rho_baja)



#DENSIDAD. Observamos que la densidad es mayor en las estrellas menos masivas
a=float64(10**7*array(rho_alta));b=float64(10**7*array(rho_baja))
rho_alta_log=log10(a);rho_baja_log=log10(b)
plt.figure()
plt.title('Densidad en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(r_down_norm_alta,rho_alta_log,color='blue',label='11.3M_sun')
plt.plot(r_down_norm_baja,rho_baja_log,color='red',label='1.75M_sun')
plt.ylabel('log(rho)')
plt.xlabel('r/R')
plt.legend()

#MASA. La masa está muy concentrada hacia el centro (cerca del 75% de masa en menos de 0.5r/R que en volumen es...). La estrella menos masiva tiene la masa más concentrada al centro
plt.figure()
plt.title('Masa relativa en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(r_down_norm_alta,M_norm_alta,color='blue',label='11.3M_sun')
plt.plot(r_down_norm_baja,M_norm_baja,color='red',label='1.75M_sun')
plt.ylabel('m/M')
plt.xlabel('r/R')
plt.legend()

#TEMPERATURA. La temperatura aumenta rápidamente hacia el centro. La estrella más masiva tiene mayor T en cada punto
plt.figure()
plt.title('Temperatura en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(r_down_norm_alta,T_alta,color='blue',label='11.3M_sun')
plt.plot(r_down_norm_baja,T_baja,color='red',label='1.75M_sun')
plt.ylabel('T')
plt.xlabel('r/R')
plt.legend()

#LUMNINOSIDAD. Para 1.75M_sun el 90% de la energía se genera en el 15% interior de la masa y para 11.3M_Sun el 90% de energía se genera en el 25% interior de la masa
plt.figure()
plt.title('Luminosidad relativa en función de la masa relativa',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(M_norm_alta,L_norm_alta,color='blue',label='11.3M_sun')
plt.plot(M_norm_baja,L_norm_baja,color='red',label='1.75M_sun')
plt.ylabel('l/L')
plt.xlabel('m/M')
plt.legend()

#GENERACIÓN DE ENERGÍA. La generación de energía está más concentrada en el centro en las estrellas más masivas (ciclo CNO vs cadena pp)
epsilon_alta=zeros(len(T_alta));epsilon_baja=zeros(len(T_baja))
for i in range(len(T_alta)):
    if ritmo_generacion_energia('pp',T_alta[i]*10,X,Z)>=ritmo_generacion_energia('CNO',T_alta[i]*10,X,Z): #Domina el ciclo pp
        epsilon_alta[i]=ritmo_generacion_energia('pp',T_alta[i]*10,X,Z,rho=1) #T se introduce en unidades 10**6K
    else: #Domina el ciclo CNO
        epsilon_alta[i]=ritmo_generacion_energia('CNO',T_alta[i]*10,X,Z,rho=1)
for i in range(len(T_baja)):
    if ritmo_generacion_energia('pp',T_baja[i]*10,X,Z)>=ritmo_generacion_energia('CNO',T_baja[i]*10,X,Z): #Domina el ciclo pp
        epsilon_baja[i]=ritmo_generacion_energia('pp',T_baja[i]*10,X,Z,rho=1) #T se introduce en unidades 10**6K
    else: #Domina el ciclo CNO
        epsilon_baja[i]=ritmo_generacion_energia('CNO',T_baja[i]*10,X,Z,rho=1)
epsilon_alta_log=log10(epsilon_alta);epsilon_baja_log=log10(epsilon_baja)
plt.figure()
plt.title('Generación de energía en función de la masa relativa',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(M_norm_alta,epsilon_alta_log,color='blue',label='11.3M_sun')
plt.plot(M_norm_baja,epsilon_baja_log,color='red',label='1.75M_sun')
plt.ylabel('log(epsilon)')
plt.xlabel('m/M')
plt.legend()


