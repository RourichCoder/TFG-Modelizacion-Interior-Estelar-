import RtotYLtotOptimosYRepresentacion
from RtotYLtotOptimosYRepresentacion import modelo_sin_optimizar_R_y_L, modelo_optimizado_para_X_Y_y_M
import SolucionRtotYLtotFijos
from SolucionRtotYLtotFijos import epsilon1_nu, ritmo_generacion_energia

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose, identity, array, log, log10, float64, isfinite
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

'''Resuelve la estructura de dos estrellas con la misma composición química, pero distintas masas

    Args:
        -Parametros característicos de cada estrella
        -Valores iniciales de cada estrella
        -Parámetros del mallado
    Return:
        Devuelve distintas gráficas comparando sus estructuras:
            -Densidad
            -Masa
            -Temperatura
            -Luminosidad
            -Generación de energía
'''
#PARÁMETROS DEL MODELO TFG--------------------------------------------------------------------------------------------------------------------
X=0.75;Y=0.20;Z=1-X-Y;mu=1/(2*X+0.75*Y+0.5*Z); #Mantenemos constante la composición química
capas=100;it1=10;aux1=10;iteraciones=4 #Parametros del mallado

#MODELO MASA alta (~11.3 M_sun)
M_total=22.5; print('Masa = 11.3M_sun') #Parámetro característico de la estrella
R_total=33; L_total=102000 ; T_central=2.9 #Valores iniciales
(R_total_alta,M_total_alta,L_total_alta,T_central_alta,r_down_alta,P_alta,T_alta,L_alta,M_alta,rho_alta,M_frontera_alta)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=True,representacion4=False,representacion5=True)

#MODELO MASA baja (~1.75 M_sun)
M_total=3.5; print('Masa = 1.75M_sun') #Parámetro característico de la estrella
R_total=9.5; L_total=22; T_central=1.5; #Valores iniciales
(R_total_baja,M_total_baja,L_total_baja,T_central_baja,r_down_baja,P_baja,T_baja,L_baja,M_baja,rho_baja,M_frontera_baja)=modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=True)


r_down_norm_alta=r_down_alta/R_total_alta
r_down_norm_baja=r_down_baja/R_total_baja
P_norm_alta=P_alta/amax(P_alta);T_norm_alta=T_alta/amax(T_alta); L_norm_alta=L_alta/amax(L_alta); M_norm_alta=M_alta/amax(M_alta);rho_norm_alta=rho_alta/amax(rho_alta)
P_norm_baja=P_baja/amax(P_baja);T_norm_baja=T_baja/amax(T_baja); L_norm_baja=L_baja/amax(L_baja); M_norm_baja=M_baja/amax(M_baja);rho_norm_baja=rho_baja/amax(rho_baja)

#%%
#DENSIDAD. Observamos que la densidad es mayor en las estrellas menos masivas
a=float64(10**7*array(rho_alta));b=float64(10**7*array(rho_baja))
rho_alta_log=log10(a);rho_baja_log=log10(b)
plt.figure()
plt.title('Densidad en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=11)
plt.plot(r_down_norm_alta,rho_alta_log,color='blue',label=r'$11.3M_\odot$')
plt.plot(r_down_norm_baja,rho_baja_log,color='red',label=r'$1.75M_\odot$')
plt.ylabel(r'log($\rho$)')
plt.xlabel(r'r/$R_\mathrm{total}$')
plt.legend()

#MASA. La masa está muy concentrada hacia el centro (cerca del 75% de masa en menos de 0.5r/R que en volumen es...). 
plt.figure()
plt.title('Masa relativa en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=11)
plt.plot(r_down_norm_alta,M_norm_alta,color='blue',label=r'$11.3M_\odot$')
plt.plot(r_down_norm_baja,M_norm_baja,color='red',label=r'$1.75M_\odot$')
plt.ylabel(r'$m/M_\mathrm{total}$')
plt.xlabel(r'$r/R_\mathrm{total}$')
plt.legend()

#TEMPERATURA. La temperatura aumenta rápidamente hacia el centro. La estrella más masiva tiene mayor T en cada punto
plt.figure()
plt.title('Temperatura en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=11)
plt.plot(r_down_norm_alta,T_alta,color='blue',label=r'$11.3M_\odot$')
plt.plot(r_down_norm_baja,T_baja,color='red',label=r'$1.75M_\odot$')
plt.ylabel(r'$T\quad 10^7 \mathrm{K}$')
plt.xlabel(r'$r/R_\mathrm{total}$')
plt.legend()

#LUMNINOSIDAD. Aproximadamente, para 1.75M_sun el 90% de la energía se genera en el 15% interior de la masa y para 11.3M_Sun el 90% de energía se genera en el 25% interior de la masa 
plt.figure()
plt.title('Luminosidad relativa en función de la masa relativa',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=11)
plt.plot(M_norm_alta,L_norm_alta,color='blue',label=r'$11.3M_\odot$')
plt.plot(M_norm_baja,L_norm_baja,color='red',label=r'$1.75M_\odot$')
plt.ylabel(r'$l/L_\mathrm{total}$')
plt.xlabel(r'$m/M_\mathrm{total}$')
plt.legend()

#GENERACIÓN DE ENERGÍA. La generación de energía es mayor en la estrella más masiva, también tiene generación en capas más externas que la estrella menos masivas porque sigue alcanzando la temperatura suficiente para la cadena pp.
epsilon_alta=zeros(len(T_alta));epsilon_baja=zeros(len(T_baja))
for i in range(len(T_alta)):
    if ritmo_generacion_energia('pp',T_alta[i]*10,X,Z)>=ritmo_generacion_energia('CNO',T_alta[i]*10,X,Z): #Domina el ciclo pp
        epsilon_alta[i]=ritmo_generacion_energia('pp',T_alta[i]*10,X,Z,rho=1) #T se introduce en unidades K
    else: #Domina el ciclo CNO
        epsilon_alta[i]=ritmo_generacion_energia('CNO',T_alta[i]*10,X,Z,rho=1)
for i in range(len(T_baja)):
    if ritmo_generacion_energia('pp',T_baja[i]*10,X,Z)>=ritmo_generacion_energia('CNO',T_baja[i]*10,X,Z): #Domina el ciclo pp
        epsilon_baja[i]=ritmo_generacion_energia('pp',T_baja[i]*10,X,Z,rho=1) #T se introduce en unidades K
    else: #Domina el ciclo CNO
        epsilon_baja[i]=ritmo_generacion_energia('CNO',T_baja[i]*10,X,Z,rho=1)
epsilon_alta_log=log10(epsilon_alta);epsilon_baja_log=log10(epsilon_baja)        

plt.figure()
plt.title('Generación de energía en función del radio relativo',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=11)
plt.plot(r_down_norm_alta,epsilon_alta_log,color='blue',label=r'$11.3M_\odot$')
plt.plot(r_down_norm_baja,epsilon_baja_log,color='red',label=r'$1.75M_\odot$')
plt.ylabel(r'$log(\varepsilon\,[\mathrm{erg\,g^{-1}\,s^{-1}}])$')
plt.xlabel(r'$r/R_\mathrm{total}$')
plt.legend()



