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
            -
'''

#VARIACIÓN DE LA MASA TOTAL
X=0.75;Y=0.20;Z=1-X-Y;mu=1/(2*X+0.75*Y+0.5*Z) #Mantenemos constante la composición química
capas=100;it1=10;aux1=10;iteraciones=4 #Parametros del mallado

vector_Masa_total=[13,12.5,12,11.5,11,10.5,10,9.5,9,8.5,8,7.5,7,6.5,6,5.5,5,4.5,4,3.5]
vector_Valores_iniciales=[[23.3,6590,2.48],[20.8,4750,2.4],[20.5,3900,2.4],[20,3020,2.4],[19.8,2425,2.32],[19,1890,2.3],[18.5,1480,2.3],[17.5,1150,2.2],[17,840,2.2],[16,600,2.1],[15.5,460,2.1],[14.5,325,2.1],[14,230,2.1],[14,155,2.2],[14,95.0,2],[14,70.0,1.8],[12,40.0,1.5],[12, 40.0, 1.5],[10,30,1.5],[9.5, 22, 1.5]]

# iteraciones=1
# vector_Masa_total=[13,12.5,12]
# vector_Valores_iniciales=[[23.3,6590,2.48],[20.8,4750,2.4],[20.5,3900,2.4]]

R_total=[None]*len(vector_Masa_total);M_total=[None]*len(vector_Masa_total);L_total=[None]*len(vector_Masa_total);T_central=[None]*len(vector_Masa_total); 
r_down=[None]*len(vector_Masa_total);P=[None]*len(vector_Masa_total);M=[None]*len(vector_Masa_total);L=[None]*len(vector_Masa_total);T=[None]*len(vector_Masa_total);rho=[None]*len(vector_Masa_total)

for i in range(len(vector_Masa_total)):
    print(vector_Masa_total[i])
    # print(vector_Valores_iniciales[i][0],vector_Valores_iniciales[i][1],vector_Valores_iniciales[i][2])
    (R_total[i],M_total[i],L_total[i],T_central[i],r_down[i],P[i],T[i],L[i],M[i],rho[i])=modelo_optimizado_para_X_Y_y_M(vector_Masa_total[i],vector_Valores_iniciales[i][0],vector_Valores_iniciales[i][1],vector_Valores_iniciales[i][2],X,Y,capas,it1,aux1,iteraciones,representacion1=False,representacion2=False,representacion3=False,representacion4=False,representacion5=False)
    

#Relación Luminosidad total - Masa total (proporcionalidad k=18, ajustada a mano)
plt.figure()
plt.title('Relación masa total - luminosidad total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(M_total,L_total,color='blue',label='Resultado modelo')
plt.plot(M_total,18*mu**4*array(M_total)**3,color='red',label='Relación homóloga') 
plt.ylabel('L_total')
plt.xlabel('M_total')
plt.legend()

plt.figure()
plt.title('Relación masa total - luminosidad total (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(L_total)/3.828),color='blue',label='Resultado modelo')
plt.plot(log10(array(M_total)/1.989),log10((18*mu**4*array(M_total)**3)/3.828),color='red',label='Relación homóloga') 
plt.ylabel('log(L_total/L_sun)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Radio total - Masa total (falta añadir proporcionalidad relación homóloga)
plt.figure()
plt.title('Relación masa total - radio total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(M_total,R_total,color='blue',label='Resultado modelo')
plt.plot(M_total,mu**0.67*array(M_total)**0.81,color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
plt.plot(M_total,array(M_total)**0.43,color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('R_total')
plt.xlabel('M_total')
plt.legend()

plt.figure()
plt.title('Relación masa total - radio total (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(array(M_total)/1.989),log10(array(R_total)/6.957),color='blue',label='Resultado modelo')
plt.plot(log10(array(M_total)/1.989),log10((mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
plt.plot(log10(array(M_total)/1.989),log10(array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('log(R_total/R_sun)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Masa total - Temperatura central (falta añadir las relaciones homólogas)
plt.figure()
plt.title('Relación masa total - temperatura central (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
Tc=[]
for i in range(len(T)):
    Tc+=[T[i][-1]]
plt.plot(log10(array(M_total)/1.989),log10(array(Tc)*10**7),color='blue',label='Resultado modelo')
# plt.plot(logM,log((mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
# plt.plot(logM,log(array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('log(T_c)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Masa total - Densidad central
plt.figure()
plt.title('Relación masa total - densidad central (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
rhoc=[]
for i in range(len(rho)):
    rhoc+=[rho[i][-1]]
plt.plot(log10(array(M_total)/1.989),log10(array(rhoc)*10**7),color='blue',label='Resultado modelo')
# plt.plot(logM,log((mu**0.67*array(M_total)**0.81)/6.957),color='red',label='Relación homóloga (ciclo CNO)') #Ciclo CNO (v=18)
# plt.plot(logM,log(array(M_total)**0.43/6.957),color='orange',label='Relación homóloga (cadena pp') #Cadena pp (v=4)
plt.ylabel('log(rho_c)')
plt.xlabel('log(M_total/M_sun)')
plt.legend()


#Relación Presión - Temperatura (logarítmica) para varias masas
fig, ax = plt.subplots()
plt.title('Relación presión-temperatura (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
for i in range(len(vector_Masa_total)//4): #Representamos solo algunas masas
    ax.plot(log10(array(P[i+4])*10**15), log10(array(T[i+4]*10**7)), label=str(M_total[i]),linewidth=0.75)
plt.xlabel('log(P)')
plt.ylabel('log(T)')
plt.legend()


#Diagrama HR (Luminosidad total - Temperatura efectiva) (Falta añadir relaciones homólogas)
plt.figure()
plt.title('Diagrama HR',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
Tef=6.12060158*(array(L_total)**(1/4)*10**6.5)/(array(R_total)**(1/2)*10**4)
plt.plot(Tef,array(L_total)/3.828)
plt.ylabel('L_total/L_sun')
plt.xlabel('T_ef')
plt.gca().invert_xaxis() #El eje x en el diagrama HR está invertido
plt.legend()

plt.figure()
plt.title('Diagrama HR (logarítmica)',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
plt.plot(log10(Tef),log10(array(L_total)/3.828),color='blue',label='Resultado modelo')
# plt.plot(logTef,5.6*logTef,color='red',label='M baja, cadena p-p')
# plt.plot(logTef,8.7*logTef,color='orange',label='M alta, ciclo CNO')
plt.ylabel('log(L_total/L_sun)')
plt.xlabel('log(T_ef)')
plt.gca().invert_xaxis() #El eje x en el diagrama HR está invertido
plt.legend()




