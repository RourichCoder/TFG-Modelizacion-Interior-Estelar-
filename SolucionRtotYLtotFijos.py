from numba import njit
from numpy import zeros, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt


def interpolacion_lineal(d,x): 
    """
    Interpolación lineal para un valor entre dos puntos P_1 y P_2
    
    Args:
        d: Matriz 2x2 con los dos pares de puntos P_1=(x_1=d[0][0],y_1=d[0][1]) y P_2(d[1][0],d[1][1])
        x: Coordenada x del punto que queremos interpolar. Debe verificar d[0][0]<x<d[1][0]
    
    Returns:
        valor_inteporlado: valor interpolado de P_1 y P_2 en x
    """
    valor_interpolado=d[0][1]+(x-d[0][0])*((d[1][1]-d[0][1])/(d[1][0]-d[0][0]))  
    return valor_interpolado

@njit
def epsilon1_nu(tipo_ciclo,T): 
    """
    Cálculo de los parámetros epsilon1 y nu en función del ciclo usado y la temperatura.
    
    Args:
        tipo_ciclo: Dos posibles valores: 'pp' o 'CNO' en función del mecanismo de generación de energía. Para cualquier otro valor devuelve los parámetros iguales a 0
        T: Temperatura en unidades 10**6K
    
    Returns:
        epsilon1: Constante para el cálculo de la generación de la energía.  
        nu: Número entero necesario para el cálculo de la generación de energía 
    """
    pp=[[4,6,-6.84,6.0],[6,9.5,-6.04,5],[9.5,12,-5.56,4.5],[12,16.5,-5.02,4],[16.5,24,-4.4,3.5]] #Intervalo (T1-T2)/10^6, log_10epsilon1, nu
    cno=[[12,16,-22.2,20],[16,22.5,-19.8,18],[22.5,27.5,-17.1,16],[27.5,36,-15.6,15],[36,50,-12.5,13]]
    epsilon1=0;nu=0
    if tipo_ciclo=='pp':
        for i in range(len(pp)):
            if pp[i][0]<=T<pp[i][1]:
                epsilon1=10**pp[i][2]       
                nu=pp[i][3]  
    elif tipo_ciclo=='CNO':
        for i in range(len(pp)):
            if cno[i][0]<=T<cno[i][1]:
                epsilon1=10**cno[i][2]
                nu=cno[i][3]
    return epsilon1,nu

@njit
def ritmo_generacion_energia(tipo_ciclo,T,X,Z,rho=1):
    """
    Devuelve el ritmo de generacion de E que depende del mecanismo de fusion del hidrogeno, la composicion química y la temperatura. 
    
    Args:
        tipo_ciclo: Dos posibles valores: 'pp' o 'CNO' en función del mecanismo de generación de energía. Para cualquier otro valor devuelve los parámetros iguales a 0
        T: Temperatura en K
        X: Fracción en masa de hidrógeno
        Z: Metalicidad. Fracción en masa de todo aquello que no es ni hidrógeno ni helio (Z=1-X-Y)
        rho: Densidad. Por defecto es 1
    
    Returns:
       Valor de epsilón, ritmo de generación de energía, en erg g^1 s^-1
    """
    if T<4: #Si la T esta fuera de los intervalos asociados a los ciclos CNO y pp entonces la generacion de E es nula
        return 0 
    else:
        if tipo_ciclo=='pp':
            X1=X;X2=X
        elif tipo_ciclo=='CNO':
            X1=X;X2=1/3*Z
    (epsilon1,nu)=epsilon1_nu(tipo_ciclo,T)
    return epsilon1*X1*X2*rho*T**nu 

@njit
def error_relativo_absoluto(calculado,estimado,E_relativo_maximo=0.0001): 
    """
    Calcula si el error relativo absoluto de dos magnitudes es menor que una cota.
    
    Args:
        calculado: Primera magnitud
        estimado: Segundo magnitud 
        E_relativo_maximo: Cota. Por defecto 0.0001
    
    Returns:
       True si el error relativo absoluto es menor que E_relativo_maximo y False en caso contrario.
    """
    if abs(calculado-estimado)/calculado<=E_relativo_maximo:
        return True
    else:   
        return False

@njit
def ecuacion_19_21(r,P,T,L,M,X,Z,mu):
    """
    Ecuaciones 19 y 21. Calcula f_P y f_T cuando hay transporte radiativo en la capa i.
    
    Args:
        r: Valor de r en la capa i
        P: Valor de P en la capa i
        T: Valor de T en la capa i
        L: Valor de L en la capa i 
        M: Valor de M en la capa i
        X: Fracción en masa de hidrógeno
        Z: Metalicidad. 
        mu: Peso molecular medio
    
    Returns:
        f_P: Valor de dP/dr en la capa i
        f_T: Valor de dT/dr en la capa i
    """
    C_P=8.084*mu
    C_T=0.01679*Z*(1+X)*mu**2
    f_P=-C_P*P*M/(T*r**2)
    f_T=-C_T*P**2*L/(T**8.5*r**2)
    return f_P,f_T

@njit
def ecuacion_35_36(r,M_total, R_total,L_total,X,Z,mu):
    """
    Ecuaciones 35 y 36. Calcula la T y P en la capa i. Solo es válido para capas exteriores y asume transporte radiativo.
    
    Args:
        r: Valor de r en la capa i
        M_total: Masa total de la estrella
        R_total: Radio total de la estrella
        L_total: Luminosidad total de la estrella
        X: Fracción en masa de hidrógeno
        Z: Metalicidad. 
        mu: Peso molecular medio
    
    Returns:
        P: Valor de P en la capa i
        T: Valor de T en la capa i
    """
    A_1=1.9022*mu*M_total
    A_2=10.645*(M_total/(L_total*mu*Z*(1+X)))**(0.5)
    T=A_1*(1/r-1/R_total)
    P=A_2*T**4.25
    return P,T

@njit
def ecuacion_43_44_45_46(mu,T_central,r,K,X,Z,j,M,L,T,P): 
    """
    Ecuaciones 43, 44, 45 y 46. Calcula la M, L, T y P en la capa j+1. Válido para las capas centrales asumiendo tranporte convectivo.

    Args:
        mu: Peso molecular medio
        T_central: Temperatura en el centro de la estrellas [10**6K]
        r: Vector equiespaciado de la superficie al centro
        K: Constante del polítropo
        X: Fracción en masa de hidrógeno
        Z: Metalicidad
        j: Capa-1 en la que queremos calcular las magnitudes
        M: Valor de M en la capa j
        L: Valor de L en la capa j
        T: Valor de T en la capa j
        P: Valor de P en la capa j
    
    Returns:
        M: Valor de M en la capa j+1
        L: Valor de L en la capa j+1
        T: Valor de T en la capa j+1
        P: Valor de P en la capa j+1
    """
    #Cálculo  de M,L,T,P asumiendo transporte convectivo
    #Asumimos transporte convectivo
    X1=X;X2=1/3*Z
    M[j]=0.005077*mu*K*T_central**1.5*r[j]**3
    T[j]=T_central-0.008207*mu**2*K*T_central**1.5*r[j]**2
    (epsilon1,nu)=epsilon1_nu('CNO',T[j]*10)
    L[j]=0.006150*epsilon1*X1*X2*10**nu*mu**2*K**2*T_central**(3+nu)*r[j]**3
    P[j]=K*T[j]**2.5
    return M,L,T,P

@njit
def delta1(h,f,i): 
    """
    Devuelve 1^\Delta_i de la magnitud f (diferencia de primer orden de la derivada primera) en la capa i

    Args:
        h: Paso de la discretización 
        f: Magnitud de la que queremos calcular la diferencia
        i: Capa en la que realizamos el cálculo
    
    Returns:
        Diferencia de primer orden en la capa i de la magnitud f
    """
    #
    return h*(f[i]-f[i-1])

@njit
def delta1y2(h,f,i): 
    """
    Devuelve 1^\Delta_i y 2^\Delta_i  de la magnitud f (diferencias de primer y segundo orden, respectivamente, de la derivada primera) en la capa i

    Args:
        h: Paso de la discretización 
        f: Magnitud de la que queremos calcular la diferencia
        i: Capa en la que realizamos el cálculo
    
    Returns:
        delta1_i: Diferencia de primer orden en la capa i de la magnitud f
        delta2_i: Diferencia de segundo orden en la capa i de la magnitud f
    """
    delta1_i=delta1(h,f,i)
    delta1_i1=delta1(h,f,i-1)
    delta2_i=delta1_i-delta1_i1
    return delta1_i,delta2_i

@njit
def paso2(h,P,f_P,T,f_T,i):
    """
    Paso 2 de los algoritmos. Calcula la presión y temperatura estimada en la capa i+1

    Args:
        h: Paso de la discretización
        P: Valor de P en la capa j
        f_P: Vector con el valor de dP/dr a lo largo de la discretización
        T: Valor de T en la capa j
        f_T: Vector con el valor de dT/dr a lo largo de la discretización
        i: Capa-1 en la que queremos calcular las magnitudes estimadas
        
    Returns:
        P_i1_est: Presión  estimada en la capa i+1
        T_i1_est: Temperatura estimada en la capa i+1
    """
    (delta1_i_P,delta2_i_P)=delta1y2(h,f_P,i)
    P_i1_est=P[i]+h*f_P[i]+1/2*delta1_i_P+5/12*delta2_i_P
    delta1_i_T=delta1(h,f_T,i)
    T_i1_est=T[i]+h*f_T[i]+1/2*delta1_i_T
    return P_i1_est,T_i1_est

@njit
def paso2bis(K,h,T,f_T,i):
    """
    Paso 2bis de los algoritmos. Calcula la temperatura estimada en la capa i+1

    Args:
        h: Paso de la discretización
        K: Constante del polítropo       
        T: Valor de T en la capa j
        f_T: Vector con el valor de dT/dr a lo largo de la discretización
        i: Capa-1 en la que queremos calcular las magnitudes estimadas

    Returns:
        T_i1_est: Temperatura estimada en la capa i+1
    """
    delta1_i=delta1(h,f_T,i)
    T_i1_est=T[i]+h*f_T[i]+1/2*delta1_i
    return T_i1_est

@njit
def paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M):
    """
    Paso 3 de los algoritmos. Calcula la "masa calculada" y f_M en la capa i+1.

    Args:
        mu: Peso molecular medio
        T_i1_est: Tempratura estimada en la capa i+1
        P_i1_est: Presión estimada en la capa i+1
        r: Vector equiespaciado de la superficie al centro
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        f_M: Vector con el valor de dM/dr a lo largo de la discretización
        h: Paso de la discretización
        T: Valor de M en la capa i

    Returns:
        M_i1_cal: Masa calculada en la capa i+1
        f_M: Valor de dM/dr en la capa i+1
    """
    f_M[i+1]=0.01523*mu*P_i1_est*r[i+1]**2/T_i1_est
    delta1_i1_M=delta1(h,f_M,i+1)
    M_i1_cal=M[i]+h*f_M[i+1]-1/2*delta1_i1_M
    return M_i1_cal,f_M

@njit
def paso4(h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est):
    """
    Paso 4 de los algoritmos. Calcula la "presión calculada" y f_P en la capa i+1.

    Args:
        h: Paso de la discretización
        mu: Peso molecular medio
        M_i1_est: Masa estimada en la capa i+1
        r: Vector equiespaciado de la superficie al centro
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        P: Vector con la presión a lo largo de la discretización

        P_i1_est: Presión estimada en la capa i+1
        f_P: Vector con el valor de dP/dr a lo largo de la discretización
        T_i1_est: Tempratura estimada en la capa i+1
        
    Returns:
        P_i1_cal: Presión calculada en la capa i+1
        f_P: Valor de dP/dr en la capa i+1
    """
    f_P[i+1]=-8.084*mu*P_i1_est*M_i1_cal/(T_i1_est*r[i+1]**2)
    delta1_i1_P=delta1(h,f_P,i+1)
    P_i1_cal=P[i]+h*f_P[i+1]-1/2*delta1_i1_P
    return P_i1_cal,f_P

@njit
def paso5(P_i1_cal,h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est,f_M,M):
    """
    Paso 5 de los algoritmos. Realiza el "loop 3" (repetir paso 3 y paso 4) hasta que la presión calculada y estimada en la capa i+1 difieran menos del error relativo absoluto maximo.

    Args:
        P_i1_cal: Presión calculada en la capa i+1
        h: Paso de la discretización
        mu: Peso molecular medio
        M_i1_cal: Masa calculada en la capa i+1
        r: Vector equiespaciado de la superficie al centro
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        P: Vector con la presión a lo largo de la discretización
        P_i1_est: Presión estimada en la capa i+1
        f_P: Vector con el valor de dP/dr a lo largo de la discretización
        T_i1_est: Tempratura estimada en la capa i+1
        f_M: Vector con el valor de dM/dr a lo largo de la discretización
        M: Vector con la masa a lo largo de la discretización

    Returns:
        P_i1_cal: Presión calculada en la capa i+1
        f_P: Valor de dP/dr en la capa i+1
    """
    while not error_relativo_absoluto(P_i1_cal,P_i1_est):
        P_i1_est=P_i1_cal
        (M_i1_cal,f_M)=paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M)
        (P_i1_cal,f_P)=paso4(h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est)
    return P_i1_cal,f_P

@njit
def paso6(T_i1_est,P_i1_cal,L,f_L,h,i,r,mu,X,Z):
    """
    Paso 6 de los algoritmos. Calcula la "luminosidad calculada" y f_L en la capa i+1

    Args:
        T_i1_est: Tempratura estimada en la capa i+1
        P_i1_cal: Presión calculada en la capa i+1
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización
        h: Paso de la discretización
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        r: Vector equiespaciado de la superficie al centro
        mu: Peso molecular medio
        X: Fracción en masa de hidrógeno
        Z: Metalicidad
        
    Returns:
        L_i1_cal: Luminosidad calculada en la capa i+1
        f_L: Valor de dL/dr en la capa i+1
    """
    #Para hallar epsilon1,mu,nu,X1 y X2 tenemos que estudiar el tipo de ciclo. Para ello simplemente vemos cual genera más energía.
    #En este caso podemos asegurar que hay generación de energía pues T>4x10**6
    if ritmo_generacion_energia('pp',T_i1_est*10,X,Z)>=ritmo_generacion_energia('CNO',T_i1_est*10,X,Z): #Domina el ciclo pp
        (epsilon1,nu)=epsilon1_nu('pp',T_i1_est*10) #T se introduce en unidades 10**6
        X1=X;X2=X
    else: #Domina el ciclo CNO
        (epsilon1,nu)=epsilon1_nu('CNO',T_i1_est*10) #T se introduce en unidades 10**6
        X1=X;X2=1/3*Z
    f_L[i+1]=0.01845*epsilon1*X1*X2*10**nu*mu**2*P_i1_cal**2*T_i1_est**(nu-2)*r[i+1]**2
    (delta1_i1_L,delta2_i1_L)=delta1y2(h,f_L,i+1)
    L_i1_cal=L[i]+h*f_L[i+1]-1/2*delta1_i1_L-1/12*delta2_i1_L
    return L_i1_cal,f_L

@njit
def paso7(h,X,Z,mu,L_i1_cal,i,P_i1_cal,T_i1_est,r,T,f_T):
    """
    Paso 7 de los algoritmos. Calcula la "temperatura calculada" y f_T en la capa i+1

    Args:
        h: Paso de la discretización
        X: Fracción en masa de hidrógeno
        Z: Metalicidad
        mu: Peso molecular medio
        L_i1_cal: Luminosidad calculada en la capa i+1
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        P_i1_cal: Presión calculada en la capa i+1
        T_i1_est: Tempratura estimada en la capa i+1
        r: Vector equiespaciado de la superficie al centro
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        
    Returns:
        T_i1_cal: Temperatura calculada en la capa i+1
        f_T: Valor de dT/dr en la capa i+1
    """
    f_T[i+1]=-0.01679*Z*(1+X)*mu**2*P_i1_cal**2*L_i1_cal/(T_i1_est**8.5*r[i+1]**2)
    delta1_i1_T=delta1(h,f_T,i+1)
    T_i1_cal=T[i]+h*f_T[i+1]-1/2*delta1_i1_T
    return T_i1_cal,f_T

@njit
def paso7bis(r,h,mu,i,T,T_i1_est,f_T,M_i1_cal):
    """
    Paso 7bis de los algoritmos. Calcula la "temperatura calculada" y f_T en la capa i+1. 

    Args:
        r: Vector equiespaciado de la superficie al centro
        h: Paso de la discretización
        mu: Peso molecular medio
        i: Capa-1 en la que queremos calcular las magnitudes calculadas       
        T: Vector con la temperatura a lo largo de la discretización
        T_i1_est: Temperatura estimada en la capa i+1
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        M_i1_cal: Masa calculada en la capa i+1
             
    Returns:
        T_i1_cal: Temperatura calculada en la capa i+1
        f_T: Valor de dT/dr en la capa i+1
    """
    if r[i+1]==0: #En el centro
        T_i1_cal=T_i1_est
        f_T[i+1]=0
    else:
        f_T[i+1]=-3.234*mu*M_i1_cal/r[i+1]**2
        delta1_i1=delta1(h,f_T,i+1)
        T_i1_cal=T[i]+h*f_T[i+1]-1/2*delta1_i1
    return T_i1_cal, f_T

@njit
def paso8(T_i1_cal,P_i1_cal,h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est,Z,T,f_T,M,f_M,L,f_L,X):
    """
    Paso 8 de los algoritmos. Realiza el "loop 2" (repetir el "loop 3" y los pasos 6 y 7) hasta que la temperatura calculada y estimada en la capa i+1 difieran menos del error relativo absoluto maximo.

    Args:
        T_i1_cal: Temperatura calculada en la capa i+1
        P_i1_cal: Presión calculada en la capa i+1
        h: Paso de la discretización
        mu: Peso molecular medio
        M_i1_cal: Masa calculada en la capa i+1
        r: Vector equiespaciado de la superficie al centro
        i: Capa-1 en la que queremos calcular las magnitudes
        P: Vector con la presión a lo largo de la discretización
        P_i1_est: Presión estimada en la capa i+1
        f_P: Vector con el valor de dP/dr a lo largo de la discretización    
        T_i1_est: Temperatura estimada en la capa i+1
        Z: Metalicidad
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        M: Vector con la masa a lo largo de la discretización
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización    
        X: Fracción en masa de hidrógeno
             
    Returns:
        T_i1_cal: Temperatura calculada en la capa i+1
        f_T: Valor de dT/dr en la capa i+1
    """
    while not error_relativo_absoluto(T_i1_cal, T_i1_est): #Repetimos las fases 3, 4, 5, 6 y 7
        T_i1_est=T_i1_cal
        (M_i1_cal,f_M)=paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M)
        (P_i1_cal,f_P)=paso4(h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est)
        (P_i1_cal,f_P)=paso5(P_i1_cal,h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est,f_M,M)
        (L_i1_cal,f_L)=paso6(T_i1_est,P_i1_cal,L,f_L,h,i,r,mu,X,Z)
        (T_i1_cal,f_T)=paso7(h,X,Z,mu,L_i1_cal,i,P_i1_cal,T_i1_est,r,T,f_T)
    return T_i1_cal,f_T

def paso8bis(K,T_i1_est,mu,r,i,f_M,h,M,T,f_T,T_i1_cal):
    """
    Paso 8bis de los algoritmos. Realiza el "loop 2" (repetir "polítropo" y los pasos 3 y 7bis) hasta que la temperatura calculada y estimada en la capa i+1 difieran menos del error relativo absoluto maximo.

    Args:
        K: Constante del polítropo
        T_i1_est: Temperatura estimada en la capa i+1
        mu: Peso molecular medio
        r: Vector equiespaciado de la superficie al centro
        i: Capa-1 en la que queremos calcular las magnitudes ca
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        h: Paso de la discretización
        M: Vector con la masa a lo largo de la discretización
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        T_i1_cal: Temperatura calculada en la capa i+1
             
    Returns:
        T_i1_cal: Temperatura calculada en la capa i+1
        f_T: Valor de dT/dr en la capa i+1
    """
    while not error_relativo_absoluto(T_i1_cal,T_i1_est): #Repetimos polítropo y las fases 3 y 7bis
        T_i1_est=T_i1_cal
        P_i1_est=politropo(K,T_i1_est)
        (M_i1_cal,f_M)=paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M)
        (T_i1_cal,f_T)=paso7bis(r,h,mu,i,T,T_i1_est,f_T,M_i1_cal)
    return T_i1_cal,f_T

@njit
def paso9(T_i1_cal,P_i1_cal,f_P,f_T,i):
    """
    Paso 9 de los algoritmos. Calcula el valor de n (parámetro esencial para ver cuando domina el tranporte convectivo)

    Args:
        T_i1_cal: Temperatura calculada en la capa i+1
        P_i1_cal: Presión calculada en la capa i+1
        f_P: Vector con el valor de dP/dr a lo largo de la discretización    
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        i: Capa-1 en la que queremos calcular las magnitudes ca
             
    Returns:
        n: Parámetro n
    """
    n=(T_i1_cal/P_i1_cal)*(f_P[i+1]/f_T[i+1])-1
    return n

def paso10(n,i):
    """
    Paso 10 de los algoritmos. Comprueba si el valor de n+1 es menor o igual que 2.5. 
    En caso afirmativo, sigue siendo válido el transporte radiativo y los resultados de la capa i+1
    son válidos. 

    Args:
        n: Vector con el parámetro n  a lo largo de la parametrización
        i: Capa-1 en la que queremos calcular las magnitudes ca
             
    Returns:
        Fase_A1: Variable booleana. True indica que continua la fase radiativa y los valores hallados en la capa i+1 son válidos. False indica que el transporte es convectivo y desecha los cálculos en la fase i+1.
    """
    if n[i+1]+1<=2.5:
        Fase_A1=False #Empezamos fase convectiva
    else:   
        Fase_A1=True #Continuamos la fase radiativa
    return Fase_A1

@njit
def politropo(K,T_i1_est):
    """
    Paso "politropo" de los algoritmos. Calcula la "presión calculada" en la capa i+1 usando la expresión de un polítropo de constante n=3

    Args:
        K: Constante del polítropo
        T_i1_est: Temperatura estimada en la capa i+1
             
    Returns:
        P_i1_est: Presión estimada en la capa i+1
    """
    P_i1_est=K*T_i1_est**2.5
    return P_i1_est

@njit
def primeras_3_capas_externas(r_down,P,f_P,T,f_T,L,f_L,M,f_M,L_total,M_total,R_total,X,Z,mu,h_down):
    """
    Calcula los valores de la presión, temperatura, luminosidad y masa y d/dr de todas de ellas en las 3 primeras capas de la discretización empezando en la superficie.

    Args:
        r_down: Vector equiespaciado de la superficie al centro
        P: Vector con la presión a lo largo de la discretización
        f_P: Vector con el valor de dP/dr a lo largo de la discretización    
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización    
        M: Vector con la masa a lo largo de la discretización
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        L_total: Luminosidad total
        M_total: Masa total
        R_total: Radio total
        X: Fracción en masa de hidrógeno
        Z: Metalicidad
        mu: Peso molecular medio
        h_down: Paso de la discretización
             
    Returns:
        P: Vector con la presión en las 3 primeras capas
        f_P: Vector con el valor de dP/dr en las 3 primeras capas
        T: Vector con la temperatura en las 3 primeras capas
        f_T: Vector con el valor de dT/dr en las 3 primeras capas
        L: Vector con la luminosidad en las 3 primeras capas
        f_L: Vector con el valor de dL/dr en las 3 primeras capas
        M: Vector con la masa en las 3 primeras capas
        f_M: Vector con el valor de dM/dr en las 3 primeras capas
    """
    for i in range(0,3): 
        (P[i],T[i])=ecuacion_35_36(r_down[i],M_total,R_total,L_total,X,Z,mu)
        L[i]=L_total; M[i]=M_total; 
        f_L[i]=0;f_M[i]=0 #La M y L permanecen constantes
        (f_P[i],f_T[i])=ecuacion_19_21(r_down[i],P[i],T[i],L[i],M[i],X,Z,mu) 
    return P,f_P,T,f_T,L,f_L,M,f_M

def fase_radiativa_A1(h,P,f_P,T,f_T,i,mu,r,M,f_M,L,f_L,X,Z,E,n):
    """
    Calcula los valores de la presión, temperatura, luminosidad y masa y d/dr de todas de ellas 
    en las capas de la discretización con transporte radiativa. Para ello usa el algoritmo A.1.

    Args:
        h: Paso de la discretización
        P: Vector con la presión a lo largo de la discretización
        f_P: Vector con el valor de dP/dr a lo largo de la discretización    
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización
        i: Capa en la que estamos haciendo los cálculos
        mu: Peso molecular medio
        r: Vector equiespaciado de la superficie al centro
        M: Vector con la masa a lo largo de la discretización
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización    
        X: Fracción en masa de hidrógeno
        Z: Metalicidad
        E: Lista que indica el tipo de ciclo usado (pp o CNO) para la generación de energía en cada capa de la discretización. En caso de no haber generación de energía devuelve '---'
        n: Vector con el parámetro n  a lo largo de la parametrización
             
    Returns:
        P: Vector con la presión en las 3 primeras capas
        f_P: Vector con el valor de dP/dr en las 3 primeras capas
        T: Vector con la temperatura en las 3 primeras capas
        f_T: Vector con el valor de dT/dr en las 3 primeras capas
        L: Vector con la luminosidad en las 3 primeras capas
        f_L: Vector con el valor de dL/dr en las 3 primeras capas
        M: Vector con la masa en las 3 primeras capas
        f_M: Vector con el valor de dM/dr en las 3 primeras capas
        n: Vector con el parámetro n  a lo largo de la parametrización
        E: Lista que indica el tipo de ciclo usado (pp o CNO) para la generación de energía en cada capa de la discretización. En caso de no haber generación de energía devuelve '---'
        Fase_A1: Variable booleana. True indica que continua la fase radiativa y los valores hallados en la capa i+1 son válidos. False indica que el transporte es convectivo y desecha los cálculos en la fase i+1.
        K: Constante politrópica
    """
    (P_i1_est,T_i1_est)=paso2(h,P,f_P,T,f_T,i)
    (M_i1_cal,f_M)=paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M)
    (P_i1_cal,f_P)=paso4(h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est)
    (P_i1_cal,f_P)=paso5(P_i1_cal,h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est,f_M,M)
    (L_i1_cal,f_L)=paso6(T_i1_est,P_i1_cal,L,f_L,h,i,r,mu,X,Z)
    (T_i1_cal,f_T)=paso7(h,X,Z,mu,L_i1_cal,i,P_i1_cal,T_i1_est,r,T,f_T)
    (T_i1_cal,f_T)=paso8(T_i1_cal,P_i1_cal,h,mu,M_i1_cal,r,i,P,P_i1_est,f_P,T_i1_est,Z,T,f_T,M,f_M,L,f_L,X)
    n[i+1]=paso9(T_i1_cal,P_i1_cal,f_P,f_T,i)
    fase_A1=paso10(n,i)
    if fase_A1==True: #El transporte sigue siendo radiativo y los cálculos son válidos
        P[i+1]=P_i1_cal
        T[i+1]=T_i1_cal
        M[i+1]=M_i1_cal
        L[i+1]=L_i1_cal
        
        #Veamos cual es el ciclo de fusión de H predominante en la capa i+1
        if ritmo_generacion_energia('pp',T[i+1]*10,X,Z)==0:
            E+=['---']
        else:
            if ritmo_generacion_energia('pp',T[i+1]*10,X,Z)>=ritmo_generacion_energia('CNO',T[i+1]*10,X,Z): #Domina el ciclo pp
                E+=['PP ']
            else: #Domina el ciclo CNO
                E+=['CNO']
    #En caso contrario, fase_A1==False, el tranporte pasa a ser convectivo y los cálculos en la última capa no son válidos
    K=P_i1_cal/T_i1_cal**2.5 #Constante del polítropo que usaremos en la fase A.2 suponiendola constante
    return P,f_P,T,f_T,L,f_L,M,f_M,n,E,fase_A1,K

def fase_convectiva_A2(K,h,i,r,mu,T,f_T,M,f_M,P,f_P,L,f_L,Z,X,E):
    """
    Calcula los valores de la presión, temperatura, luminosidad y masa y d/dr de todas de ellas 
    en las capas de la discretización con transporte convectivo. Para ello usa el algoritmo A.2.

    Args:
        K: Constante politrópica
        h: Paso de la discretización
        i: Capa en la que estamos haciendo los cálculos
        r: Vector equiespaciado de la superficie al centro
        mu: Peso molecular medio
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización
        M: Vector con la masa a lo largo de la discretización
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        P: Vector con la presión a lo largo de la discretización
        f_P: Vector con el valor de dP/dr a lo largo de la discretización    
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización    
        Z: Metalicidad
        X: Fracción en masa de hidrógeno
        E: Lista que indica el tipo de ciclo usado (pp o CNO) para la generación de energía en cada capa de la discretización. En caso de no haber generación de energía devuelve '---'
             
    Returns:
        T: Vector con la temperatura en las 3 primeras capas
        f_T: Vector con el valor de dT/dr en las 3 primeras capas
        P: Vector con la presión en las 3 primeras capas
        f_P: Vector con el valor de dP/dr en las 3 primeras capas
        M: Vector con la masa en las 3 primeras capas
        f_M: Vector con el valor de dM/dr en las 3 primeras capas            
        L: Vector con la luminosidad en las 3 primeras capas
        f_L: Vector con el valor de dL/dr en las 3 primeras capas
        E: Lista que indica el tipo de ciclo usado (pp o CNO) para la generación de energía en cada capa de la discretización. En caso de no haber generación de energía devuelve '---'
        Fase_A2: Variable booleana. True indica que continua la fase convectiva y los valores hallados en la capa i+1 son válidos. False indica que hemos llegado al centro de la estrella.
    """
    T_i1_est=paso2bis(K,h,T,f_T,i)
    P_i1_est=politropo(K,T_i1_est)
    (M_i1_cal,f_M)=paso3(mu,T_i1_est,P_i1_est,r,i,f_M,h,M)
    (T_i1_cal,f_T)=paso7bis(r,h,mu,i,T,T_i1_est,f_T,M_i1_cal)
    (T_i1_cal,f_T)=paso8bis(K,T_i1_est,mu,r,i,f_M,h,M,T,f_T,T_i1_cal)
    P_i1_cal=politropo(K,T_i1_cal)
    (L_i1_cal,f_L)=paso6(T_i1_cal,P_i1_cal,L,f_L,h,i,r,mu,X,Z)
    if r[i+1]>0: #Todavía no hemos llegado al centro
        fase_A2=True 
        T[i+1]=T_i1_cal
        P[i+1]=P_i1_cal
        M[i+1]=M_i1_cal
        L[i+1]=L_i1_cal
    else: #r[i+1]==0 valores en el centro
        fase_A2=False 
        T[i+1]=T_i1_cal
        P[i+1]=P_i1_cal
        M[i+1]=M_i1_cal
        L[i+1]=L_i1_cal
    #Veamos cual es el ciclo de fusión de H predominante en la capa i+1
    if ritmo_generacion_energia('pp',T[i+1]*10,X,Z)>=ritmo_generacion_energia('CNO',T[i+1]*10,X,Z): #Domina el ciclo pp
        E+=['PP ']
    else: #Domina el ciclo CNO
        E+=['CNO']
    return T,f_T,P,f_P,M,f_M,L,f_L,E,fase_A2

def valores_en_la_frontera_down(n,r,P,T,L,M,i_frontera): 
    """
    Calcula los valores interpolados de la presión, temperatura, luminosidad y masa en el radio donde
    se encuentra la frontera entre el núcleo y envoltura cuando integramos desde la superficie.

    Args:
        n: Vector con el parámetro n  a lo largo de la parametrización
        r: Vector equiespaciado de la superficie al centro
        P: Vector con la presión a lo largo de la discretización
        T: Vector con la temperatura a lo largo de la discretización
        L: Vector con la luminosidad a lo largo de la discretización
        M: Vector con la masa a lo largo de la discretización
        i: Capa previa a la frontera
             
    Returns:
        r_frontera: Valor del radio donde se encuentra la frontera (donde n=2.5)
        P_frontera: Valor interpolado de la presión en la frontera
        T_frontera: Valor interpolado de la temperatura en la frontera
        L_frontera: Valor interpolado de la luminosidad en la frontera
        M_frontera: Valor interpolado de la masa en la frontera
    """
    n_temp=[n[i_frontera],n[i_frontera+1]]
    r_temp=[r[i_frontera],r[i_frontera+1]]; r_frontera=interpolacion_lineal([[n_temp[0],r_temp[0]],[n_temp[1],r_temp[1]]],1.5)
    P_temp=[P[i_frontera],P[i_frontera+1]]; P_frontera=interpolacion_lineal([[n_temp[0],P_temp[0]],[n_temp[1],P_temp[1]]],1.5)
    T_temp=[T[i_frontera],T[i_frontera+1]]; T_frontera=interpolacion_lineal([[n_temp[0],T_temp[0]],[n_temp[1],T_temp[1]]],1.5)
    L_temp=[L[i_frontera],L[i_frontera+1]]; L_frontera=interpolacion_lineal([[n_temp[0],L_temp[0]],[n_temp[1],L_temp[1]]],1.5)
    M_temp=[M[i_frontera],M[i_frontera+1]]; M_frontera=interpolacion_lineal([[n_temp[0],M_temp[0]],[n_temp[1],M_temp[1]]],1.5)
    return r_frontera,P_frontera,T_frontera,L_frontera,M_frontera

@njit
def primeras_3_capas_internas(r,M,f_M,L,f_L,T,f_T,P,K,mu,Z,X,T_central):
    """
    Calcula los valores de la presión, temperatura, luminosidad y masa y d/dr de todas de ellas en las 3 primeras capas de la discretización empezando en el centro.

    Args:
        r: Vector equiespaciado del centro a la superficie
        M: Vector con la masa a lo largo de la discretización
        f_M: Vector con el valor de dM/dr a lo largo de la discretización    
        L: Vector con la luminosidad a lo largo de la discretización
        f_L: Vector con el valor de dL/dr a lo largo de la discretización    
        T: Vector con la temperatura a lo largo de la discretización
        f_T: Vector con el valor de dT/dr a lo largo de la discretización    
        P: Vector con la presión a lo largo de la discretización
        K: Constante politrópica
        mu: Peso molecular medio
        Z: Metalicidad
        X: Fracción en masa de hidrógeno
        T_central: Temperatura en el centro de la estrella
             
    Returns:
        M: Vector con la masa en las 3 primeras capas
        f_M: Vector con el valor de dM/dr en las 3 primeras capas
        L: Vector con la luminosidad en las 3 primeras capas
        f_L: Vector con el valor de dL/dr en las 3 primeras capas
        T: Vector con la temperatura en las 3 primeras capas
        f_T: Vector con el valor de dT/dr en las 3 primeras capas
        P: Vector con la presión en las 3 primeras capas
    """
    T[0]=T_central;L[0]=0;M[0]=0
    P[0]=K*T[0]**2.5
    for j in range(1,3):
        (M,L,T,P)=ecuacion_43_44_45_46(mu,T_central,r,K,X,Z,j,M,L,T,P)
    return M,f_M,L,f_L,T,f_T,P #No necesitamos f_P porque usamos la ecuación del polítropo

def valores_en_la_frontera_up(n,r,P,T,L,M,j_frontera,r_frontera):
    """
    Calcula los valores interpolados de la presión, temperatura, luminosidad y masa en el radio donde
    se encuentra la frontera entre el núcleo y envoltura cuando integramos desde el núcleo.

    Args:
        n: Vector con el parámetro n  a lo largo de la parametrización
        P: Vector con la presión a lo largo de la discretización
        T: Vector con la temperatura a lo largo de la discretización
        L: Vector con la luminosidad a lo largo de la discretización
        M: Vector con la masa a lo largo de la discretización
        j_frontera: Capa previa a la frontera
        r_frontera: Valor del radio donde se encuentra la frontera (donde n=2.5)
             
    Returns:
        P_frontera: Valor interpolado de la presión en la frontera
        T_frontera: Valor interpolado de la temperatura en la frontera
        L_frontera: Valor interpolado de la luminosidad en la frontera
        M_frontera: Valor interpolado de la masa en la frontera
    """
    r_temp=[r[j_frontera],r[j_frontera+1]]
    P_temp=[P[j_frontera],P[j_frontera+1]]; P_frontera=interpolacion_lineal([[r_temp[0],P_temp[0]],[r_temp[1],P_temp[1]]],r_frontera)
    T_temp=[T[j_frontera],T[j_frontera+1]]; T_frontera=interpolacion_lineal([[r_temp[0],T_temp[0]],[r_temp[1],T_temp[1]]],r_frontera)
    L_temp=[L[j_frontera],L[j_frontera+1]]; L_frontera=interpolacion_lineal([[r_temp[0],L_temp[0]],[r_temp[1],L_temp[1]]],r_frontera)
    M_temp=[M[j_frontera],M[j_frontera+1]]; M_frontera=interpolacion_lineal([[r_temp[0],M_temp[0]],[r_temp[1],M_temp[1]]],r_frontera)
    return P_frontera,T_frontera,L_frontera,M_frontera

def main(M_total,R_total,L_total,T_central,X,Y,capas=100,it1=12,aux1=20,plot1=False): 
    """
    Resuelve la estructura de la estrella (devuelve las magnitudes físicas a lo largo del radio), manteniendo fija la masa total, el radio total, la luminosidad total y la composición química. Encuentra la temperatura central que optimiza este modelo y devuelve el error relativo total asociado a las diferencias entre las soluciones al integrar desde la superficie y desde el núcleo.
    
    Args:
        M_total: Masa total de la estrella. Valor característico de la estrella
        R_total: Radio total de la estrella. Valor inicial
        L_total: Luminosidad total de la estrella. Valor inicial
        T_central: Temperatura central de la estrella. Valor inicial
        X: Fracción en masa de hidrógeno. Valor característico de la estrella
        Y: Fracción en masa de helio. Valor característico de la estrella
        capas: Número de divisiones de la discretización. Por defecto 100
        it1: Número de iteraciones acotando el intervalo de la temperatura central centrándo cada nuevo intervalo en la que menor error relativo total obtuvo en la anterior iteración. Por defecto 12
        aux1: Número de divisiones del intervalo donde se busca la temperatura óptima .Por defecto 20
        plot1=False: Variable booleana. Si es True devuelve la gráfica del error total en función de la temperatura central. Por defecto False
        
    Returns:
        E: Lista que indica el tipo de ciclo usado (pp o CNO) para la generación de energía en cada capa de la discretización. En caso de no haber generación de energía devuelve '---'
        fase:
        i: Vector entero con los números de cada capa
        r_down: Vector con el radio en cada capa de la discretización (de la superficie al centro)
        P: Vector con la presión a lo largo de la discretización (de la superficie al centro)
        T: Vector con la temperatura a lo largo de la discretización (de la superficie al centro)
        L: Vector con la luminosidad a lo largo de la discretización (de la superficie al centro)
        M: Vector con la masa a lo largo de la discretización (de la superficie al centro)
        rho: Vector con la densidad a lo largo de la discretización (de la superficie al centro)
        n: Vector con el parámetro n  a lo largo de la parametrización
        T_central: Temperatura central de la estrella. Valor que optimiza el modelo
        r_frontera: Radio donde se encuentra la frontera entre el núcleo convectivo y la envoltura radiativa
        mu: Peso molecular medio
        Z: Metalicidad
        Error: Error relativo absoluto entre las soluciones al integrar desde abajo y desde arriba
        A: Recopilación de los resultados para su representación en tabla de la integración desde la superficie. 
        B: Recopilación de los resultados para su representación en tabla de la integración desde el núcleo.
        modelocompleto: Recopilación de los resultados finales (una vez pegadas las dos soluciones y añadidas las capas desde 0.9R_total hasta R_total) para su representación en tabla
    """
    Z=1-X-Y 
    mu=1/(2*X+0.75*Y+0.5*Z) 
    P=zeros(capas+1); T=zeros(capas+1);L=zeros(capas+1);M=zeros(capas+1)
    f_P=zeros(capas+1);f_T=zeros(capas+1);f_L=zeros(capas+1);f_M=zeros(capas+1) #Lo que cambia cada magnitud
    E=[];fase=[];
    
    #--------------------------------------------------------------------------------------------------------------------------------    
    #INTEGRACION DESDE LA SUPERFICIE-------------------------------------------------------------------------------------------------
    R_inicial=0.9*R_total
    h_down=-R_inicial/capas
    r_down=linspace(R_inicial,0,101)
    
    #PRIMERAS 3 CAPAS
    primeras3capasexternas=[['E','fase','i  ','r    ','P    ','T    ','L    ','M     ']]
    P,f_P,T,f_T,L,f_L,M,f_M=primeras_3_capas_externas(r_down,P,f_P,T,f_T,L,f_L,M,f_M,L_total,M_total,R_total,X,Z,mu,h_down) 
    for i in range(0,3):
        E+=['---'];fase+=['inicio']
        primeras3capasexternas+=[[E[i],fase[i],i,r_down[i],P[i],T[i],L[i],M[i]]]
    i_final=i
    
    #FASE A.1. ENVOLTURA RADIATIVA.
    faseA1=[['E','fase','i  ','r    ','P    ','T    ','L    ','M     ','n+1   ']]
    fase_A1=True
    n=[0]*101
    while fase_A1==True:
        (P,f_P,T,f_T,L,f_L,M,f_M,n,E,fase_A1,K)=fase_radiativa_A1(h_down,P,f_P,T,f_T,i,mu,r_down,M,f_M,L,f_L,X,Z,E,n)
        i+=1
    i-=1 #Porque los cálculos en la última capa se desechan por dominar el transporte convectivo frente al radiativo
    for j in range(i_final+1,i+1):
        fase+=['A.1']
        faseA1+=[[E[j],fase[j],j,r_down[j],P[j],T[j],L[j],M[j],n[j]+1]]
    i_frontera=i #Última capa radiativa
    
    #FASE A.2. NÚCLEO CONVECTIVO
    faseA2=[['E','fase','i  ','r    ','P    ','T    ','L    ','M     ','n+1     ']]
    fase_A2=True
    while fase_A2==True:
        (T,f_T,P,f_P,M,f_M,L,f_L,E,fase_A2)=fase_convectiva_A2(K,h_down,i,r_down,mu,T,f_T,M,f_M,P,f_P,L,f_L,Z,X,E)
        i+=1
    for j in range(i_frontera+1,i+1):
        fase+=['CONVEC ']
        faseA2+=[[E[j],fase[j],j,r_down[j],P[j],T[j],L[j],M[j],n[j]+1]]
        
    #VALORES EN LA FRONTERA. (Queremos hallar los valores de los parámetros físicos en la frontera, es decir, cuando n+1==2.5. Interpolamos linealmente todos los parámetros)
    frontera_envoltura_nucleo=[['n','r(n=2.5)','P(n=2.5)','T(n=2.5)','L(n=2.5)','M(n=2.5)']]
    (r_frontera,P_frontera,T_frontera,L_frontera,M_frontera)=valores_en_la_frontera_down(n,r_down,P,T,L,M,i_frontera)
    frontera_envoltura_nucleo+=[[2.5,r_frontera,P_frontera,T_frontera,L_frontera,M_frontera]]
    
    #RECOPILACIÓN DE LOS RESULTADOS PARA LA INTEGRACIÓN DESDE LA SUPERFICIE
    A=[['Primeras 3 capas externas:','Fase A.1. Envoltura radiativa:','Fase A.2. Núcleo convectivo:','Parámetros físicos interpolados en la frontera entre la envoltura y el núcleo:'],[primeras3capasexternas,faseA1,faseA2,frontera_envoltura_nucleo]]
    
    
    #--------------------------------------------------------------------------------------------------------------------------------
    #INTEGRACIÓN DESDE EL CENTRO-----------------------------------------------------------------------------------------------------
    capas_nucleo=capas-i_frontera+1
    P_nuc=zeros(capas_nucleo);T_nuc=zeros(capas_nucleo);L_nuc=zeros(capas_nucleo);M_nuc=zeros(capas_nucleo);
    f_P_nuc=zeros(capas_nucleo+1);f_T_nuc=zeros(capas_nucleo+1);f_L_nuc=zeros(capas_nucleo+1);f_M_nuc=zeros(capas_nucleo+1);
    h_up=R_inicial/capas;r_up=linspace(0,R_inicial,101)
      
    #HALLAMOS LA TEMPERATURA CENTRAL QUE MINIMIZA EL ERROR TOTAL---------------------------------------------------------------------
    iteracion=1;E_up=[]
    while iteracion<it1+1: #Hacemos it1 iteraciones acotando el intervalo de T_central centrándo cada nuevo intervalo en la T_central que menor Err_total haya obtenido en la anterior iteración
        Tc_i=T_central-0.05*T_central;Tc_f=T_central+0.05*T_central
        Tc=linspace(Tc_i,Tc_f,aux1);Err_tot=zeros(len(Tc))            
        
        for s in range(0,len(Tc)): #Calculamos el error relativo total entre las soluciones UP y DOWN para aux1 valores de la T_central
            #PRIMERAS 3 CAPAS INTERNAS (ahora utilizaremos como índice 'j')
            (M_nuc,f_M_nuc,L_nuc,f_L_nuc,T_nuc,f_T_nuc,P_nuc)=primeras_3_capas_internas(r_up,M_nuc,f_M_nuc,L_nuc,f_L_nuc,T_nuc,f_T_nuc,P_nuc,K,mu,Z,X,Tc[s])
            j=2
            #CAPAS POSTERIORES HASTA LA FRONTERA DEL NÚCLEO
            fase_A2=True #Reutilizamos el algoritmo A.2 porque estamos en la fase convectiva, pero esta vez la condición de parada es llegar a la frontera
            while fase_A2==True:
                (T_nuc,f_T_nuc,P_nuc,f_P_nuc,M_nuc,f_M_nuc,L_nuc,f_L_nuc,E_up,fase_A2)=fase_convectiva_A2(K,h_up,j,r_up,mu,T_nuc,f_T_nuc,M_nuc,f_M_nuc,P_nuc,f_P_nuc,L_nuc,f_L_nuc,Z,X,E_up)
                j+=1
                if r_up[j]>r_frontera: #Hemos llegado a la frontera
                    fase_A2=False
            j_frontera=j-1
            #VALORES EN LA FRONTERA (queremos hallar los valores de los parámetros físicos en la frontera, es decir, cuando r[j]=r_frontera. Interpolamos linealmente todos los parámetros)
            (P_frontera_down,T_frontera_down,L_frontera_down,M_frontera_down)=valores_en_la_frontera_up(n,r_up,P_nuc,T_nuc,L_nuc,M_nuc,j_frontera,r_frontera)
            #COMPARACIÓN VALORES EN LA FRONTERA DE LAS SOLUCIONES DOWN Y UP
            Err_P=abs(P_frontera-P_frontera_down)/P_frontera_down*100
            Err_T=abs(T_frontera-T_frontera_down)/T_frontera_down*100
            Err_L=abs(L_frontera-L_frontera_down)/L_frontera_down*100
            Err_M=abs(M_frontera-M_frontera_down)/M_frontera_down*100
            Err_tot[s]=(Err_P**2+Err_T**2+Err_L**2+Err_M**2)**0.5
        indice_min = int(where(Err_tot==min(Err_tot))[0]);
        T_central=Tc[indice_min]
        iteracion+=1
        Error=min(Err_tot)
        
    if plot1:        
        fig, ax = plt.subplots()
        titulo='Error total en función de la temperatura central para R='+str(R_total)+' y L='+str(L_total)
        plt.plot(Tc,Err_tot);ax.set_title(titulo,fontdict = {'fontsize':8.5, 'fontweight':'bold', 'color':'tab:blue'}),ax.set_xlabel('Temperatura central');ax.set_ylabel('Error total')
        plt.show()

    #INTEGRACIÓN DESDE EL CENTRO CON LA MEJOR TEMPERATURA CENTRAL (repetimos los cálculos con la Tc óptima para recopilar los valores en B y poder tabularlos fácilemente)-----------------------------------------------
    fase_up=[];E_up=[]
    
    #PRIMERAS 3 CAPAS INTERNAS (ahora utilizaremos como índice 'j')
    primeras3capasinternas=[['E','fase','i  ','r    ','P    ','T    ','L    ','M    ']]    
    (M_nuc,f_M_nuc,L_nuc,f_L_nuc,T_nuc,f_T_nuc,P_nuc)=primeras_3_capas_internas(r_up,M_nuc,f_M_nuc,L_nuc,f_L_nuc,T_nuc,f_T_nuc,P_nuc,K,mu,Z,X,T_central)
    for i in range(0,3):
        E_up+=['---'];fase_up+=['centro']
        primeras3capasinternas+=[[E_up[i],fase_up[i],i,r_up[i],P_nuc[i],T_nuc[i],L_nuc[i],M_nuc[i]]]
    j=2
    
    #CAPAS POSTERIORES HASTA LA FRONTERA DEL NÚCLEO (Reutilizamos el algoritmo A.2 porque estamos en la fase convectiva, pero esta vez la condición de parada es llegar a la frontera)
    faseA2interna=[['E','fase','i  ','r    ','P    ','T    ','L    ','M    ']]
    fase_A2=True
    while fase_A2==True:
        (T_nuc,f_T_nuc,P_nuc,f_P_nuc,M_nuc,f_M_nuc,L_nuc,f_L_nuc,E_up,fase_A2)=fase_convectiva_A2(K,h_up,j,r_up,mu,T_nuc,f_T_nuc,M_nuc,f_M_nuc,P_nuc,f_P_nuc,L_nuc,f_L_nuc,Z,X,E_up)
        j+=1
        if r_up[j]>r_frontera: #Hemos llegado a la frontera
            fase_A2=False
    j_frontera=j-1
    for i in range(3,j):
        fase_up+=['CONVEC ']
        faseA2interna+=[[E_up[i],fase_up[i],i,r_up[i],P_nuc[i],T_nuc[i],L_nuc[i],M_nuc[i]]]

    #VALORES EN LA FRONTERA (queremos hallar los valores de los parámetros físicos en la frontera, es decir, cuando r[j]=r_frontera. Interpolamos linealmente todos los parámetros).
    capaadicional=[['E','fase','i  ','r    ','P    ','T    ','L    ','M    ']]
    fase_up+=['CONVEC '];
    capaadicional+=[[E_up[j],fase_up[j],j,r_up[j],P_nuc[j],T_nuc[j],L_nuc[j],M_nuc[j]]]
    frontera_nucleo_envoltura=[['n','r(n=2.5)','P(n=2.5)','T(n=2.5)','L(n=2.5)','M(n=2.5)']]
    (P_frontera_down,T_frontera_down,L_frontera_down,M_frontera_down)=valores_en_la_frontera_up(n,r_up,P_nuc,T_nuc,L_nuc,M_nuc,j_frontera,r_frontera)
    frontera_nucleo_envoltura+=[[2.5,r_frontera,P_frontera,T_frontera,L_frontera,M_frontera]]
    
    #COMPARACIÓN VALORES EN LA FRONTERA DE LAS SOLUCIONES DOWN Y UP
    errorrelativodownup=[['ErrP   ','ErrT   ','ErrL   ','ErrM   ','ErrTotal   ']]
    Err_P=abs(P_frontera-P_frontera_down)/P_frontera_down
    Err_T=abs(T_frontera-T_frontera_down)/T_frontera_down
    Err_L=abs(L_frontera-L_frontera_down)/L_frontera_down
    Err_M=abs(M_frontera-M_frontera_down)/M_frontera_down
    Err_tot[s]=((Err_P**2+Err_T**2+Err_L**2+Err_M**2)**0.5)*100 #Error en porcentaje
    errorrelativodownup+=[[Err_P,Err_T,Err_L,Err_M,Err_tot[s]]]
    
    #--------------------------------------------------------------------------------------------------------------------------------
    #AÑADIMOS LAS PRIMERAS CAPAS-----------------------------------------------------------------------------------------------------
    capas_previas=int((R_total-R_inicial)//h_up) #Número de capas que faltan por meter
    r_temp=arange(R_inicial+h_up,R_total, h_up); r_temp=r_temp[::-1] #Invertimos el orden para que vaya desde R_total a R_inicial
    a=zeros(capas_previas);b=ones(capas_previas);
    r_down=concatenate((r_temp,r_down));P=concatenate((a,P));T=concatenate((a,T));L=concatenate((b*L_total,L));M=concatenate((b*M_total,M)) #En las capas previa L=L_total y M=M_total
    for i in range(0,capas_previas): #Calculamos P y T desde R_total a R_inicial
        (P[i],T[i])=ecuacion_35_36(r_down[i],M_total,R_total,L_total,X,Z,mu)
        (f_P[i],f_T[i])=ecuacion_19_21(r_down[i],P[i],T[i],L[i],M[i],X,Z,mu) 
     
    #--------------------------------------------------------------------------------------------------------------------------------
    #MODELO COMPLETO PARA R_TOTAL Y L_TOTAL------------------------------------------------------------------------------------------
    #Cogemos las componentes interiores al núcleo de P,T,L,M y definimos con ellas P_nuc,T_nuc,L_nuc,M_nuc (que están invertidas de orden y cuya última componente antes de invertir, que hemos usado para interpolar en la frontera del núcleo, debe eliminarse)
    P_nuc=delete(P_nuc,-1);T_nuc=delete(T_nuc,-1);L_nuc=delete(L_nuc,-1);M_nuc=delete(M_nuc,-1) #Eliminamos la última componente
    P_nuc=P_nuc [::-1];T_nuc=T_nuc [::-1];L_nuc=L_nuc[::-1];M_nuc=M_nuc [::-1] #Los invertimos de orden
    for s in range(capas-i_frontera): #Eliminamos las componentes del interior del núcleo de P,T,L,M
        P=delete(P,-1);T=delete(T,-1);L=delete(L,-1);M=delete(M,-1);del E[-1]
    P=concatenate((P,P_nuc));T=concatenate((T,T_nuc));L=concatenate((L,L_nuc));M=concatenate((M,M_nuc)); #Pegamos las dos soluciones (integración desde la superficie y desde el núcleo) bien ordenadas
    modelocompleto=[['E','fase','i','r','P','T','L','M','Rho (E-7)','n']]
    i=arange(-capas_previas,capas+1,1) #El índice asociado a las capas entre R_inicial y R_total es negativo para no tener que renumerar todas las capas posteriores
    del fase[-1];del fase[-1];del fase[-1];fase=['^^^^^^']*capas_previas+fase+['CENTRO']*3
    n=[round(elem, 5) for elem in n];n=[0]*capas_previas+n;n=['---' if x<=0.001 else x for x in n]
    del E_up[-1];E_up=E_up[::-1] #La última componente de E_up sin invertir corresponde a la capa adicional fuera del núcleo usada para interpolar
    E=['---']*capas_previas+E+E_up    
    rho=P*mu/(T*8.31447*10**7);rho_107=rho*10**7 #rho está en g/cm^3 y rho_107 está en g/cm^-3 (10^-7)
    for s in range(0,capas+capas_previas+1):
        modelocompleto+=[[E[s],fase[s],str(i[s]),r_down[s],P[s],T[s],L[s],M[s],rho_107[s],n[s]]]
    B=[['Primeras 3 capas internas:','Capas posteriores hasta la frontera del núcleo:','Salimos del núcleo. Calculamos una capa adicional suponiendo convectivo:','Parámetros físicos interpolados en la frontera entre el núcleo y la envoltura:','Errores relativos entre las soluciones down y up (%):'],[primeras3capasinternas,faseA2interna,capaadicional,frontera_nucleo_envoltura,errorrelativodownup]]
    return E,fase,i,r_down,P,T,L,M,rho,n,T_central,r_frontera,mu,Z,Error,A,B,modelocompleto
#------------------------------------------------------------------------------------------------------------------------------------    
#------------------------------------------------------------------------------------------------------------------------------------
