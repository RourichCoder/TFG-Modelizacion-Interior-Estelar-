import SolucionRtotYLtotFijos
from SolucionRtotYLtotFijos import main

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose,identity,array
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

#PARA R_TOTAL Y L_TOTAL PREDETERMINADOS----------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
def modelo_sin_optimizar_R_y_L(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1=False,plot2=False):
    """
    Resuelve la estructura de la estrella, manteniendo el radio total y la luminosidad total fijos, a partir del código en "SolucionRtotYLtotFijos.py" y representa los resultados gráficamente y con tablas.
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
        plot2=False: Variable booleana. Si es True devuelve la gráfica de las magnitudes físicas (presión, temperatura, luminosidad, masas y densidad) normalizadas en función del radio. Por defecto False
        
    Returns:
        La función no devuelve ninguna variable. Sin embargo, devuelve los resultados en forma de tabla siguiendo el siguiente orden:
            -Integración desde la superficie: [Primeras 3 capas externas - Fase A.1. - Fase A.2 - Parámertros inteporlados en la frontera]
            -La temperatura central óptima para el R_total y L_total y el error asociado entre las soluciones al integrar desde la superficie y desde el centro con esa temperatura central
            -Integración desde el centro: [Primeras 3 capas internas - Fase A.2. (hasta la frontera del núcleo) - Parámertros inteporlados en la frontera (requiere calcular una capa extra en la envoltura)]
            -Errores relativos entre las soluciones down y up
            -Tabla del modelo completo (unión de las soluciones down y up y concatenación de las capas desde 0.9R_total a R_total)
        Adicionalmente, 
            -Si plot1=True: Representa el error total en función de la temperatura central
            -Si plot2=True: Representa la gráfica de las magnitudes físicas (presión, temperatura, luminosidad, masas y densidad) normalizadas en función del radio
    """
    (E,fase,i,r_down,P,T,L,M,rho,n,T_central,r_frontera,mu,Z,Error,A,B,modelocompleto,M_frontera)=main(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1)

    print('-'*110);print('-'*110);print(' '*22,'INTEGRACION DESDE LA SUPERFICIE');print('-'*110);print('-'*110)
    for s in range(len(A[0])):
        print(A[0][s])
        print(tabulate(A[1][s], headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))
    print('-'*110);print('La mejor T_central para R_total =',"{0:.3f}".format(R_total),'y L_total =',"{0:.3f}".format(L_total),'es', "{0:.6f}".format(T_central))
    print('EL error total asociado es',"{0:.4f}".format(Error));print('-'*110)
    print('-'*110); print(' '*14,'INTEGRACIÓN DESDE EL CENTRO PARA Tc=', "{0:.6f}".format(T_central)); print('-'*110);print('-'*110)
    for s in range(len(B[0])):
        print(B[0][s])
        print(tabulate(B[1][s], headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))
    print('-'*110);print('-'*110);print('MODELO COMPLETO para R_total =',"{0:.4f}".format(R_total),', L_total =',"{0:.4f}".format(L_total),'y T_central =', "{0:.4}".format(T_central))
    print(tabulate(modelocompleto, headers='firstrow', tablefmt='fancy_grid',stralign="left",floatfmt='.7f'))
    print('El error total de este modelo es',"{0:.4f}".format(Error));print('-'*110)
    
    if plot2==True:
        P_norm=P/amax(P);T_norm=T/amax(T); L_norm=L/amax(L); M_norm=M/amax(M);rho_norm=rho/amax(rho)
        plt.figure()
        p1, = plt.plot(r_down, P_norm, linestyle='-', label='Presión') # Gráfica de la Presión con línea continua
        p2, = plt.plot(r_down, T_norm, linestyle='--', label='Temperatura')# Gráfica de la Temperatura con línea punteada       
        p3, = plt.plot(r_down, L_norm, linestyle='-.', label='Luminosidad') # Gráfica de la Luminosidad con línea de puntos y guiones
        p4, = plt.plot(r_down, M_norm, linestyle=':', label='Masa')# Gráfica de la Masa con línea de puntos más largos
        p5, = plt.plot(r_down, rho_norm, linestyle='dotted', label='Densidad') # Gráfica de la Densidad con línea de puntos pequeños
        plt.axvline(x=r_frontera, linestyle='dashed',color='grey',label='Frontera',alpha=0.35) #Opacidad del 35% para poder ver el salto de las magnitudes
        plt.title('Magnitudes normalizadas para M='+str("{0:.3f}".format(M_total)),fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=10)
        plt.xlabel('Radio '+r'$(10^{10} cm)$')
        plt.ylabel('Magnitudes físicas normalizadas')
        plt.legend(handles=[p1, p2, p3, p4, p5,plt.Line2D([], [], linestyle='dashed', color='grey', label='Frontera', alpha=0.2)])

        
#Para exportar la tabla "modelo completo" a latex: print(tabulate(modelocompleto,headers='firstrow',tablefmt='latex',stralign='center',floatfmt='.7f'))



#PARA R_TOTAL Y L_TOTAL QUE OPTIMIZAN EL MODELO----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
def modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1,representacion2,representacion3,representacion4,representacion5,deltaR=0.1,deltaL=1,profundidad=12): 
    """
    Resuelve la estructura de la estrella hallando el radio total y la luminosidad total óptimos a partir del código en "SolucionRtotYLtotFijos.py" y representa los resultados gráficamente y con tablas.
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
        iteraciones: Número de veces que se refina el mallado para hallar el radio total y la luminosidad total óptimos
        representacion1: Variable booleana. Si es True devuelve la gráfica del mallado del error total en función de las luminosidad y radios totales en cada iteración (tantas gráficas como iteraciones)
        Representación2: Variable booleana. Si es True devuelve la gráfica del mallado de la temperatura central óptima en función de las luminosidad y radios totales en cada iteración (tantas gráficas como iteraciones)
        Representación3: Variable booleana. Si es True devuelve las tablas del modelo que optimiza el error
        Representación4: Variable booleana. Si es True devuelve la gráfica de las magnitudes física en función del radio (un plot para cada magnitud, por lo que devuelve 5 gráficas)
        Representación5: Variable booleana. Si es True devuelve la gráfica de todas las magnitudes físicas normalizadas (una única gráfica)
        deltaR: Paso del radio en el mallado para hallar el radio total y la luminosidad total óptimos. Por defecto 0.1
        deltaL: Paso de la luminosidad en el mallado para hallar el radio total y la luminosidad total óptimos. Por defecto 1
        profundidad: Entero par (pues es  la suma de casillas a derecha e izquierda del mallado). 
        
    Returns:
        M_total: Masa total de la estrella. Valor característico de la estrella
        R_total: Radio total de la estrella. Valor que optimiza el modelo
        L_total: Luminosidad total de la estrella. Valor que optimiza el modelo
        T_central: Temperatura central de la estrella. Valor que optimiza el modelo
        r_down: Vector con el radio en cada capa de la discretización (de la superficie al centro)
        P: Vector con la presión a lo largo de la discretización (de la superficie al centro)
        T: Vector con la temperatura a lo largo de la discretización (de la superficie al centro)
        L: Vector con la luminosidad a lo largo de la discretización (de la superficie al centro)
        M: Vector con la masa a lo largo de la discretización (de la superficie al centro)
        rho: Vector con la densidad a lo largo de la discretización (de la superficie al centro)
        M_frontera: Masa del núcleo convectivo
        
        Además de volver las anteriores variables y las gráficas en función de los valores de las variables booleanas de entradas. Si la variable booleana "Representación3" es True entonces devuelve la siguiente tabla:
            En cada iteración (cada vez que hacemos el mallado de R y L):
            -Devuelve la luminosidad total, el radio total y la temperatura central y su error total asociado
            Para la última iteración (que será la de menor error):
            -Integración desde la superficie: [Primeras 3 capas externas - Fase A.1. - Fase A.2 - Parámertros inteporlados en la frontera]
            -La temperatura central óptima para el R_total y L_total y el error asociado entre las soluciones al integrar desde la superficie y desde el centro con esa temperatura central
            -Integración desde el centro: [Primeras 3 capas internas - Fase A.2. (hasta la frontera del núcleo) - Parámertros inteporlados en la frontera (requiere calcular una capa extra en la envoltura)]
            -Errores relativos entre las soluciones down y up
            -Tabla del modelo completo (unión de las soluciones down y up y concatenación de las capas desde 0.9R_total a R_total)
    """
    #RESOLUCIÓN DE LA ESTRELLA VARIANDO R_total Y L_total--------------------------------------------------------------------------------------------------------
    R_vector=[R_total];L_vector=[L_total]
    # R_vector=[R_total];L_vector=[42.5] #Para los datos propuestos dice buscar en los intervalos (R_total=[11.5,12.5] y L_total=[35.0,50.0])
    for r in range(profundidad//2):
        R_vector=[R_vector[0]-deltaR]+R_vector+[R_vector[-1]+deltaR]
        L_vector=[L_vector[0]-deltaL]+L_vector+[L_vector[-1]+deltaL]   
    R_vector=array(R_vector); L_vector=array(L_vector)
        
    Error_mat=zeros((len(L_vector),len(R_vector)));Tc_mat=zeros((len(L_vector),len(R_vector))) #Mallado
    repeticion=0;temp=10**6;x=0
    
    while repeticion<iteraciones: #Hacemos "iteraciones" del mallado, en cada uno con un paso más fino que el anterior
        for s in range(len(L_vector)):
            for p in range(len(R_vector)):
                (E,fase,i,r_down,P,T,L,M,rho,n,T_central,r_frontera,mu,Z,Error_mat[profundidad-s][p],A,B,modelocompleto,M_frontera)=main(M_total,R_vector[p],L_vector[s],T_central,X,Y,capas,it1,aux1,plot1=False)
                Tc_mat[profundidad-s][p]=T_central
                if Error_mat[profundidad-s][p]<temp: #Guardamos temporalmente el mejor resultado
                    r_downsave=r_down;Psave=P;Tsave=T;Lsave=L;Msave=M;rhosave=rho;Asave=A;Bsave=B;modelocompletosave=modelocompleto;r_fronterasave=r_frontera
                    L_totalsave=L_vector[s]; R_totalsave=R_vector[p]; T_centralsave=T_central
                    temp=Error_mat[profundidad-s][p] #Actualizamos el nuevo mejor error conseguido
                    mejora=True
        Err_total=temp
        R_total=R_totalsave; L_total=L_totalsave;  #Los nuevos Rtotal y Ltotal corresponden al menor error del mallado
        
        if mejora and representacion1:
            plt.figure(figsize=(32,22));
            plt.pcolormesh(R_vector,L_vector[::-1],Error_mat,cmap='viridis')
            if x<2: #En las primeras iteraciones fijamos el intervalo del colorbar, para que los valores que diverjan no saturen el gráfic
                plt.clim(0, 60)
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=25)
            plt.tick_params(axis='both', labelsize=35)
            plt.xlabel("Radio total", size = 40) 
            plt.ylabel("Luminosidad total", size = 40) 
            plt.title('Error total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold','size': 48})
            plt.scatter(R_total,L_total,color='red')
            plt.show()
            x+=1
            
            if representacion2:
                plt.figure(figsize=(32,22));
                plt.pcolormesh(R_vector,L_vector[::-1],Tc_mat,cmap='viridis')
                cbar = plt.colorbar()
                cbar.ax.tick_params(labelsize=25)
                plt.tick_params(axis='both', labelsize=35)
                plt.xlabel("Radio total", size = 40) 
                plt.ylabel("Luminosidad total", size = 40) 
                plt.title('Tcentral que minimiza el error para cada Rtot y Ltot',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold','size': 48})
                plt.show()
        mejora=False #Solo representamos Err_Mat cuando en la iteración actual el error en en alguno casilla es menor que el error mínimo alcanzado hasta ahora
        
        deltaR*=0.7; deltaL*=0.7; #Hacemos el paso del mallado más fino
        R_vector=[R_total];L_vector=[L_total]
        for r in range(profundidad//2):
            R_vector=[R_vector[0]-deltaR]+R_vector+[R_vector[-1]+deltaR]
            L_vector=[L_vector[0]-deltaL]+L_vector+[L_vector[-1]+deltaL]   
        R_vector=array(R_vector); L_vector=array(L_vector)
        Error_mat=zeros((len(L_vector),len(R_vector)));Tc_mat=zeros((len(L_vector),len(R_vector)))
        
        print('El nuevo error total es',"{0:.3f}".format(Err_total),'para una luminosidad total',"{0:.3f}".format(L_total),', un radio total',"{0:.3f}".format(R_total),'y Tc=', "{0:.4}".format(T_central));print('-'*105);
        repeticion+=1

    #PRINT DEL MODELO QUE OPTIMIZA EL ERROR---------------------------------------------------------------------------------------------------------
    r_down=r_downsave;P=Psave;T=Tsave;L=Lsave;M=Msave;rho=rhosave;A=Asave;B=Bsave;modelocompleto=modelocompletosave;T_central=T_centralsave
    if representacion3:
        print('-'*103);print(' '*32,'MODELO QUE OPTIMIZA EL ERROR');print('-'*103)
        print('-'*103);print(' '*32,'INTEGRACION DESDE LA SUPERFICIE');print('-'*103)
        for s in range(len(A[0])):
            print(A[0][s])
            print(tabulate(A[1][s], headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))
        print('-'*80);print('La mejor T_central para R_total =',"{0:.3f}".format(R_total),'y L_total =',"{0:.3f}".format(L_total),'es', "{0:.6f}".format(T_central))
        print('EL error total asociado es',"{0:.4f}".format(Err_total));print('-'*80)
        print('-'*80); print(' '*14,'INTEGRACIÓN DESDE EL CENTRO PARA Tc=', "{0:.6f}".format(T_central)); print('-'*80);print('-'*80)
        for s in range(len(B[0])):
            print(B[0][s])
            print(tabulate(B[1][s], headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))
        print('-'*103);print('-'*103);print('MODELO COMPLETO para R_total =',"{0:.4f}".format(R_total),', L_total =',"{0:.4f}".format(L_total),'y T_central =', "{0:.4}".format(T_central))
        print(tabulate(modelocompleto, headers='firstrow', tablefmt='fancy_grid',stralign='center',floatfmt='.7f'))
    print('El error total de este modelo es',"{0:.4f}".format(Err_total))
    # print(tabulate(modelocompleto,headers='firstrow',tablefmt='latex',stralign='center',floatfmt='.7f'))
    
    #REPRESENTACIÓN GRÁFICA  DEL MODELO QUE OPTIMIZA EL ERROR----------------------------------------------------------------------------------------
    if representacion4:
        #Sin normalizar
        y=[[P,T,L,M,rho],['Presión','Temperatura','Luminosidad','Masa','Densidad'],['Presión  '+r'$(10^{15}$' +' din '+ r'$cm^{-2})$', 'Temperatura  '+r'$(10^{7}K)$', 'Luminosidad '+r'$(10^{33}$' +' erg '+ r'$s^{-1})$', 'Masa ' +r'$(10^{33}g)$', 'Densidad '+r'$(10^{33}$'+ ' g '+r'$10^{-30}$'+ ' cm)']]
        for s in range(len(y[0])):
            plt.figure()
            plt.title(y[1][s] +' en función del radio',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=12)
            plt.xlabel('Radio '+r'$(10^{10} cm)$')
            plt.ylabel(y[2][s])
            plt.plot(r_down,y[0][s])
        P_norm=P/amax(P); T_norm=T/amax(T); L_norm=L/amax(L); M_norm=L/amax(M);
    if representacion5:
        #Normalizado
        P_norm=P/amax(P);T_norm=T/amax(T); L_norm=L/amax(L); M_norm=M/amax(M);rho_norm=rho/amax(rho)
        plt.figure()
        p1, = plt.plot(r_down, P_norm, linestyle='-', label='Presión') # Gráfica de la Presión con línea continua
        p2, = plt.plot(r_down, T_norm, linestyle='--', label='Temperatura')# Gráfica de la Temperatura con línea punteada       
        p3, = plt.plot(r_down, L_norm, linestyle='-.', label='Luminosidad') # Gráfica de la Luminosidad con línea de puntos y guiones
        p4, = plt.plot(r_down, M_norm, linestyle=':', label='Masa')# Gráfica de la Masa con línea de puntos más largos
        p5, = plt.plot(r_down, rho_norm, linestyle='dotted', label='Densidad') # Gráfica de la Densidad con línea de puntos pequeños
        plt.axvline(x=r_fronterasave, linestyle='dashed',color='grey',label='Frontera',alpha=0.35) #Opacidad del 35% para poder ver el salto de las magnitudes
        plt.title('Magnitudes normalizadas para M='+str("{0:.3f}".format(M_total)),fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=10)
        plt.xlabel('Radio '+r'$(10^{10} cm)$')
        plt.ylabel('Magnitudes físicas normalizadas')
        plt.legend(handles=[p1, p2, p3, p4, p5,plt.Line2D([], [], linestyle='dashed', color='grey', label='Frontera', alpha=0.2)])
    M_total=M[0];L_total=L[0]
    return (R_total,M_total,L_total,T_central,r_down,P,T,L,M,rho,M_frontera)

#Para exportar la tabla "modelo completo" a latex: print(tabulate(modelocompleto,headers='firstrow',tablefmt='latex',stralign='center',floatfmt='.7f'))
