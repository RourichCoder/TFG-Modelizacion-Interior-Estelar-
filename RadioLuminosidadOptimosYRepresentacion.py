import SolucionParaRtotYLtotFijos
from SolucionParaRtotYLtotFijos import main

from numba import njit
from numpy import zeros, ones, linspace, where, concatenate, ones, arange, delete, amin, amax, pi,format_float_scientific, meshgrid, transpose, arange, transpose,identity,array
from tabulate import tabulate   
import time
import matplotlib.pyplot as plt

#PARA R_TOTAL Y L_TOTAL PREDETERMINADOS----------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
def modelo_sin_optimizar_R_y_L(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1=False):
    (E,fase,i,r_down,P,T,L,M,rho,n,T_central,r_frontera,mu,Z,Error,A,B,modelocompleto)=main(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,plot1)
    
    #PRINT DEL MODELO--------------------------------------------------------------------------------------------------------------------------
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
    
    # print(tabulate(modelocompleto,headers='firstrow',tablefmt='latex',stralign='center',floatfmt='.7f'))
    
    # P_norm=P/amax(P);T_norm=T/amax(T); L_norm=L/amax(L); M_norm=M/amax(M);rho_norm=rho/amax(rho)
    # plt.figure()
    # p1,p2,p3,p4,p5=plt.plot(r_down,P_norm,r_down,T_norm,r_down,L_norm,r_down,M_norm,r_down,rho_norm)
    # plt.title('Magnitudes normalizadas para M='+str("{0:.3f}".format(M_total)),fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=10)
    # plt.xlabel('Radio '+r'$(10^{10} cm)$')
    # plt.ylabel('Magnitudes físicas normalizadas')
    # plt.legend(['Presión','Temperatura','Luminosidad','Masa','Densidad'])        


#PARA R_TOTAL Y L_TOTAL QUE OPTIMIZAN EL MODELO----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------------
def modelo_optimizado_para_X_Y_y_M(M_total,R_total,L_total,T_central,X,Y,capas,it1,aux1,iteraciones,representacion1,representacion2,representacion3,representacion4,representacion5): 
    deltaR=0.1; deltaL=1; profundidad=12 #Paso inicial. Profundidad debe ser PAR, pues es  la suma de casillas a derecha e izquierda
    
    R_vector=[R_total];L_vector=[L_total]
    for r in range(profundidad//2):
        R_vector=[R_vector[0]-deltaR]+R_vector+[R_vector[-1]+deltaR]
        L_vector=[L_vector[0]-deltaL]+L_vector+[L_vector[-1]+deltaL]   
    R_vector=array(R_vector); L_vector=array(L_vector)
        
    Error_mat=zeros((len(L_vector),len(R_vector)));Tc_mat=zeros((len(L_vector),len(R_vector))) #Mallado
    repeticion=0;temp=10**6
    
    while repeticion<iteraciones: #Hacemos "iteraciones" mallados, en cada uno con un paso más fino que el anterior
        for s in range(len(L_vector)):
            for p in range(len(R_vector)):
                (E,fase,i,r_down,P,T,L,M,rho,n,T_central,r_frontera,mu,Z,Error_mat[profundidad-s][p],A,B,modelocompleto)=main(M_total,R_vector[p],L_vector[s],T_central,X,Y,capas,it1,aux1,plot1=False)
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
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=25)
            plt.tick_params(axis='both', labelsize=35)
            plt.xlabel("Radio total", size = 40) 
            plt.ylabel("Luminosidad total", size = 40) 
            plt.title('Error total',fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold','size': 48})
            plt.scatter(R_total,L_total,color='red')
            plt.show()
            
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
        
        deltaR*=0.8; deltaL*=0.8; #Hacemos el paso del mallado más fino
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
        p1,p2,p3,p4,p5=plt.plot(r_down,P_norm,r_down,T_norm,r_down,L_norm,r_down,M_norm,r_down,rho_norm)
        plt.axvline(x=r_fronterasave,color="grey")
        plt.title('Magnitudes normalizadas para M='+str("{0:.3f}".format(M_total)),fontdict={'family': 'serif', 'color' : 'darkblue', 'weight': 'bold'},fontsize=10)
        plt.xlabel('Radio '+r'$(10^{10} cm)$')
        plt.ylabel('Magnitudes físicas normalizadas')
        plt.legend(['Presión','Temperatura','Luminosidad','Masa','Densidad'])
    M_total=M[0];L_total=L[0]
    return (R_total,M_total,L_total,T_central,r_down,P,T,L,M,rho)
    
#Para exportar la tabla a latex: print(tabulate(modelocompleto,headers='firstrow',tablefmt='latex',stralign='center',floatfmt='.7f'))
#0.04