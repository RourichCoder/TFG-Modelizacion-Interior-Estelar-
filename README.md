# TFG-Modelizacion-Interior-Estelar-
Modelización del interior de una estrella. TFG Física UCM 2023 - Alejandro Rourich González

*ANTES DE EJECUTAR CUALQUIER ARCHIVO*
- Es necesario tener instaladas las librerias "numpy", "numba" y "tabulate". 
(Utilizar el instalador "pip")
- Si se ejecuta el archivo "DatosEstrellaTFG" la línea 123 de "RtotYLtotOptimosYRepresentacion.py" debe estar comentada y la 124 descomentada. 
(Asegura el intervalo correcto de busqueda del radio y la luminosidad óptimos)	
- Para ejecutar el resto de archivos la línea 124 de "RtotYLtotOptimosYRepresentacion.py" debe estar comentada y la 123 descomentada. 

*Todos los archivos tienen su propia documentación. A continuación se expone un resumen de la función de cada uno:*

- "SolucionRtotYLtotFijos.py": Para una masa y composición química resuelve la estrella sin modificar el radio ni la luminosidad (ambos valores iniciales).
El único valor inicial que modifica minimizando el error es la temperatura central. Ejecutar este archivo no produce ningún resultado, pues está compuesto únicamente por funciones.
- "RtotYLtotOptimosYRepresentacion.py": Compuesto únicamente por dos funciones (si se ejecuta no hace nada). Ambas hacen uso de la función "main" de"SolucionRtotYLtotFijos.py".
Las funciones son "modelo_sin_optimizar_R_y_L" que representa con gráficas y tablas los cálculos realizados en "main" y "modelo_optimizado_para_X_Y_y_M" que halla el radio y la luminosidad
que minimizan el error del modelo. Además, dependiendo de los valores de las variables booleanas se obtienen diferentes representaciones (gráficas y tablas).
- "DatosEstrellaTFG.py": Script que ejecuta las dos funciones del archivo “RtotYLtotOptimosYRepresentacion.py” con los datos de la estrella propuesta Mtotal=5x10^{33}g, X=0.75 e
Y=0.20.
- "ComparativaMasaBajaAlta.py": Script sin funciones que utiliza la función "modelo_optimizado_para_X_Y_y_M" para resolver la estructura de dos estrellas con la misma composición química, pero distinta masa.
- "VectorDeMasas.py": Script sin funciones que utiliza la función "modelo_optimizado_para_X_Y_y_M" para resolver 20 estrellas con masas entre 1.75M_sun y 6.5M_sun y la misma composición
química.
- "ComposicionQuimica.py": Script sin funciones que utiliza la función "modelo_optimizado_para_X_Y_y_M" para resolver la estructura de dos estrellas con la misma masa, pero distinta composición química.


