#!/usr/bin/python3.7
# -*- coding: UTF-8 -*-

# El primer codigo ordenado de manera aleatoria los archivos y genera un archivo NPY
import numpy as np
import os # Permite interactuar con el sistema operativo, para manejar directorios y archivos
import random 
from datetime import datetime as DT
import sys # Se uso para acceder a los argumentos del sistema

path = sys.argv[0] = "C:/Users/artur/Documents/Replicacion_de_articulos/Ambiente virtual TF 3.7/Modelo_CNN/program_1/validation_samples"
# Enlista los archivos que se encuentran en la ruta anterior
dirs = os.listdir(path)

# Establece la ruta y nombre del archivo que se va a generar
output_file_name=sys.argv[0] = "C:/Users/artur/Documents/Replicacion_de_articulos/Ambiente virtual TF 3.7/Modelo_CNN/program_1/output/Example_validation_4908_TCGA_samples"

time = DT.utcnow() #Toma la hora y fecha actual
# genera una semilla
seed = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute) + str(time.microsecond)
# Genera la semilla aleatoria
random.seed(seed)
# Reordenado de manera aleatoria los archivos para reduccion de sesgo 
random.shuffle(dirs)  
print (seed)

# Creacion de las variables
sample_titles = np.array ( dirs )
samples = []
labels = []
elements = []
O_names = []

for name in dirs:
    O_names.append ( name )
    data = name.split('-')
    labels.append ( data[2] )
    elements_name = []
    with open ( path+"/"+name ) as file:
        for line in file:
            line = line.strip().split()
            elements_name.append(line)
    elements.append(elements_name)

#------------------------ Conversion de los datos a matrices NumPy ------------------------
#convierte la lista elements en una matriz que representa las caracteristicas de cada dato
x_samples = np.array ( elements ).astype ( np.float32 )
# Convierte la lista labels en una matriz que contiene las etiquetas de cada muestra
y_labels = np.array (labels).astype ( np.int32 )

# guardar los datos

# Guarda la matriz de caracteristicas con la extension .npy
np.save ( output_file_name+".npy", x_samples )
# Guarda la matriz de etiquetas con la extesion .npy
np.save ( output_file_name+"_label.npy", y_labels )

np.save ( output_file_name+"_title.npy", sample_titles )

print ( x_samples.shape, y_labels.shape)
