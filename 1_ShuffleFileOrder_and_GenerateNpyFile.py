#!/usr/bin/python3.7
# -*- coding: UTF-8 -*-
import numpy as np
import os
import random
from datetime import datetime as DT
import sys

path = sys.argv[0] = "C:/Users/artur/Documents/Replicacion de articulos/Ambiente virtual TF 3.7/CNN_model/program_1/validation_samples"
dirs = os.listdir(path)

output_file_name=sys.argv[0] = "C:/Users/artur/Documents/Replicacion de articulos/Ambiente virtual TF 3.7/CNN_model/program_1/output/Example_validation_4908_TCGA_samples"

time = DT.utcnow()
seed = str(time.year) + str(time.month) + str(time.day) + str(time.hour) + str(time.minute) + str(time.microsecond)
random.seed(seed)
random.shuffle(dirs)  # shuffle input order
print (seed)

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

x_samples = np.array ( elements ).astype ( np.float32 )
y_labels = np.array (labels).astype ( np.int32 )
np.save ( output_file_name+".npy", x_samples )
np.save ( output_file_name+"_label.npy", y_labels )
np.save ( output_file_name+"_title.npy", sample_titles )

print ( x_samples.shape, y_labels.shape)
