"""
CNN model build with Keras 
Created date : 18/12/18 
author : MOLA lIN
"""
# Desarrollo de modelo CNN
###  ----------------------------------------------------------------------
import os
os.environ ["CUDA_VISIBLE_DEVICES"] = "0"  # assign specific graphic card
os.environ ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ignore system informaion and error

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
import numpy as np
import matplotlib
matplotlib.use( "Agg" )
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from datetime import datetime as DT
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization as B_nor
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
import keras.backend as K
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score

###  ----------------------------------------------------------------------
###  Crear save_train_history para guardar los procesos de entrenamiento
###  ----------------------------------------------------------------------
class Metrics(Callback):
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        print("F1 score: ",_val_f1," - Recall: ",_val_recall," - Precision: ",_val_precision)
        logs['val_f1s']=_val_f1
        logs['val_recalls']=_val_recall
        logs['val_precisions']=_val_precision
        return
# Define la funcion para guardar el historico del entrenamiento (perdida, precision, etc.)
def save_train_history ( train_history, train, validation, name ):
    plt.plot ( train_history.history[train] )
    plt.plot ( train_history.history[validation] )
    plt.title ( 'Train History' )
    plt.ylabel ( train )
    plt.xlabel ( 'Epoch' )
    plt.legend ( ['train', 'validation'], loc = 'upper left' )
    name = name + ".png"
    plt.savefig( name, bbox_inches = "tight" )
    plt.close()

# funciones para calcular verdaderos positivos, verdaderos positivos, falsos positivos, falsos negativos
# y coeficiente de correlacion (MCC) para evaluacion del desempeño
def TP(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    return tp

def TN(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)
    return tn

def FP(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp = K.sum(y_neg * y_pred_pos)
    return fp

def FN(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fn = K.sum(y_pos * y_pred_neg)
    return fn
# Funcion para calcular el el MCC Coeficiente de Correlacion de Matthews
def MCC(y_true, y_pred): # Matthews correlation coefficient
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

###  ----------------------------------------------------------------------
###  Cargar datos de entrenamiento y su etiqueta
###  y hacer la conversión del formato de entrada
###  ----------------------------------------------------------------------
training_set = sys.argv[0] = "C:/Users/artur/Documents/Replicacion_de_articulos/Ambiente virtual TF 3.7/Modelo_CNN/program_1/output/Example_training_1228_TCGA_samples.npy" ### "Input data npy file" 
x_sample = np.load ( training_set )
x_sample = x_sample.reshape ((x_sample.shape[0], 100, 100,1)) # No. de muestras, Longitud, Ancho, No. de filtros

training_label = sys.argv[0] = "C:/Users/artur/Documents/Replicacion_de_articulos/Ambiente virtual TF 3.7/Modelo_CNN/program_1/output/Example_training_1228_TCGA_samples_label.npy" ### "Input label npy file" 
y_label = np.load ( training_label )
y_label = np_utils.to_categorical ( y_label )

out_file_name=sys.argv[0] = "C:/Users/artur/Documents/Replicacion_de_articulos/Ambiente virtual TF 3.7/Modelo_CNN/program_2/output/Example_training_1228_TCGA_samples"

# El numero de epocas para el entrenamiento de la CNN
num_epochs=100

###  ----------------------------------------------------------------------
###  Construccion del modelo convolucional
###  ----------------------------------------------------------------------
model = Sequential()

### Conv 1
model.add ( Conv2D ( filters = 64, kernel_size = ( 5, 5 ), input_shape = ( 100, 100, 1 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) ) #B_nor (Normalizacion Batch) se usa para acelerar el entrenamiento y mejorar la estabilidad
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Conv 2
model.add ( Conv2D ( filters = 64, kernel_size = ( 3, 3 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) )
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Conv 3
model.add ( Conv2D ( filters = 64, kernel_size = ( 3, 3 ) ) )
model.add ( B_nor ( axis = 2, epsilon = 1e-5 ) )
model.add ( MaxPooling2D ( pool_size = (2, 2) , padding = "same" ) )

### Fully connect
model.add ( Flatten() ) # La capa Flatten es para convertir los datos a un vector unidimensional

### Hidden layers
model.add ( Dense ( 1000, activation = "relu" ) )
model.add ( Dense ( 600, activation = "relu" ) )
model.add ( Dense ( 80, activation = "relu" ) )
model.add ( Dense ( 12, activation = "softmax" ) ) # La ultima capa es para la prediccion de clases de las muestras (12 posibles)

###  ----------------------------------------------------------------------
###  Compilar y ejecutar el modelo con entrada
###  ----------------------------------------------------------------------
Adam = Adam ( lr = 1e-4 )
model.compile ( loss = 'categorical_crossentropy', # entropia cruzada categorica para clasificacion de multiclases
                 optimizer = Adam, metrics = ['accuracy',MCC] ) # Optimizador Adam con una tasa de aprendizaje para actilizacion de pesos de las capas
                 # Metricas presicion y MCC para evalucion del rendimiento del modelo
###  ----------------------------------------------------------------------
###  Definir las funciones de callback 
###  ----------------------------------------------------------------------
# Se guarda el entrenamiento del modelo con los mejores pesos en un archivo .hdf5
best_model = "best_weights_"+out_file_name+".hdf5" 
# Se utiliza para detener el entrenamiento anticipadamente si el MCC no mejora durante un numero especifico de epocas
Estop = EarlyStopping ( monitor = "val_MCC", patience = 60, verbose = 0, mode = "max" )
# Guarda el modelo con los mejores pesos en funcion de la metrica MCC
Checkpoint = ModelCheckpoint ( best_model, monitor = "val_MCC", verbose = 0, save_best_only = True, mode = "max" )

###  ----------------------------------------------------------------------
###  Guardar el registro del entrenamiento
###  ----------------------------------------------------------------------
# Se define para crear registros de Tensorboard (Para visualizar el entrenamiento en tiempo real)
tbCallBack=TensorBoard(log_dir=out_file_name+'_logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

###  ----------------------------------------------------------------------
###  Comienzo del entrenamiento *
###  ----------------------------------------------------------------------
train_history = model.fit ( x_sample, y_label, # Se cargan los datos de entrenamiento y etiquetas
                            validation_split = 0.25, #Se utiliza para reservar un 25% de datos para la validacion
                            epochs = num_epochs, #Numero de epocas de entrenamiento
                 batch_size = 24, verbose = 1 ) # Se utiliza un lote de tamaño de 24 muestras y verbose se activa para ver progreso del entrenamiento

# Guardar la informacion detallada sobre cada epoca de entrenamiento
out = open(out_file_name+"_tranning_message", 'w')
step=0
for i in train_history.history['accuracy']:
	print('Epoch:\t%d/%d\tLoss:\t%f\tAcc:\t%f\tMCC:\t%f\tVal_Loss:\t%f\tVal_Acc:\t%f\tVal_MCC:\t%f' % (step+1, num_epochs, 
		train_history.history['loss'][step], i, train_history.history['MCC'][step], train_history.history['val_loss'][step], 
		train_history.history['val_accuracy'][step], train_history.history['val_MCC'][step]), file=out)
	step +=1
out.close()
###  ----------------------------------------------------------------------
###  Guardar el proceso de entrenamiento
###  ----------------------------------------------------------------------
final_model = out_file_name+".save" #"Modelo final"
model.save_weights ( final_model ) #Guardar los pesos sinapticos .save_weights

###  ----------------------------------------------------------------------
###  funcion para generar graficos que representan el proceso de entrenamiento, perdida vs epoca y perdidda de validadcion vs epoca
###  ----------------------------------------------------------------------
save_train_history ( train_history, "accuracy", "val_accuracy", out_file_name+"-acc" )
save_train_history ( train_history, "loss", "val_loss", out_file_name+"-loss" )

# Imprime un resumen de la arquitectura del modelo, incluido el numero de capas, parametros y formas de la entrada y salida.
model.summary()
