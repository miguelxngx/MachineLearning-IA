import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import random

from absl import app
from random import seed
from random import randrange
from csv import reader
from math import exp
from keras.utils import to_categorical
from joblib import Parallel, delayed

__errors__ = []
__temp_params__ = []
__h__ = []

#Load CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return np.array(dataset)

#print array to file
def printToFile(a, file):
    f= open("./"+file+".txt","w+")
    m = np.array(a)
    for row in a:
        for element in row:
            f.write("%d " % element)
        f.write("\n")
    f.close()

#Get y to 1 or 0
def encodeYs(samples):
    y = list()
    for sample in samples:
        if sample == 'e':
            y.append(1)
        else:
            y.append(0)
    return np.array(y)

def transform_char_column_to_int(column):
    newColumn = list()
    for value in column:
        newColumn.append(ord(value))
    return newColumn

def encode_label(column):
    labelValue = {}
    new_column = []
    x = 0
    for value in column:
        if value not in labelValue:
            labelValue[value] = x
            x = x + 1
        new_column.append(labelValue[value])
    return np.array(new_column)

def get_error(samples, params,y):
    global __errors__
    error_acum =0
    error = 0
    global __h__
    for i in range(len(samples)):
        hyp = __h__[i]
        
        if(y[i] == 1): # avoid the log(0) error
            if(hyp == 0):
                hyp = 0.0001
            error = (-1) * np.log(hyp)
        if(y[i] == 0):
            if(hyp == 1):
                hyp = 0.9999
            error = (-1) * np.log(1 - hyp)
        #print( "error %f  hyp  %f  y %f " % (error, hyp,  y[i])) 
        error_acum = error_acum + error
    mean_error_param = error_acum / len(samples)
    __errors__.append(mean_error_param)
    return mean_error_param

def calculate_h(instance, params):
    acum = 0
    for i in range(len(params)):
        acum = acum + params[i]*instance[i]
    acum = acum*(-1)
    acum = 1/(1+ exp(acum))
    return acum

def GD(samples, params, y, alpha,index):
    temp_param = params[index]
    global __h__
    acum=0
    for j in range(len(samples)):
        error = __h__[j] - y[j]
        acum = acum + error*samples[j][index]
    temp_param = params[index] - alpha*(1/len(samples))*acum

    return temp_param   
        

if __name__ == "__main__":
    mp.freeze_support()
    filename = "agaricus-lepiota.data"
    dataset = load_csv(filename)

    y = encodeYs(dataset[:,0])
    samples = np.ones(len(dataset))

    #odor
    column = dataset[:,5]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #spore print
    column = dataset[:,20]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #stalk surface bellow ring
    column = dataset[:,13]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #stalk color above ring
    column = dataset[:,14]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #habitat
    column = dataset[:,22]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #cap color
    column = dataset[:,3]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #population
    column = dataset[:,21]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #gill color
    column = dataset[:,9]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #ring number
    column = dataset[:,18]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    #ring type
    column = dataset[:,19]
    column = encode_label(column)
    encoded_column = to_categorical(column)
    samples = np.c_[samples, encoded_column]

    printToFile(samples, "test")

    params = [0]*len(samples[0])
    print(len(params))

    alpha = 0.8
    random.shuffle(samples)
    train_samples = samples[0:, :]
    train_samples = train_samples.tolist()
    test_samples = samples[4000:,:]
    epoch = 1
    __h__ = [0]*len(train_samples)
    num_cores = mp.cpu_count()
    while True:
        old_params = list(params)
        #print(params)
        #print("calculating h...")
        __h__ = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(calculate_h)(instance, params) for instance in train_samples)
        #print("calculating parameters...")
        params = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(GD)(train_samples, params, y, alpha, i) for i in range(len(params)))
        #print("calculating error...")
        print(params)
        error = get_error(train_samples,params,y)
        print("Epoch: ", epoch, " error: ", error)
        epoch = epoch + 1
        
        if(old_params == params or error < 0.0001):
            printToFile(train_samples, "Samples")
            print ("final params:")
            print (params)
            break
    plt.plot(__errors__)
    plt.show()
