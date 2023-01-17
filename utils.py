from __future__ import division, print_function
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np

import os

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature, trainDataName, testLabelName):
    import deepdish.io as ddio
    mkdir_recursive('mitdb')
    trainData = ddio.load('mitdb/'+trainDataName)
    testlabelData= ddio.load('mitdb/'+testLabelName)
    X = trainData[feature]
    # X.extend(trainData["ECG2"])
    X = np.float32(X)

    # print(X)

    # indices = [0, 2, 3, 4, 6, 7, 8, 12, 13, 33, 34, 54, 55, 67, 75, 76, 94, 96, 97, 99, 102, 117, 118, 119, 120, 121, 122, 123, 127, 128, 134, 136, 138, 139, 140, 141, 142, 143, 148, 149, 155, 159, 160, 161, 162, 163, 164, 169, 170, 176]

    # create new array with the selected indices
    # X_subset = []
    # for element in X:
    #   X_subset.append(element[indices])
    
    
    # X_subset = np.float32(X_subset)
    # print(X_subset[0])
    # print(X_subset.shape)

    y = testlabelData[feature]
    # y.extend(testlabelData["ECG2"])

    labels = list()
    for label in y:
        labels.append(label.index(1)) # get the index e.g [0,0,1,0] = 2

    labels = np.float32(labels)

    # for index, element in enumerate(labels): 
    #     if element == 3 :
    #       print(X[index])
        

    unique, counts = np.unique(labels, return_counts=True)
    # print(dict(zip(unique, counts)))
    # print("length :",len(X[0]))
    # print("example :",X[0], labels[0])


    
    
   

    # print (X, len(X))
    # print (labels, len(labels))
    return X,labels
