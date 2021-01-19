'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 myKNN.py

 KNN Algorithm.

'''
import numpy as np
from scipy.spatial.distance import cdist

def myKNN(training_data, test_data, k) :
    predictions = []
    for instance in test_data :
        euclidean_distance_class_dict = {}
        training_classes = training_data[:,-1]
        num_col = test_data.shape[1]
        euclidean_distance = cdist(training_data[:,:-1], [instance[0:num_col-1]])

        for i in range(len(euclidean_distance)) :
            euclidean_distance_class_dict[euclidean_distance[i][0]] = training_classes[i]

        arr = sorted(euclidean_distance_class_dict.items())

        sorted_distances = [x[0] for x in arr]
        corresponding_classes =[ x[1] for x in arr ]


        knn_classes = []
        for i in range(k) :
            knn_classes.append(corresponding_classes[i])

        x = np.bincount(knn_classes)
        predictions.append(np.argmax(x))

    return predictions