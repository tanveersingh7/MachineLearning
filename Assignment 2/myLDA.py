'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 myKNN.py

 Function that implements the LDA Algorithm.

'''

import numpy as np

def myLDA(data, num_principal_components) :
    feature_matrix = np.array(data[:, :-1])
    response = np.array(data[:, -1])
    classes = np.unique(response)
    num_classes = len(classes)
    Sb = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))
    Sw = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))

    total_mean = np.mean(feature_matrix, axis=0)
    class_data = []
    class_mean = []
    for i in classes:
        data_i = data[data[:, -1] == i]
        class_data.append(data_i)
        feature_matrix_i = data_i[:, :-1]
        num_instances = len(data_i)
        #print("Number of instances for instance i= ", i, " is :", num_instances)
        mean_i = np.mean(feature_matrix_i, axis=0)
        class_mean.append(mean_i)
        #print("Mean vector for Class i: ", i, " is : ", mean_i)
        Si = np.zeros((feature_matrix.shape[1], feature_matrix.shape[1]))
        for instance in feature_matrix_i:
            Si = Si + np.dot((instance - mean_i).T, (instance - mean_i))
        #print("Si is for i :", i, " is :", Si)
        #print("Si shape is :", Si.shape)
        Sw = Sw + Si
        Sb = Sb + (num_instances * np.dot((mean_i - total_mean).T, (mean_i - total_mean)))

    #eigen_vals, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.pinv(Sw + (10 ** -6) * np.eye(64)), Sb))
    eigen_vals, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.pinv(Sw ), Sb))

    eigen_dict = {}
    for i in range(len(eigen_vals)):
        eigen_dict[eigen_vals[i]] = eigen_vectors[:, i]

    arr = sorted(eigen_dict.items(), reverse=True)

    eigen_vals_dec = np.array([x[0] for x in arr])
    eigen_vectors_dec = np.array([x[1] for x in arr])
    #print(eigen_vectors_dec[0].shape)

    eigen_vec_count = num_principal_components

    eigen_vals_pc = [eigen_vals_dec[i] for i in range(num_principal_components)]

    num_features = feature_matrix.shape[1]

    matrix_w = eigen_vectors_dec[0].reshape(num_features, 1)
    for i in range(1, eigen_vec_count) :
        matrix_w = np.hstack((matrix_w, eigen_vectors_dec[i].reshape(num_features, 1)))

    return matrix_w, eigen_vals_pc
