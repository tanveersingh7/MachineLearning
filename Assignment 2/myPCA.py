'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 myKNN.py

 Function that implements the PCA Algorithm.

'''
import numpy as np
import matplotlib.pyplot as plt

def myPCA(data, num_principal_components) :
    feature_matrix = data[:,:-1]

    mean_vector = np.mean(feature_matrix, axis = 0)
    covariance_matrix = np.cov((feature_matrix-mean_vector).T)

    eigen_vals, eigen_vectors = np.linalg.eigh(covariance_matrix)

    eigen_dict = {}
    for i in range(len(eigen_vals)) :
        eigen_dict[eigen_vals[i]] = eigen_vectors[:,i]

    arr = sorted(eigen_dict.items(), reverse = True)

    eigen_vals_dec = np.array([x[0] for x in arr])
    eigen_vectors_dec = np.array([x[1] for x in arr])

    eigen_vec_count = num_principal_components

    eigen_vals_pc = [eigen_vals_dec[i] for i in range(num_principal_components)]

    #print("Eigen values for the principal components are :", eigen_vals_pc)
    num_features = feature_matrix.shape[1]

    matrix_w = eigen_vectors_dec[0].reshape(num_features, 1)
    for i in range(1, eigen_vec_count) :
        matrix_w = np.hstack((matrix_w, eigen_vectors_dec[i].reshape(num_features, 1)))

    return matrix_w, eigen_vals_pc

