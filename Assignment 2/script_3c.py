'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_3c.py

PCA of Face Train dataset.
'''

import numpy as np
import matplotlib.pyplot as plt
from myPCA import myPCA

trn_data = np.loadtxt('face_train_data_960.txt')
tst_data = np.loadtxt('face_test_data_960.txt')

trn_data_reduced = trn_data[:5]

training_feature_matrix = np.array(trn_data_reduced[:,:-1])
training_response = np.array(trn_data_reduced[:,-1])

K = [10, 50, 100]

for i in K :
    mean_vector = np.mean(training_feature_matrix, axis=0)

    matrix_w, eigen_values = myPCA(trn_data, i)

    train_pca = training_feature_matrix.dot(matrix_w)

    reconstructed_feature_matrix = train_pca.dot((matrix_w).T)

    fully_reconstructed_feature_matrix = reconstructed_feature_matrix + mean_vector

    for instance in fully_reconstructed_feature_matrix :
        plt.imshow(np.reshape(instance, (30, 32)))
        plt.show()




