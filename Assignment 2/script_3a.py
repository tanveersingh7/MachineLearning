'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_2d.py

Eigenfaces of Face Train dataset.
'''

import numpy as np
import matplotlib.pyplot as plt

trn_data = np.loadtxt('face_train_data_960.txt')
tst_data = np.loadtxt('face_test_data_960.txt')

combined_data = np.vstack((trn_data, tst_data))
combined_data_feature_matrix = np.array(combined_data[:,:-1])
combined_data_response = np.array(combined_data[:,-1])

mean_vector = np.mean(combined_data_feature_matrix, axis=0)
covariance_matrix = np.cov((combined_data_feature_matrix-mean_vector).T)

eigen_vals, eigen_vectors = np.linalg.eigh(covariance_matrix)

eigen_dict = {}
for i in range(len(eigen_vals)):
    eigen_dict[eigen_vals[i]] = eigen_vectors[:, i]

arr = sorted(eigen_dict.items(), reverse=True)

eigen_vals_dec = np.array([x[0] for x in arr])
eigen_vectors_dec = np.array([x[1] for x in arr])

eigen_vec_count = 5

eigen_faces_pc = [eigen_vectors_dec[i] for i in range(5)]

for eigen_face in eigen_faces_pc :
    plt.imshow(np.reshape(eigen_face, (30, 32)))
    plt.show()

