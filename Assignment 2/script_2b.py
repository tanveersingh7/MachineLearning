'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_2b.py

PCA of Optdigits dataset.
'''

import numpy as np
import matplotlib.pyplot as plt
from myKNN import myKNN
from myPCA import myPCA

trn_data = np.loadtxt('optdigits_train.txt', delimiter = ',')
tst_data = np.loadtxt('optdigits_test.txt', delimiter = ',')

training_feature_matrix = np.array(trn_data[:,:-1])
training_response = np.array(trn_data[:,-1])
test_feature_matrix = np.array(tst_data[:,:-1])
test_response =  np.array(tst_data[:,-1])

mean_vector = np.mean(training_feature_matrix, axis=0)
# print(mean_vector)

covariance_matrix = np.cov((training_feature_matrix-mean_vector).T)
# print(covariance_matrix)

eigen_vals, eigen_vectors = np.linalg.eigh(covariance_matrix)

eigen_dict = {}
for i in range(len(eigen_vals)):
    eigen_dict[eigen_vals[i]] = eigen_vectors[:, i]

arr = sorted(eigen_dict.items(), reverse=True)

eigen_vals_dec = ([x[0] for x in arr])
eigen_vectors_dec = np.array([x[1] for x in arr])

explained_variance = 90

eigen_vals_total = np.sum(eigen_vals_dec)

var_exp = [(i / eigen_vals_total) * 100 for i in eigen_vals_dec]
prop_var = [(i / eigen_vals_total) for i in eigen_vals_dec]
cum_prop_var = np.cumsum(prop_var)
#print(cum_prop_var)
cum_var_exp = np.cumsum(var_exp)
#print(cum_var_exp)

fig = plt.subplot(111)
fig.plot(cum_prop_var,'-rx')
fig.set(xlabel='Eigenvectors',ylabel='Prop. of var.')
fig.set_title('Proportion of variance explained')
plt.show()

eigen_vec_count = 0

for index, percentage in enumerate(cum_var_exp):
    if percentage > explained_variance:
        eigen_vec_count = index + 1
        break

print("Minimum number of eigenvectors that explain 90% of variance : ", eigen_vec_count)

matrix_w, eigen_vals_pc = myPCA(trn_data, eigen_vec_count)

train_pca = training_feature_matrix.dot(matrix_w)
test_pca = test_feature_matrix.dot(matrix_w)

train_pca_concat = np.hstack((train_pca,training_response.reshape(1500,1)))
test_pca_concat = np.hstack((test_pca, test_response.reshape(297,1)))

k_values = [1,3,5,7]

error_rates_dict = {}
for x in k_values :
    print("KNN for k value :", x)
    predictions = myKNN(train_pca_concat, test_pca_concat, x)
    error = 0
    for j in range(len(predictions)) :
        if(test_response[j] != predictions[j]) :
            error += 1

    print("Number of errors :", error)
    error_rate = error / len(predictions)
    print("Error rate for k :",x, " is = ", error_rate)
    error_rates_dict[x] = error_rate

    print("===============================================================================")

print(error_rates_dict)

