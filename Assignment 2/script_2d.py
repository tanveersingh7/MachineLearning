'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_2d.py

LDA of Optdigits dataset.
'''

import numpy as np
from myKNN import myKNN
from myLDA import myLDA

trn_data = np.loadtxt('optdigits_train.txt', delimiter = ',')
tst_data = np.loadtxt('optdigits_test.txt', delimiter = ',')

training_feature_matrix = np.array(trn_data[:,:-1])
training_response = np.array(trn_data[:,-1])
test_feature_matrix = np.array(tst_data[:,:-1])
test_response =  np.array(tst_data[:,-1])
'''
classes = np.unique(training_response)
print(classes)
num_classes = len(classes)
Sb = np.zeros((training_feature_matrix.shape[1], training_feature_matrix.shape[1]))
Sw = np.zeros((training_feature_matrix.shape[1], training_feature_matrix.shape[1]))
print(Sb.shape)
print(Sw.shape)

total_mean = np.mean(training_feature_matrix, axis =0)
print(total_mean.shape)

class_data = []
class_mean = []
for i in classes :
    data_i = trn_data[trn_data[:,-1] == i]
    class_data.append(data_i)
    feature_matrix_i = data_i[:, :-1]
    num_instances = len(data_i)
    print("Number of instances for instance i= ",i," is :", num_instances)
    mean_i = np.mean(feature_matrix_i, axis =0)
    class_mean.append(mean_i)
    print("Mean vector for Class i: ",i," is : ",mean_i)
    Si = np.zeros((training_feature_matrix.shape[1], training_feature_matrix.shape[1]))
    for instance in feature_matrix_i :
        Si = Si + np.dot((instance - mean_i), (instance - mean_i).T)
    print("Si is for i :",i," is :", Si)
    print("Si shape is :", Si.shape)
    Sw = Sw + Si

print("Sw is :", Sw)

'''
L_values = [2,4,9]

for l in L_values :
    print("=============================================================")
    print("Projection into dimension of size :", l)
    print()
    matrix_w, eigen_values = myLDA(trn_data, l)
    train_lda = training_feature_matrix.dot(matrix_w)
    test_lda = test_feature_matrix.dot(matrix_w)

    train_lda_concat = np.hstack((train_lda, training_response.reshape(1500, 1)))
    test_lda_concat = np.hstack((test_lda, test_response.reshape(297, 1)))

    k_values = [1, 3, 5]

    error_rates_dict = {}
    for x in k_values:
        print("KNN for k value :", x)
        predictions = myKNN(train_lda_concat, test_lda_concat, x)
        error = 0
        for j in range(len(predictions)):
            if (test_response[j] != predictions[j]):
                error += 1

        print("Number of errors :", error)
        error_rate = error / len(predictions)
        print("Error rate for k :", x, " is = ", error_rate)
        error_rates_dict[x] = error_rate

        print("===============================================================================")

    print(error_rates_dict)


