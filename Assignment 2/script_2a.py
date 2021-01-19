'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_2a.py

Classification of Optdigits dataset using KNN Algorithm.
'''

import numpy as np
from myKNN import myKNN

trn_data = np.loadtxt('optdigits_train.txt', delimiter = ',')
tst_data = np.loadtxt('optdigits_test.txt', delimiter = ',')

training_feature_matrix = trn_data[:,:-1]
training_response = trn_data[:,-1]
test_feature_matrix = tst_data[:,:-1]
test_response =  tst_data[:,-1]

k_values = [1,3,5,7]

error_rates_dict = {}
for x in k_values :
    print("KNN for k value :", x)
    predictions = myKNN(trn_data, tst_data, x)
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