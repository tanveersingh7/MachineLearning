'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 1
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 Bayesian_learning.py

'''


import numpy as np
import math

def Bayesian_Learning(training_data, validation_data) :
    X = training_data[:, :-1]
    y = training_data[:, 100]

    X1 = validation_data[:, :-1]
    y1 = validation_data[:, 100]

    sigma = np.array([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3, 4, 5, 6])

    p1, p2 = [], []

#Getting the probability distributions for both classes from training data
    for j in range(0, 100):
        x_j = X[:, j]
        den1, den2, num1, num2, pij1, pij2 = 0, 0, 0, 0, 0, 0
        for i in range(0, 1600):
            if (y[i] == 1):
                den1 += y[i]
                num1 += x_j[i] * y[i]
            elif (y[i] == 2):
                den2 += y[i]
                num2 += x_j[i] * y[i]
        pij1 = num1 / den1
        p1.append(pij1)
        pij2 = num2 / den2
        p2.append(pij2)
        print()

    print("---------------------------------------------------------------------------------")
    p1 = np.array(p1)
    p2 = np.array(p2)
    print("Table containing probability distributions for class 1 : ", p1)
    print("Table containing probability distributions for class 2 : ", p2)
    print("---------------------------------------------------------------------------------")

#Storing the prior values for both classes for different sigma values
    c = 0
    prior1, prior2 = [], []

    for i in sigma:
        neg_sigma = i * -1
        prior1_i = 1 - math.exp(neg_sigma)
        prior1.append(prior1_i)
        prior2_i = math.exp(neg_sigma)
        prior2.append(prior2_i)
        print()
        c += 1

    prior1 = np.array(prior1)
    prior2 = np.array(prior2)
    print("Prior1 list: ", prior1)
    print("Prior2 list: ", prior2)

    print("--------------------------------------------------------")


#Getting the error values for each prior on the validation set
    error_table = []
    for k in range(len(sigma)):
        print("for sigma = ", sigma[k])
        prior1_k = prior1[k]
        prior2_k = prior2[k]
        error, error_rate = 0, 0
        for i in range(0, 200):
            prob_xc1 = 1
            prob_xc2 = 1

            for j in range(0, 100):
                prob_xc1 *= math.pow(p1[j], X1[i][j]) * math.pow(1 - p1[j], 1 - X1[i][j])
                prob_xc2 *= math.pow(p2[j], X1[i][j]) * math.pow(1 - p2[j], 1 - X1[i][j])

            posterior1 = prior1_k * prob_xc1
            posterior2 = prior2_k * prob_xc2

            if (posterior1 > posterior2):
                if (1 - y1[i] != 0):
                    error += 1
            else:
                if (2 - y1[i] != 0):
                    error += 1

        error_rate = error / 200
        print("Error rate for sigma = ", sigma[k], " is : ", error_rate)
        error_table.append(error_rate)
        print("-------------------------------------------------------------------------------------")

    #Printing the error table for classification error for different values of sigma
    error_table = np.array(error_table)
    print("Error table for different sigma values is : ", error_table)
    min_error = np.min(error_table)
    min_index = list(error_table).index(min_error)
    print("Min error is :", min_error," and index of the min error is : ", min_index)
    min_sigma = sigma[min_index]
    print("Min sigma :", min_sigma)
    pc1 = prior1[min_index]
    pc2 = prior2[min_index]


    return p1, p2, pc1, pc2