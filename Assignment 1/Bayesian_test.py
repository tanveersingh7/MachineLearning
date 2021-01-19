'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 1
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 Bayesian_test.py

'''

import numpy as np
from Bayesian_learning import Bayesian_Learning
import math

training_data = np.loadtxt('training_data.txt')

validation_data = np.loadtxt('validation_data.txt')

test_data = np.loadtxt('testing_data.txt')

prob_c1, prob_c2, pc1, pc2 = Bayesian_Learning(training_data, validation_data)

def Bayesian_Testing(test_data, p1, p2, pc1, pc2) :
    print("--------------------------------------------------------------------------------------")
    X2 = test_data[:, :-1]
    y2 = test_data[:, 100]

    error = 0
    for i in range(0, 200):
        prob_xc1 = 1
        prob_xc2 = 1

        for j in range(0, 100):
            prob_xc1 *= math.pow(p1[j], X2[i][j]) * math.pow(1 - p1[j], 1 - X2[i][j])
            prob_xc2 *= math.pow(p2[j], X2[i][j]) * math.pow(1 - p2[j], 1 - X2[i][j])

        posterior1 = pc1 * prob_xc1
        posterior2 = pc2 * prob_xc2

        if (posterior1 > posterior2):
            if (1 - y2[i] != 0):
                error += 1
        else:
            if (2 - y2[i] != 0):
                error += 1

    error_rate = error / 200
    print("Error rate on the test dataset for sigma = 2 is : ", error_rate)

    print("-------------------------------------------------------------------------------------")

Bayesian_Testing(test_data, prob_c1, prob_c2, pc1, pc2)