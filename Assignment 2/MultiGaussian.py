'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 myKNN.py

 Python code for Multivariate Gaussian Classification

'''
import numpy as np

def discriminant_function(model, prior_probability, X, mean, S) :
    prior_term = np.log(prior_probability)
    if model == 1 :
        t1 = - 0.5 * np.log(np.linalg.det(S))
        t2 = - 0.5 * np.matmul((X - mean).T , np.matmul(np.linalg.pinv(S),(X - mean)))
        return prior_term + t1 + t2

    elif model == 2 :
        t1 = - 0.5 * np.matmul((X - mean).T , np.matmul(np.linalg.pinv(S),(X - mean)))
        return prior_term + t1
    elif model == 3 :
        t1 = - 0.5 * np.log(np.linalg.det(S))
        t2 = - 0.5 * np.matmul((X - mean).T, np.matmul(np.linalg.pinv(S), (X - mean)))
        return prior_term + t1 + t2


def MultiGaussian(training_data, testing_data, Model) :
    training_feature_matrix = training_data[:, :-1]
    test_feature_matrix = testing_data[:, :-1]
    test_response = testing_data[:, -1]

    class1_training_data = training_data[training_data[:,-1] == 1]
    class2_training_data = training_data[training_data[:,-1] == 2]
    class1_training_feature_matrix = class1_training_data[:,:-1]
    class2_training_feature_matrix = class2_training_data[:,:-1]

    mean_class1 = np.reshape(np.mean(class1_training_feature_matrix, axis=0),(8,1))
    mean_class2 = np.reshape(np.mean(class2_training_feature_matrix, axis=0),(8,1))
    print("Mean vector for class 1 : ", mean_class1)
    print("Mean vector for class 2 :", mean_class2)
    print()

    prior_probability_class1 = len(class1_training_data) / len(training_data)
    prior_probability_class2 = len(class2_training_data) / len(training_data)
    print("Prior Probability of class 1: ", prior_probability_class1)
    print("Prior Probability of class 2: ", prior_probability_class2)
    print()

    #S1 = np.cov((class1_training_feature_matrix).T)
    #S2 = np.cov((class2_training_feature_matrix).T)

    S1 = np.zeros((8,8))
    S2 = np.zeros((8,8))

    for i in range(len(class1_training_feature_matrix)):
        x1 = np.reshape(class1_training_feature_matrix[i], (8,1)) - np.reshape(mean_class1,(8,1))
        S1 += np.matmul(x1, x1.T)
    S1 /= len(class1_training_feature_matrix)
    #print(S1)

    for i in range(len(class2_training_feature_matrix)):
        x2 = np.reshape(class2_training_feature_matrix[i], (8,1)) - np.reshape(mean_class2,(8,1))
        S2 += np.matmul(x2, x2.T)
    S2 /= len(class2_training_feature_matrix)
    #print(S2)

    S = S1 * prior_probability_class1 + S2 * prior_probability_class2
    #print(S)

    alpha1 = 0
    alpha2 = 0
    for i in range(8):
        for t in range(len(class1_training_feature_matrix)):
            alpha1 += (class1_training_feature_matrix[t][i] - mean_class1[i])

    alpha1 /= len(class1_training_feature_matrix)*8

    for j in range(8):
        for t in range(len(class2_training_feature_matrix)):
            alpha2 += (class2_training_feature_matrix[t][j] - mean_class2[j])

    alpha2 /= len(class2_training_feature_matrix)*8

    S31 = alpha1 * np.identity(8)
    S32 = alpha2 * np.identity(8)

    if Model == 1 :
        print("Covariance matrix of class 1 :", S1)
        print("Covariance matrix of class 2: ", S2)
        prediction_array = []
        for instance in test_feature_matrix :
            g1 = discriminant_function(1,prior_probability_class1,np.reshape(instance,(8,1)),mean_class1,S1)
            g2 = discriminant_function(1,prior_probability_class2,np.reshape(instance,(8,1)),mean_class2,S2)
            if g1 >= g2 :
                prediction_array.append(1)
            else :
                prediction_array.append(2)

        error = 0
        for i in range(len(test_response)) :
            if prediction_array[i] != test_response[i] :
                error += 1

        error_rate = error/len(prediction_array)
        print("Error rate for Model 1 is : ", error_rate)

    elif Model == 2:
        print("Shared Covariance between class1 and class 2 : ", S)
        prediction_array = []
        for instance in test_feature_matrix:
            g1 = discriminant_function(2, prior_probability_class1, np.reshape(instance,(8,1)), mean_class1, S)
            g2 = discriminant_function(2, prior_probability_class2, np.reshape(instance,(8,1)), mean_class2, S)
            if g1 > g2:
                prediction_array.append(1)
            else:
                prediction_array.append(2)

        error = 0
        for i in range(len(test_response)):
            if prediction_array[i] != test_response[i] :
                error += 1

        error_rate = error / len(prediction_array)
        print("Error rate for Model 2 is : ", error_rate)

    elif Model == 3:
        prediction_array = []
        print("Alpha 1 :", alpha1)
        print("Alpha 2 :", alpha2)
        for instance in test_feature_matrix:
            g1 = discriminant_function(3, prior_probability_class1, np.reshape(instance,(8,1)), mean_class1, S31)
            g2 = discriminant_function(3, prior_probability_class2, np.reshape(instance,(8,1)), mean_class2, S32)
            if g1 > g2:
                prediction_array.append(1)
            else:
                prediction_array.append(2)

        error = 0
        for i in range(len(test_response)):
            if prediction_array[i] != test_response[i] :
                error += 1

        error_rate = error / len(prediction_array)
        print("Error rate for Model 3 is : ", error_rate)


for i in range(3) :
    if(i == 0) :
        trn_data = np.loadtxt("training_data1.txt", delimiter=',')
        tst_data = np.loadtxt("test_data1.txt", delimiter=',')
        print("============================================================")
        print("For test set 1 :")
        print()
        for j in range(1,4):
            print("Model : ", j)
            print("------------")
            MultiGaussian(trn_data, tst_data, j)
    elif i == 1 :
        trn_data = np.loadtxt("training_data2.txt", delimiter=',')
        tst_data = np.loadtxt("test_data2.txt", delimiter=',')
        print("============================================================")
        print("For test set 2 :")
        print()
        for j in range(1,4):
            print("Model : ", j)
            print("------------")
            MultiGaussian(trn_data, tst_data, j)
    else :
        trn_data = np.loadtxt("training_data3.txt", delimiter=',')
        tst_data = np.loadtxt("test_data3.txt", delimiter=',')
        print("============================================================")
        print("For test set 3 :")
        print()
        for j in range(1,4):
            print("Model : ", j)
            print("------------")
            MultiGaussian(trn_data, tst_data, j)