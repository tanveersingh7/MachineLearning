import numpy as np
from MLPtrain import ReLU, softmax, one_hot_encoding

def MLPtest(test_data, W, V, H) :
    testing_data = np.loadtxt(test_data, delimiter=',')
    testing_feature_matrix = testing_data[:,:-1]
    testing_output_vector = testing_data[:,-1]

    K = 10
    testing_output = one_hot_encoding(testing_output_vector, K)

    weight_matrix1 = W[:-1, :]
    weight_matrix1 = np.transpose(weight_matrix1)
    bias1 = np.reshape(W[-1, :], (-1,1))

    weight_matrix2 = V[:-1, :]
    weight_matrix2 = np.transpose(weight_matrix2)
    bias2 = np.reshape(V[-1, :], (-1, 1))

    Z_matrix = []
    error = 0
    for t in range(len(testing_feature_matrix)) :
        instance = np.reshape(testing_feature_matrix[t], (-1,1))
        response = np.reshape(testing_output[t], (-1,1))

        # Calculate first layer
        linear_combination = np.matmul(np.hstack((bias1, weight_matrix1)), np.vstack(([1], instance)))

        # Apply ReLU
        Z = ReLU(linear_combination)
        Z_matrix.append(np.reshape(Z, (1, H)))

        # Calculate the final layer  and apply softmax
        prediction = np.matmul(np.hstack((bias2, weight_matrix2)), np.vstack(([1], Z)))
        prediction = softmax(prediction)

        if np.argmax(prediction) != np.argmax(response):
            error += 1

    Z_matrix = np.reshape(Z_matrix, (len(testing_feature_matrix), H))
    total_error_rate = error / len(testing_feature_matrix)
    print("Testing error rate  is  : ", total_error_rate)
    return Z_matrix, total_error_rate