import numpy as np
import matplotlib.pyplot as plt


def relu_backprop(error, activation):
    error[activation < 0] = 0
    return error


def backprop(error, weight, bias, instance, prediction):
    dinstance = []
    for w in range(len(weight)):
        dinstance.append(error[w][0] * weight[w])
    dinstance = np.array(dinstance)
    dinstance = np.reshape(np.sum(dinstance, axis=0), (-1, 1))
    dweight = np.matmul(error, np.transpose(instance))
    return dweight, dinstance, error

def one_hot_encoding(response_vector, K) :
    identity = np.identity(K)
    encoded_response_vector = []
    for response in response_vector :
        encoded_response_vector.append(identity[int(response)])
    encoded_response_vector = np.array(encoded_response_vector)
    return encoded_response_vector


def softmax(forward_pass_output):
    expo = np.exp(forward_pass_output)
    s = np.sum(expo)
    return np.divide(expo,s)

def ReLU(activations):
    for activation in range(len(activations)):
        if activations[activation] < 0:
            activations[activation] = 0
    return activations

def error_rate(data, K, W, V, b1, b2) :
    feature_matrix = data[:, :-1]
    output_vector = data[:, -1]

    output = one_hot_encoding(output_vector, K)

    error = 0

    for t in range(len(feature_matrix)):
        instance = feature_matrix[t]
        instance = np.reshape(instance, (-1, 1))
        response = output[t]
        response = np.reshape(response, (-1,1))

        # Feed-Forward
        linear_combination_matrix = np.matmul(np.hstack((b1, W)), np.vstack(([1], instance)))

        # Activation function
        Z = ReLU(linear_combination_matrix)

        output_linear_combination_matrix = np.matmul(np.hstack((b2, V)), np.vstack(([1], Z)))

        output_matrix = softmax(output_linear_combination_matrix)

        if np.argmax(output_matrix) != np.argmax(response):
            error += 1

    total_error_rate = error / len(feature_matrix)
    return total_error_rate



def MLPtrain(train_data, val_data, K, H):
    print("=======================================================")
    print("H : ", H)
    print("K : ", K)

    train_data = np.loadtxt(train_data, delimiter=',')
    val_data = np.loadtxt(val_data, delimiter=',')
    training_feature_matrix = train_data[:, :-1]
    training_response = one_hot_encoding(train_data[:, -1], K)

    # Initialize weight matrices and bias weights
    W = 0.01 * np.random.standard_normal((H, training_feature_matrix.shape[1]))
    bias1 = np.random.standard_normal((H, 1))
    V = 0.01 * np.random.standard_normal((K, H))
    bias2 = np.random.standard_normal((K, 1))

    epoch_loss = []
    learning_rate = 0.005
    decay = 0.1
    print("-----------------------------")
    # Train the MLP for 80 epochs
    for epoch in range(80):
        print("Epoch : ", epoch+1)
        if epoch % 20 == 0:
            learning_rate *= decay

        losses = []
        epoch_error = 0

        Z_matrix = []

        for t in range(len(train_data)):
            delta_W = 0
            delta_bias1 = 0
            delta_V = 0
            delta_bias2 = 0

            # Reshape instance and response and add the bias term
            instance = np.reshape(training_feature_matrix[t], (-1, 1))
            response = np.reshape(training_response[t], (-1, 1))

            # Calculate first layer
            linear_combination = np.matmul(np.hstack((bias1, W)), np.vstack(([1], instance)))

            #Apply ReLU
            Z = ReLU(linear_combination)
            Z_matrix.append(np.reshape(Z, (1, H)))

            # Calculate the final layer  and apply softmax
            prediction = np.matmul(np.hstack((bias2, V)), np.vstack(([1], Z)))
            prediction = softmax(prediction)

            # Calculate error and loss
            loss = -1 * np.sum(response * np.log(prediction))
            error = prediction - response
            losses.append(loss)

            if np.argmax(response) != np.argmax(prediction):
                epoch_error += 1

            # Backpropagation

            d_V, d_Z, d_b2 = backprop(error, V, bias2, Z, prediction)
            delta_V += d_V
            delta_bias2 += d_b2

            d_Relu_Z = relu_backprop(d_Z, linear_combination)
            d_W, d_X, d_b1 = backprop(d_Relu_Z, W, bias1, instance, linear_combination)
            delta_W += d_W
            delta_bias1 += d_b1

            # Update the weight matrix
            W = W - learning_rate * delta_W
            bias1 = bias1 - learning_rate * delta_bias1
            V = V - learning_rate * delta_V
            bias2 = bias2 - learning_rate * delta_bias2

        epoch_loss.append(np.mean(losses))
        epoch_error_rate = epoch_error / len(training_feature_matrix)
        Z_matrix = np.reshape(Z_matrix, (len(training_feature_matrix), H))
        print("Epoch training error rate : ", epoch_error_rate)

    print("-------------------------------")
    training_error_rate = error_rate(train_data, K, W, V, bias1, bias2)
    print('Training error after {} epochs for h = {} : {}'.format(epoch+1 , H, training_error_rate))
    validation_error_rate = error_rate(val_data, K, W, V, bias1, bias2)
    print('Validation error after {} epochs for h = {} : {}'.format(epoch+1 , H, validation_error_rate))

    final_W_matrix = np.hstack((W, bias1)).T
    final_V_matrix = np.hstack((V, bias2)).T

    return Z_matrix, final_W_matrix, final_V_matrix, training_error_rate, validation_error_rate