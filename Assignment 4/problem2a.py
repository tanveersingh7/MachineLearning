import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from MLPtrain import MLPtrain
from MLPtest import MLPtest

H = [3,6,9,12,15,18]
K = 10
training_error_array = []
validation_error_array = []
min_error = 100
desired_h = 0
desired_W = []
desired_V = []

for h in H :
    Z_matrix, W, V, t_err, v_err = MLPtrain('optdigits_train.txt','optdigits_test.txt', K, h)
    training_error_array.append(t_err)
    validation_error_array.append(v_err)
    if v_err < min_error :
        min_error = v_err
        desired_h =h
        desired_W = W
        desired_V = V

np.save('W_matrix.npy', desired_W)
np.save('V_matrix.npy', desired_V)

plt.plot(H, training_error_array, label = 'training_error')
plt.plot(H, validation_error_array, label = 'validation_error')
plt.legend(['Training error', 'Validation error'], loc = 'upper right')
plt.title('Variation of training and validation error rate with H')
plt.ylabel('Error_rate')
plt.xlabel('H')
plt.show()

print("The number of hidden units(H) which give the lowest error_rate is : ", desired_h)

Z_matrix, test_err = MLPtest('optdigits_test.txt', desired_W, desired_V, desired_h)
