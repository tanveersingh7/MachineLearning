import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from MLPtrain import ReLU, softmax, one_hot_encoding

training_data = np.loadtxt('optdigits_train.txt', delimiter = ',')
validation_data = np.loadtxt('optdigits_valid.txt', delimiter = ',')

combined_data = np.vstack((training_data, validation_data))
combined_data_feature_matrix = combined_data[:,:-1]
combined_data_output_vector = combined_data[:,-1]

K=10
combined_data_output = one_hot_encoding(combined_data_output_vector, K)

W = np.load('W_matrix.npy')
V = np.load('V_matrix.npy')

weight_matrix1 = W[:-1, :]
weight_matrix1 = np.transpose(weight_matrix1)
bias1 = np.reshape(W[-1, :], (-1,1))

weight_matrix2 = V[:-1, :]
weight_matrix2 = np.transpose(weight_matrix2)
bias2 = np.reshape(V[-1, :], (-1, 1))

error = 0
for t in range(len(combined_data_feature_matrix)):
    instance = np.reshape(combined_data_feature_matrix[t], (-1, 1))
    response = np.reshape(combined_data_output[t], (-1, 1))

    # Calculate first layer
    linear_combination = np.matmul(np.hstack((bias1, weight_matrix1)), np.vstack(([1], instance)))

    # Apply ReLU
    Z = ReLU(linear_combination)

    # Calculate the final layer  and apply softmax
    prediction = np.matmul(np.hstack((bias2, weight_matrix2)), np.vstack(([1], Z)))
    prediction = softmax(prediction)

    if np.argmax(prediction) != np.argmax(response):
        error += 1

total_error_rate = error / len(combined_data_feature_matrix)
print("Combined training+validation data error rate  is  : ", total_error_rate)

Z_matrix = []
for t in range(len(combined_data_feature_matrix)):
    instance = np.reshape(combined_data_feature_matrix[t], (-1, 1))

    linear_combination = np.matmul(np.hstack((bias1, weight_matrix1)), np.vstack(([1], instance)))
    Z = ReLU(linear_combination)
    Z_matrix.append(Z.T)

Z_matrix = np.asarray(Z_matrix)
x,y,z= Z_matrix.shape
Z_matrix = np.reshape(Z_matrix,(x,z))

#2 dimensional plot
print("projection of the hidden values to 2 dimensions")
pca_2d = PCA(n_components=2)
pca_2d_projected_data = pca_2d.fit_transform(Z_matrix)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title("Projection of hidden values to first 2 principal components")
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.grid(True,linestyle='-',color='0.75')
x = pca_2d_projected_data[:,0]
y = pca_2d_projected_data[:,1]
z = combined_data_output_vector
ax.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.jet )
for i in range(len(combined_data_feature_matrix)//50) :
    ax.text(pca_2d_projected_data[i][0], pca_2d_projected_data[i][1], str(combined_data_output_vector[i]))
plt.show()

#3 dimensional plot
print("Projection of the hidden values to 3 dimensions")
pca_3d = PCA(n_components=3)
pca_3d_projected_data = pca_3d.fit_transform(Z_matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Projection of hidden values to first 3 principal components")
ax.set_xlabel("Principal component 1")
ax.set_ylabel("Principal component 2")
ax.set_zlabel("Principal component 3")
ax.grid(True,linestyle='-',color='0.75')
x = pca_3d_projected_data[:,0]
y = pca_3d_projected_data[:,1]
z = pca_3d_projected_data[:,2]
z1 = combined_data_output_vector
ax.scatter(x,y,z,s=20,c=z1, marker = 'o', cmap = cm.jet )
for i in range(len(combined_data_feature_matrix)//50) :
    ax.text(pca_2d_projected_data[i][0], pca_2d_projected_data[i][1], pca_3d_projected_data[i][2],str(combined_data_output_vector[i]))
plt.show()