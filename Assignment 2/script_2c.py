'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 2
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 September, 2019

 script_2c.py

PCA of Optdigits dataset and plotting it in 2 dimensions.
'''

import numpy as np
import matplotlib.pyplot as plt
from myPCA import myPCA


trn_data = np.loadtxt('optdigits_train.txt', delimiter = ',')
tst_data = np.loadtxt('optdigits_test.txt', delimiter = ',')

training_feature_matrix = np.array(trn_data[:,:-1])
training_response = np.array(trn_data[:,-1])
test_feature_matrix = np.array(tst_data[:,:-1])
test_response =  np.array(tst_data[:,-1])

classes = np.unique(training_response)
num_classes = len(np.unique(training_response))

matrix_w, eigen_vals_pc = myPCA(trn_data, 2)

train_pca = training_feature_matrix.dot(matrix_w)
test_pca = test_feature_matrix.dot(matrix_w)

train_pca_projected = np.hstack((train_pca,training_response.reshape(1500,1)))
test_pca_projected = np.hstack((test_pca, test_response.reshape(297,1)))

train_mean = []
for i in range(num_classes) :
    train_mean_i = train_pca_projected[train_pca_projected[:,2]==i].mean(axis = 0)
    train_mean.append(train_mean_i[:2])

test_mean = []
for i in range(num_classes) :
    test_mean_i = test_pca_projected[test_pca_projected[:,2]==i].mean(axis = 0)
    test_mean.append(test_mean_i[:2])

fig, ax = plt.subplots()
ax.set_title('Training data projection')
scatter = plt.scatter(x = [instance[0] for instance in train_pca_projected], y=[instance[1] for instance in train_pca_projected], c=[instance[2] for instance in train_pca_projected], marker = 'o', cmap = plt.cm.jet, label = [c[2] for c in train_pca_projected])
for i in range(num_classes) :
    plt.annotate(classes[i], train_mean[i], horizontalalignment='center', verticalalignment='center',size=10, weight='bold')
legend1 = plt.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()


fig, ax = plt.subplots()
ax.set_title('Test data projection')
scatter = ax.scatter(x = [instance[0] for instance in test_pca_projected], y=[instance[1] for instance in test_pca_projected], c=[instance[2] for instance in test_pca_projected], marker = 'o', cmap = plt.cm.jet, label = [c[2] for c in test_pca_projected])
for i in range(num_classes) :
    plt.annotate(classes[i], test_mean[i], horizontalalignment='center', verticalalignment='center',size=10, weight='bold' )
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
ax.add_artist(legend1)
plt.show()

