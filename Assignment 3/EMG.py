'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 3
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 October, 2019

 EMG.py

 Expectation Maximization Algorithm.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from skimage import io

def kmeans(pixel_values, k, shape) :
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter = 3)
    kmeans_fit = kmeans.fit(pixel_values)
    image_compressed = kmeans_fit.cluster_centers_[kmeans_fit.labels_]
    image_compressed = np.reshape(image_compressed, (shape))
    return image_compressed

def kmeans_initialization(pixel_values, k) :
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter = 3)
    kmeans_fit = kmeans.fit(pixel_values)
    means = kmeans_fit.cluster_centers_
    labels = kmeans_fit.labels_
    means_array = np.array(means)
    covariance_array = []
    pi_values = []
    for j in range(k) :
        covariance_array.append(np.cov(pixel_values[labels == j].T))
    np.array(covariance_array)
    for j in (np.unique(labels)) :
        pi_values.append(np.sum([labels == j])/len(labels))
    np.array(pi_values)
    print("K means initializtion done for k = ",k)

    return means_array, covariance_array, pi_values

def find_responsibility_values(pixel_values, means_array, covariance_array, pi_values, k) :
    responsibility_array = []
    for j in range(k) :
        gaussian_pdf = multivariate_normal.pdf(pixel_values, mean=means_array[j], cov=covariance_array[j])
        responsibility_array.append(pi_values[j] * gaussian_pdf)

    responsibility_array = np.array(responsibility_array).T

    sum = np.sum(responsibility_array, axis = 1)
    sum = np.reshape(sum, (len(sum),1))
    responsibility_array = responsibility_array / sum
    return responsibility_array

def likelihood(pixel_values, means_array, covariance_array, pi_values, k) :
    likelihood = []
    for j in range(k):
        gaussian_pdf = multivariate_normal.pdf(pixel_values, mean=means_array[j], cov=covariance_array[j])
        likelihood.append(pi_values[j] * gaussian_pdf)
    likelihood = np.array(likelihood)
    log_likelihood = np.log(np.sum(likelihood, axis=0))
    total_log_likelihood = np.sum(log_likelihood)
    return total_log_likelihood

def update_pi_values(responsibility_array) :
    sum = np.sum(responsibility_array, axis = 0)
    pi_values = sum / len(responsibility_array)
    return pi_values


def update_mean_values(pixel_values, responsibility_array) :
    k_values = responsibility_array.shape[1]
    sum = np.array(np.sum(responsibility_array, axis = 0))
    new_mean = []
    for j in range(k_values) :
        num = 0
        for i in range(len(pixel_values)) :
            num += responsibility_array[i][j] * pixel_values[i]

        new_mean_j = num / sum[j]
        new_mean.append(new_mean_j)

    new_mean = np.array(new_mean)
    return new_mean

def update_cov_values(pixel_values, responsibility_array, means) :
    k_values = responsibility_array.shape[1]
    sum = np.array(np.sum(responsibility_array, axis=0))
    new_covariance = []
    for j in range(k_values) :
        num = 0
        for i in range(len(pixel_values)) :
            num += responsibility_array[i][j] * np.mat(pixel_values[i] - means[j]).T * np.mat(pixel_values[i] - means[j])

        new_covariance_j = num / sum[j]
        new_covariance.append(new_covariance_j)

    new_covariance = np.array(new_covariance)
    return new_covariance

def update_regularized_cov_values(pixel_values, responsibility_array, means, lambda_value) :
    k_values = responsibility_array.shape[1]
    sum = np.array(np.sum(responsibility_array, axis=0))
    new_covariance = []
    for j in range(k_values) :
        num = 0
        for i in range(len(pixel_values)) :
            num += responsibility_array[i][j] * np.mat(pixel_values[i] - means[j]).T * np.mat(pixel_values[i] - means[j])

        covariance_j = num / sum[j]
        new_covariance_j = lambda_value * np.identity(covariance_j.shape[0]) + covariance_j 
        new_covariance.append(new_covariance_j)

    new_covariance = np.array(new_covariance)
    return new_covariance

    

def update_label(responsibility_array) :
    labels = np.argmax(responsibility_array, axis=1)
    return labels

def EMG(filepath, k, flag) :
    img = io.imread(filepath)
    img = img / 255
    img = img[:, :, :3]
    total_pixels = img.shape[0] * img.shape[1]
    print("Total pixels in the image : ", total_pixels)
    pixel_values = np.reshape(img, (total_pixels, 3))

    #k-means initialization
    means_array, cov_array, pi_vals = kmeans_initialization(pixel_values,k)

    if flag == 0 :
        new_likelihood = likelihood(pixel_values, means_array, cov_array, pi_vals, k)
    elif flag == 1 :
        new_likelihood = 2e5
    e_step_likelihood = []
    m_step_likelihood = []
    means_values = []
    means_values.append(means_array)
    for iter in range(100):
        print("Iteration number :", iter + 1)
        # E-step:
        old_likelihood = new_likelihood
        responsibility = find_responsibility_values(pixel_values, means_array, cov_array, pi_vals, k)
        new_labels = update_label(responsibility)
        e_step_likelihood.append(old_likelihood)

        # M-step:
        pi_vals = update_pi_values(responsibility)
        means_array = update_mean_values(pixel_values, responsibility)
        if flag == 0 :
            cov_array = update_cov_values(pixel_values, responsibility, means_array)
        elif flag == 1 :
            cov_array = update_regularized_cov_values(pixel_values, responsibility, means_array, 0.000001)

        new_likelihood = likelihood(pixel_values, means_array, cov_array, pi_vals, k)
        m_step_likelihood.append(new_likelihood)
        means_values.append(means_array)

    e_step_likelihood = np.array(e_step_likelihood)
    m_step_likelihood = np.array(m_step_likelihood)
    means_values = np.array(means_values)
    print("Responsibility(h) : ", responsibility.shape, responsibility)
    print("Means(m) : ", means_array.shape, means_array)
    print("Complete log likelihood(Q) : ", m_step_likelihood.shape, m_step_likelihood)
    em_compressed_image = []
    for i in range(len(new_labels)):
        em_compressed_image.append(means_array[new_labels[i]])

    em_compressed_image = np.array(em_compressed_image)
    print(em_compressed_image.shape)
    final_em_compressed_image = np.reshape(em_compressed_image, img.shape)

    #Compressed image after 100 iterations of EM
    plt.imshow(final_em_compressed_image)
    title = "EM Compressed image"
    plt.title(title)
    plt.axis('off')
    plt.show()
        
    #Plotting the complete log likelihood for given k value
    fig = plt.subplot(111)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    fig.plot(e_step_likelihood, '-rx')
    fig.plot(m_step_likelihood, '-bo')
    fig.set(xlabel='Number of iterations', ylabel='Likelihood')
    fig.set_title('Likelihood')
    plt.show()
    
    return responsibility,means_array,m_step_likelihood


#Original image Stadium.bmp
print("Stadium.bmp")
img = io.imread("stadium.bmp")
plt.imshow(img)
plt.title("Original image Stadium.bmp")
plt.axis('off')
plt.show()

#Run the EM algo for the given k value
k = [4,8,12]
print("EM for stadium.bmp for K = ",k)
for i in k :
    print("Standard EM for stadium.bmp for K = ",i)
    h,m,Q = EMG("stadium.bmp", i, 0)
    print()

#Original image Goldy.bmp
img1 = io.imread("goldy.bmp")
print("Goldy.bmp")
plt.imshow(img1)
plt.title("Original image Goldy.bmp")
plt.axis('off')
plt.show()
img1 = img1 / 255
img1 = img1[:, :, :3]
total_pixels1 = img1.shape[0] * img1.shape[1]
print("Total pixels in the image : ", total_pixels1)
pixel_values1 = np.reshape(img1, (total_pixels1, 3))



#EM compressed image for Goldy.bmp for K = 7
print("Standard EM for Goldy.bmp for K = 7")
try :
    h1,m1,Q1 = EMG("goldy.bmp", 7, 0)
except Exception :
    print("Singular covariance matrix obtained. Error!")

print()

#K-means compressed image for Goldy.bmp for K = 7
print("Running K Means to compress Goldy.bmp for k =7")
compressed_image = kmeans(pixel_values1, 7, img1.shape)
plt.imshow(compressed_image)
title = "K Means Compressed image for Goldy.bmp for K = 7"
plt.title(title)
plt.axis('off')
plt.show()

print()
#EM compressed image for Goldy.bmp for K = 7
print("Improved EM for Goldy.bmp for K = 7")
h2,m2,Q2 = EMG("goldy.bmp", 7, 1)


