'''

 INTRODUCTION TO MACHINE LEARNING - ASSIGNMENT 3
 -----------------------------------------------

 Author: Tanveer Singh Virdi
 Date: 29 October, 2019

 hw3_Q2.py

 Driver program for Question 2

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from skimage import io
from EMG import EMG
from EMG import kmeans

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
