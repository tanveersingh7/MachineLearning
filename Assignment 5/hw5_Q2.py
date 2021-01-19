import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from kernPercGD import kernPercGD

def meshgrid(X1,X2) :
    d = 0.02
    x, y = np.meshgrid(np.arange(X1.min() - 1, X1.max() + 1, d), np.arange(X2.min() - 1, X2.max() + 1, d))
    return x,y

np.random.seed(1) # For reproducibility
r1 = np.sqrt(np.random.rand(100, 1)) # Radius
t1 = 2*np.pi*np.random.rand(100, 1) # Angle
data1 = np.hstack((r1*np.cos(t1), r1*np.sin(t1))) # Points
np.random.seed(2) # For reproducibility
r2 = np.sqrt(3*np.random.rand(100, 1)+2) # Radius
t2 = 2*np.pi*np.random.rand(100, 1) # Angle
data2 = np.hstack((r2*np.cos(t2), r2*np.sin(t2))) # Points

data3 = np.vstack((data1, data2))
labels = np.ones((200, 1))
labels[0:100, :] = -1


fig, ax = plt.subplots()
plt.scatter(data1[:,0],data1[:,1], c='r')
plt.scatter(data2[:,0],data2[:,1], c='b')

classifier = SVC(C = 1, kernel='poly', degree = 2)
classifier.fit(data3, labels)

data3_0, data3_1 = data3[:,0], data3[:,1]
x0, x1 = meshgrid(data3_0, data3_1)

Z = classifier.predict(np.c_[x0.ravel(), x1.ravel()])
Z = np.reshape(Z, x0.shape)
plt.contourf(x0, x1, Z, alpha=0.4)
plt.xlim(x0.min(), x0.max())
plt.ylim(x1.min(), x1.max())
plt.show()