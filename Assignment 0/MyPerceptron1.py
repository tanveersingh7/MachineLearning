import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
'''''
Problem 3.2
'''''
w0=[1,-1]
print("Vector 'w' is:",w0)
np_w = np.array(w0)


dataset = sio.loadmat('data2.mat')
print(dataset.keys())
X = np.array(dataset['X'])
print("Feature Matrix 'X' is:", X)
x1 = dataset['X'][:,0]
feature_x1=np.array(x1)
print("Column X1 is", feature_x1)
x2 = dataset['X'][:,1]
feature_x2=np.array(x2)
print("Column X2 is",feature_x2)
y = dataset['y']
output_y = np.array(y)
print("Output vector 'y' is:", output_y)

x1_p = np.array([X[i][0] for i in range(len(y)) if y[i] == 1])
x1_n = np.array([X[i][0] for i in range(len(y)) if y[i] == -1])
x2_p = np.array([X[i][1] for i in range(len(y)) if y[i] == 1])
x2_n = np.array([X[i][1] for i in range(len(y)) if y[i] == -1])

fig3 = plt.subplot(111)
fig3.scatter(x1_p,x2_p)
fig3.scatter(x1_n,x2_n)
fig3.set(xlabel='Feature X1',ylabel='Feature X2')
fig3.set_title('Decision boundary defined by initial w')


vector_a = feature_x2
vector_b = -1*(np_w[0]*vector_a)/(np_w[1])
fig3.plot(vector_a, vector_b, color='black')
plt.show()


def perceptron(X,y,w) :
    vector_w = w
    steps = 0
    for i in range(500):
        if(X[i%40].dot(vector_w)*y[i%40] <= 0) :
            vector_w = vector_w + y[i%40]*X[i%40]
            print('Vector_w updated')
            steps += 1
        i += 1
    return vector_w,steps

updated_w1, steps = perceptron(X,output_y,np_w)
print("No of steps to converge:",steps)
print('Updated w vector after perceptron algorithm is: ', updated_w1)

vector_a1 = feature_x2
vector_b1 = -1*(updated_w1[0]*vector_a)/(updated_w1[1])

fig4=plt.subplot(111)
fig4.plot(vector_a1, vector_b1, color='black')

fig4.scatter(x1_p,x2_p)
fig4.scatter(x1_n,x2_n)
fig4.set(xlabel='Feature X1',ylabel='Feature X2')
fig4.set_ylim([-1,1])
fig4.set_title('Decision boundary defined by updated w after convergence')
plt.show()


from scipy.optimize import linprog
m,n = np.shape(X)
print(np.shape(X))
X = np.hstack((X, np.ones((m,1))))
n = n+1
f = np.append(np.zeros(n), np.ones(m) )
A1 = np.hstack((X*np.tile(output_y.T,(n,1)).T, np.eye(m)))
A2 = np.hstack((np.zeros((m,n)),np.eye(m)))
A = -np.vstack((A1,A2))
b = np.append(-np.ones(m),np.zeros(m))
x = linprog(f,A,b)
w = x['x'][0:n]
print('Updated w after LP "soft" linear classifier:', w)
print(np.shape(w))

vector_a1 = feature_x2
vector_b1 = -1*(w[0]*vector_a)/(w[1])

fig5=plt.subplot(111)
fig5.plot(vector_a1, vector_b1, color='black')

fig5.scatter(x1_p,x2_p)
fig5.scatter(x1_n,x2_n)
fig5.set(xlabel='Feature X1',ylabel='Feature X2')
fig5.set_ylim([-1,1])
fig5.set_title('Decision boundary defined by updated w after LP')
fig5.savefig('Figure3.png')
plt.show()