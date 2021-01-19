import numpy as np

def kernPercGD(X, y) :
    num_examples = X.shape[0]
    num_features = X.shape[1]

    print(X.shape)
    print(y.shape)

    alpha = np.zeros(num_examples)
    b = 0

    for j in range(20) :
        for t in range(num_examples):
            kernel_term = 0
            for i in range(num_examples):
                x_multiply = X[i,:].dot(X[t,:])
                x_multiply = x_multiply ** 2
                kernel_term += (alpha[i] * y[i] * x_multiply + b)
            final_term = kernel_term * y[t]
            if final_term <= 0:
                alpha[t] += 1
                b += y[t]

    return alpha, b


