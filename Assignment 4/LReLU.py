from keras import backend as K
import tensorflow as tf

def LReLU(x):
    return K.tf.where(K.tf.less(x,0), 0.01*x, x)

