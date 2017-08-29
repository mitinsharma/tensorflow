# Matrix Multiplication using Tensor Flow Operation

import numpy as np

a = np.array([[1,2],[3,4.0]])
b = np.array([[5,6],[7,8.0]])

print(a)
print(b)

print(np.dot(a,b))

import tensorflow as tf
a = tf.constant(a)
b = tf.constant(b)
#convert numpy array to int
#b = tf.constant(b.astype(int))

with tf.Session() as sess:
	print(tf.matmul(a,b).eval())