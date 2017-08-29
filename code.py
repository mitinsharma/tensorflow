# code.py
# By Mitin Sharma

import numpy as np
import tensorflow as tf

# Model Parametrs
m = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = m * x + b

# Define loss function (sum of squares)
loss = tf.reduce_sum(tf.square(linear_model - y))

# Define Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# Training Loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) #reset values to wrong

curr_loss = float('inf')
while(curr_loss > 0.1):
  sess.run(train, {x:x_train, y:y_train})
  


