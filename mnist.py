import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/fashion', one_hot=True)

nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500

classes = 10

batch_size = 100

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

def neural_network():
    hidden_layer1 = {
    'weights':tf.Variable(tf.random_normal([784,nodes_layer1]))
    'biases':tf.Variable(tf.random_normal(nodes_layer1))
    }
    hidden_layer2 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2]))
    'biases':tf.Variable(tf.random_normal(nodes_layer2))
    }
    hidden_layer3 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3]))
    'biases':tf.Variable(tf.random_normal(nodes_layer3))
    }
    output_layer = {
    'weights':tf.Variable(tf.random_normal([nodes_layer3,classes]))
    'biases':tf.Variable(tf.random_normal([classes]))
    }


# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(z)
