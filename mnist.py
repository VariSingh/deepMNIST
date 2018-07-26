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

    layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    layer1 = tf.nn.rely(layer1)

    layer2 = tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['biases'])
    layer2 = tf.nn.rely(layer2)

    layer3 = tf.add(tf.matmul(layer2,hidden_layer3['weights']),hidden_layer3['biases'])
    layer3 = tf.nn.rely(layer3)

    output = tf.add(tf.matmul(layer3,output_layer['weights']),output_layer['biases'])
    output = tf.nn.rely(output)

    return output


# print(test)
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     z = sess.run(test)
#     print(z)
