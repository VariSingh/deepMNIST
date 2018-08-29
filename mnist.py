import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/fashion', one_hot=True)

print(mnist)

nodes_layer1 = 500
nodes_layer2 = 500
nodes_layer3 = 500
classes = 10
batch_size = 100
epochs = 10

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

def neural_network(data):
    hidden_layer1 = {
    'weights':tf.Variable(tf.random_normal([784,nodes_layer1])),
    'biases':tf.Variable(tf.random_normal([nodes_layer1]))
    }
    hidden_layer2 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer1,nodes_layer2])),
    'biases':tf.Variable(tf.random_normal([nodes_layer2]))
    }
    hidden_layer3 = {
    'weights':tf.Variable(tf.random_normal([nodes_layer2,nodes_layer3])),
    'biases':tf.Variable(tf.random_normal([nodes_layer3]))
    }
    output_layer = {
    'weights':tf.Variable(tf.random_normal([nodes_layer3,classes])),
    'biases':tf.Variable(tf.random_normal([classes]))
    }

    layer1 = tf.add(tf.matmul(data,hidden_layer1['weights']),hidden_layer1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1,hidden_layer2['weights']),hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.add(tf.matmul(layer2,hidden_layer3['weights']),hidden_layer3['biases'])
    layer3 = tf.nn.relu(layer3)

    output = tf.add(tf.matmul(layer3,output_layer['weights']),output_layer['biases'])
    output = tf.nn.relu(output)

    return output

#predicition
prediction = neural_network(x)
#calculate cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))#we pass prediction and label to compare
#minimize cost
optimizer = tf.train.AdamOptimizer().minimize(cost)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)


    for epoch in range(epochs):
        loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            _x,_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost], feed_dict = {x:_x,y:_y})
            loss += c
            print('loss-------',loss)
        print('Epoch ',epoch, 'out of ',epochs, 'loss ',loss)
        saver.save(sess, '.\my_test_model',global_step=1000)
    correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    print('Accuracy: ',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

neural_network(x)
