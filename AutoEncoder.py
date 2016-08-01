# Created by Zander Blasingame
# For CAMEL at Clarkson University
# Class to contain autoencoder and methods

import tensorflow as tf


# AutoEncoder class
class AutoEncoder:

    # Constructor for autoencoder
    def __init__(self, num_of_inputs, num_of_units=10, subspace_size=2):
        size_parameters = []
        size_parameters.append(num_of_inputs)
        size_parameters.append(num_of_units)
        size_parameters.append(subspace_size)
        size_parameters.append(num_of_units)
        size_parameters.append(num_of_inputs)

        # Call function for initializing weights
        self.weights, self.biases = self.__initialize_weights(size_parameters)

    def __initialize_weights(self, size_parameters):
        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        def getName(name, idx):
            return name+str(idx+1) if idx < len(size_parameters)-2 else 'out'

        weights = {getName('w', i): create_weights([size_parameters[i],
                                                    size_parameters[i+1]],
                                                   getName('w', i))
                   for i in range(len(size_parameters)-1)}

        biases = {getName('b', i): create_weights([size_parameters[i+1]],
                                                  getName('b', i))
                  for i in range(len(size_parameters)-1)}

        return weights, biases

    # Method for creating network
    def create_network(self, X, keep_prob):
        weights = self.weights
        biases = self.biases

        a1 = tf.nn.sigmoid(tf.matmul(X, weights['w1']) + biases['b1'])
        a1 = tf.nn.dropout(a1, keep_prob)
        a2 = tf.nn.sigmoid(tf.matmul(a1, weights['w2']) + biases['b2'])
        a2 = tf.nn.dropout(a2, keep_prob)
        a3 = tf.nn.relu(tf.matmul(a2, weights['w3']) + biases['b3'])
        a3 = tf.nn.dropout(a3, keep_prob)
        return tf.nn.sigmoid(tf.matmul(a3, weights['out']) + biases['out'])
        # return tf.matmul(a3, weights['out']) + biases['out']

    # Returns subspace vector
    def get_subspace_vector(self, X):
        weights = self.weights
        biases = self.biases

        a1 = tf.nn.sigmoid(tf.matmul(X, weights['w1']) + biases['b1'])
        return tf.nn.sigmoid(tf.matmul(a1, weights['w2']) + biases['b2'])
