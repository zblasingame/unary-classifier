# Created by Zander Blasingame
# For CAMEL at Clarkson University
# Class to contain autoencoder and methods

import tensorflow as tf


# AutoEncoder class
class AutoEncoder:

    # Constructor for autoencoder
    def __init__(self, input_size, network_parameters, subspace_index=1):
        # network_paramerts list of dicts
        # Call function for initializing weights
        self.network = self.__init_net(input_size, network_parameters)
        self.subspace_index = subspace_index

    def __init_net(self, input_size, net_params):
        def create_weights(shape, name):
            return tf.Variable(tf.random_normal(shape, stddev=0.1), name=name)

        sizes = [entry['size'] for entry in net_params]
        sizes = [input_size] + sizes
        act_functions = [entry['act_function'] for entry in net_params]

        net = [{'weights': create_weights([sizes[i], sizes[i+1]],
                                          'w' + str(i)),
                'biases': create_weights([sizes[i+1]], 'b' + str(i)),
                'act_function': act_functions[i]}
               for i in xrange(len(sizes) - 1)]

        return net

    # Method for creating network
    def create_network(self, X, keep_prob):
        def compose_func(func, a, weights, biases):
            return func(tf.matmul(a, weights) + biases)

        activation = X
        for i, entry in enumerate(self.network):
            activation = compose_func(entry['act_function'],
                                      activation,
                                      entry['weights'],
                                      entry['biases'])

            if i != len(self.network) - 1:
                activation = tf.nn.dropout(activation, keep_prob)

        return activation

    # Returns subspace vector
    def get_subspace_vector(self, X):
        def compose_func(func, a, weights, biases):
            return func(tf.matmul(a, weights) + biases)

        activation = X
        for i, entry in enumerate(self.network):
            activation = compose_func(entry['act_function'],
                                      activation,
                                      entry['weights'],
                                      entry['biases'])

            if i == self.subspace_index:
                break

        return activation

    def get_l2_loss(self):
        weights = [entry['weights'] for entry in self.network]
        weights += [entry['biases'] for entry in self.network]

        return reduce(lambda a, b: a + tf.nn.l2_loss(b), weights, 0)
