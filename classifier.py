"""Trains and tests NeuralNet classifier

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

from NeuralNet import NeuralNet


class Classifier:
    """Unary classifier to detect anomalous behavior

    Args:
        num_input (int): Number of input for classifier
        batch_size (int = 100): Batch size
        num_epochs (int = 10): Number of training epochs
        display (bool = False): Flag to print output
        blacklist (list = []): List of features to ignore,
            cannot be used if whitelist is being used
        whitelist (list = []): List of features to use,
            cannot be used if blacklist is being used
        normalize (bool = False): Flag to determine if data is normalized
        display_step (int = 1): How often to display epoch data during training
    """

    def __init__(self, num_input, batch_size=100, num_epochs=10, display=False,
                 blacklist=[], whitelist=[], normalize=False, display_step=1):
        """Init classifier"""

        # Network parameters
        self.l_rate = 0.001
        self.dropout_prob = 0.5
        self.reg_param = 0.01
        self.std_param = 5
        self.training_epochs = num_epochs
        self.display_step = display_step
        self.batch_size = batch_size
        self.display = display
        self.normalize = normalize

        self.blacklist = blacklist
        self.whitelist = whitelist

        assert not (self.blacklist and self.whitelist), (
            'Both whitelist and blacklist are defined'
        )

        ############################
        # TensorFlow Variables below
        ############################

        # Placeholders
        self.X = tf.placeholder('float', [None, num_input], name='X')
        self.Y = tf.placeholder('int32', [None], name='Y')
        self.keep_prob = tf.placeholder('float')

        # Cost threshold for anomaly detection
        self.cost_threshold = tf.Variable(0, dtype=tf.float32)

        # for normalization
        self.feature_min = tf.Variable(np.zeros(num_input), dtype=tf.float32)
        self.feature_max = tf.Variable(np.zeros(num_input), dtype=tf.float32)

        # Create Network
        network_sizes = [num_input, 25, 2, 25, num_input]
        activations = [tf.nn.relu, tf.nn.sigmoid, tf.nn.relu, tf.nn.sigmoid]

        self.neural_net = NeuralNet(network_sizes, activations)

        prediction = self.neural_net.create_network(self.X, self.keep_prob)

        self.cost = tf.reduce_mean(tf.square(prediction - self.X))
        self.cost += self.reg_param * self.neural_net.get_l2_loss()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.l_rate)
        self.optimizer = self.optimizer.minimize(self.cost)

        self.init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver()

        # for gpu
        self.config = tf.ConfigProto(log_device_placement=False)

    def train(self, train_file=''):
        """Trains classifier

        Args:
            train_file (str): Training file location csv formatted,
                must consist of only regular behavior
        """

        trX, trY = grab_data(train_file, self.blacklist, self.whitelist)
        training_size = len(trX)

        # normalize X
        if self.normalize:
            _min = trX.min(axis=0)
            _max = trX.max(axis=0)
            trX = normalize(trX, _min, _max)

        assert self.batch_size < training_size, (
            'batch size is larger than training_size'
        )

        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)

            costs = []

            for epoch in range(self.training_epochs):
                cost = 0
                num_costs = 0
                for i in range(0, training_size, self.batch_size):
                    # for batch training
                    upper_bound = i + self.batch_size
                    if upper_bound >= training_size:
                        upper_bound = training_size - 1

                    feed_dict = {self.X: np.atleast_2d(trX[i:upper_bound]),
                                 self.Y: np.atleast_1d(trY[i:upper_bound]),
                                 self.keep_prob: self.dropout_prob}
                    _, c = sess.run([self.optimizer, self.cost],
                                    feed_dict=feed_dict)

                    cost += c
                    num_costs += 1

                    # calculate average cost on last epoch for threshold
                    if epoch == self.training_epochs - 1:
                        costs.append(c)

                if epoch % self.display_step == 0:
                    display_str = 'Epoch {0:04} with cost={1:.9f}'
                    display_str = display_str.format(epoch+1, cost/num_costs)
                    self.print(display_str)

            # assign cost threshold
            cost_threshold = np.mean(costs) + self.std_param * np.std(costs)
            sess.run(self.cost_threshold.assign(cost_threshold))

            self.print('Threshold: ' + str(cost_threshold))

            # assign normalization values
            if self.normalize:
                sess.run(self.feature_min.assign(_min))
                sess.run(self.feature_max.assign(_max))

            self.print('Optimization Finished')

            # save model
            save_path = self.saver.save(sess, './model.ckpt')
            self.print('Model saved in file: {}'.format(save_path))

    def test(self, test_file=''):
        """Tests classifier

        Args:
            test_file (str): Testing file location csv formatted

        Returns:
            (dict): Dictionary containing the following fields
                accuracy
        """

        teX, teY = grab_data(test_file, self.blacklist, self.whitelist)

        testing_size = len(teX)

        rtn_dict = {
            'num_acc': 0,
            'num_fp': 0,
            'num_tn': 0,
            'num_fn': 0,
            'num_tp': 0
        }

        with tf.Session(config=self.config) as sess:
            self.saver.restore(sess, './model.ckpt')

            # normalize data
            if self.normalize:
                _min = self.feature_min.eval()
                _max = self.feature_max.eval()

                teX = normalize(teX, _min, _max)

            for i in range(testing_size):
                cost = sess.run(self.cost,
                                feed_dict={self.X: np.atleast_2d(teX[i]),
                                           self.keep_prob: 1.0})

                t = self.cost_threshold.eval()

                class_guess = 1 if float(cost) < t else -1

                if teY[i] == 1:
                    if class_guess == 1:
                        rtn_dict['num_tn'] += 1
                    else:
                        rtn_dict['num_fp'] += 1
                else:
                    if class_guess == -1:
                        rtn_dict['num_tp'] += 1
                    else:
                        rtn_dict['num_fn'] += 1

                if teY[i] == class_guess:
                    rtn_dict['num_acc'] += 1

            rtn_dict['accuracy'] = rtn_dict['num_acc'] / testing_size
            rtn_dict['fp_rate'] = rtn_dict['num_fp'] / (rtn_dict['num_tn'] +
                                                        rtn_dict['num_fp'])
            rtn_dict['fn_rate'] = rtn_dict['num_fn'] / (rtn_dict['num_tp'] +
                                                        rtn_dict['num_fn'])

            rtn_dict['accuracy'] *= 100
            rtn_dict['fp_rate'] *= 100
            rtn_dict['fn_rate'] *= 100
            rtn_dict['tp_rate'] = 100 - rtn_dict['fn_rate']
            rtn_dict['tn_rate'] = 100 - rtn_dict['fp_rate']

        self.print(rtn_dict)

        return rtn_dict

    def print(self, val):
        """Internal function for printing"""

        if self.display:
            print(val)


def normalize(data, _min, _max):
    """Function to normalize a dataset of features

    Args:
        data (np.ndarray): Feature matrix
        _min (list): List of minimum values per feature
        _max (list): List of maximum values per feature

    Returns:
        (np.ndarray): Normalized features of the same shape as data
    """

    return (data - _min) / (_max - _min)


def grab_data(filename, blacklist=[], whitelist=[]):
    """Returns the features of a dataset

    Args:
        filename (str): File location (csv formatted)
        blacklist (list = []): List of features to ignore,
            cannot be used if whitelist is being used
        whitelist (list = []): List of features to use,
            cannot be used if blacklist is being used

    Returns:
        (tuple of np.ndarray): Tuple consisiting of the features, X
            and the labels, Y
    """

    data = pd.read_csv(filename)

    assert not (blacklist and whitelist), (
        'Both whitelist and blacklist are defined'
    )

    names = data.columns[1:]

    if not whitelist:
        for entry in blacklist:
            data = data.drop(entry, 1)
    else:
        for name in names:
            if name not in whitelist:
                data = data.drop(name, 1)

    X = data.values[:, 1:]
    Y = data.values[:, 0]

    return X.astype(np.float), Y

if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', '-r',
                        type=str,
                        default=None,
                        help='Location of training file')
    parser.add_argument('--test_file', '-t',
                        type=str,
                        default=None,
                        help='Location of testing file')
    parser.add_argument('--batch_size', '-b',
                        type=int,
                        default=100,
                        help='Size of batch for training (mini SGD)')
    parser.add_argument('--epochs', '-e',
                        type=int,
                        default=10,
                        help='Number of training epochs')
    parser.add_argument('--whitelist', '-w',
                        type=str,
                        help='Location of the whitelist file (csv formatted)')
    parser.add_argument('--normalize', '-n',
                        action='store_true',
                        help='Flag to normalize features')

    args = parser.parse_args()

    filename = args.train_file if args.train_file else args.test_file

    whitelist = []
    if args.whitelist:
        with open(args.whitelist, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                whitelist = row

    X, Y = grab_data(filename, whitelist=whitelist)
    num_input = len(X[0])

    classifier = Classifier(num_input, args.batch_size, args.epochs,
                            display=True, whitelist=whitelist,
                            normalize=args.normalize)

    if args.train_file:
        classifier.train(args.train_file)

    if args.test_file:
        classifier.test(args.test_file)
