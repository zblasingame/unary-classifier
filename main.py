# Created by Zander Blasingame
# For CAMEL at Clarkson University
# Enter `python main.py --help` for documentation


import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import custom code
from utils import parse_svm
from utils import find_subspace_bounds
from utils import check_bounds
# from AutoEncoder import AutoEncoder
from AutoEncoder_v2 import AutoEncoder


# function to find bouds
def find_bounds(dists):
    stddev = np.std(np.asarray(dists))
    mean = np.mean(dists)

    return mean - 2*stddev, mean + 2*stddev


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    action='store_true',
                    help='Flag to train neural net on dataset')
parser.add_argument('--plot_off',
                    action='store_true',
                    help='Flag to turn off plotting of testing')
parser.add_argument('--testing_off',
                    action='store_true',
                    help='Flag to turn off testing')
parser.add_argument('--train_file',
                    type=str,
                    help='Location of training file')
parser.add_argument('--test_file',
                    type=str,
                    help='Location of testing file')
parser.add_argument('dir',
                    nargs='?',
                    default=None,
                    help='Directory to find training and testing data')
args = parser.parse_args()

if not args.train and args.testing_off:
    print 'Error: Invalid parameters'
    sys.exit()


# training and testing svm data
if args.dir != None:
    train_file_path = '{}/train.svm'.format(args.dir)
    test_file_path = '{}/test.svm'.format(args.dir)
else:
    train_file_path = args.train_file
    test_file_path = args.test_file

if args.train:
    trX, trY = parse_svm(train_file_path)
    training_size = len(trX)

if not args.testing_off:
    teX, teY = parse_svm(test_file_path)
    testing_size = len(teX)

# Network parameters
learning_rate = 0.001
reg_param = 0
training_epochs = 4
display_step = 1

num_of_inputs = len(trX[0]) if args.train else len(teX[0])
subspace_size = 2

# New Autoencoder v2
# sizes = [2, num_of_inputs]
# sizes = [2, num_of_inputs]
# act_funcs = [tf.nn.sigmoid, tf.nn.sigmoid]
sizes = [25, 2, 25, num_of_inputs]
act_funcs = [tf.nn.relu, tf.nn.sigmoid, tf.nn.relu, tf.identity]

net_params = [{'size': sizes[i],
               'act_function': act_funcs[i]}
              for i in range(len(sizes))]

ae = AutoEncoder(num_of_inputs, net_params, 0)

X = tf.placeholder('float', [None, num_of_inputs])
keep_prob = tf.placeholder('float')
bounds = tf.Variable([1, -1], name='bounds', dtype=tf.float32)
sigma = tf.Variable([0, 0], name='sigma', dtype=tf.float32)
xi = tf.Variable([0, 0], name='xi', dtype=tf.float32)
mu = tf.Variable([0, 0], name='mu', dtype=tf.float32)

prediction = ae.create_network(X, keep_prob)
cost = tf.reduce_mean(tf.square(prediction - X)) + reg_param * ae.get_l2_loss()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
diff_vector_op = tf.sqrt(tf.reduce_sum(tf.square(prediction - X)))

subspace_vector = ae.get_subspace_vector(X)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op) if args.train else saver.restore(sess, 'model.ckpt')

    if args.train:
        dists = []
        ss_dists = {'x': [], 'y': []}
        vec_length = 0
        for epoch in xrange(training_epochs):
            avg_cost = 0

            for i in xrange(training_size):
                feed_X = np.atleast_2d(trX[i])

                _, c = sess.run([optimizer, cost], feed_dict={X: feed_X,
                                                              keep_prob: 1.0})
                avg_cost += c / training_size

                if epoch == training_epochs-1:
                    dists.append(sess.run([diff_vector_op],
                                          feed_dict={X: feed_X,
                                                     keep_prob: 1.0}))

                    gamma = sess.run(subspace_vector,
                                     feed_dict={X: feed_X,
                                                keep_prob: 1.0})

                    ss_dists['x'].append(gamma[0, 0])
                    ss_dists['y'].append(gamma[0, 1])

            if epoch % display_step == 0:
                print 'Epoch: {0:03} with cost={1:.9f}'.format(epoch+1,
                                                               avg_cost)

        # calculate error bounds
        sess.run(bounds.assign(find_bounds(dists)))

        # calculate subspace bounds
        _sigma, _xi, _mu = find_subspace_bounds(zip(ss_dists['x'],
                                                    ss_dists['y']))

        sess.run(sigma.assign(_sigma))
        sess.run(xi.assign(_xi))
        sess.run(mu.assign(_mu))

    # Save model
    if args.train:
        save_path = saver.save(sess, 'model.ckpt')
        print 'Model saved in file: {}'.format(save_path)

    # Test accuracy of data
    if not args.testing_off:
        accCount = 0
        pos_size = 0
        neg_size = 0
        false_pos_count = 0
        false_neg_count = 0

        error_dicts = []
        for i in xrange(testing_size):
            feed_X = np.atleast_2d(teX[i])
            label = np.atleast_2d(teY[i])

            if label == 1:
                pos_size += 1
            else:
                neg_size += 1

            outlying_factor = sess.run(diff_vector_op, feed_dict={X: feed_X,
                                                       keep_prob: 1.0})

            error_dicts.append({'OF': outlying_factor,
                                'X': feed_X,
                                'label': label})


        _bounds = bounds.eval()

        for i in xrange(len(error_dicts)):
            entry = error_dicts[i]
            guess_label = 1 if _bounds[0] <= entry['OF'] <= _bounds[1] else -1

            if guess_label == entry['label']:
                accCount += 1
            elif entry['label'] == 1:
                false_neg_count += 1
            else:
                false_pos_count += 1

        # print 'Accuracy: {}%'.format(100 * float(accCount) / testing_size)
        print '{}: {{'.format(test_file_path)
        print '\taccuracy={},'.format(100 * float(accCount) / testing_size)
        if pos_size != 0:
            print '\tfalse_neg_rate={},'.format(100 * float(false_neg_count) /  pos_size)
        if neg_size != 0:
            print '\tfalse_pos_rate={},'.format(100 * float(false_pos_count) /  neg_size)
        # print '}'

        if not args.plot_off:
            # plot outlying factor vs sample number
            plt.figure()
            good_range = np.arange(1, pos_size+1)
            bad_range = np.arange(1, testing_size-pos_size + 1)
            _good_cases = []
            _bad_cases = []

            for i in xrange(len(error_dicts)):
                if error_dicts[i]['label'] == 1:
                    _good_cases.append(error_dicts[i]['OF'])
                else:
                    _bad_cases.append(error_dicts[i]['OF'])

            good_cases = np.asarray(_good_cases)
            bad_cases = np.asarray(_bad_cases)
            plt.plot(good_range, good_cases, 'go')
            plt.plot(bad_range, bad_cases, 'ro')

            g_len = len(_good_cases)
            b_len = len(_bad_cases)
            num_of_entries = g_len if g_len >= b_len else b_len
            indicies = np.arange(1, num_of_entries)
            plt.plot(indicies, _bounds[0]*np.ones_like(indicies), 'b')
            plt.plot(indicies, _bounds[1]*np.ones_like(indicies), 'b')

            # plot subspace diff
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            # pos_cases = {'x': [], 'y': [], 'z': []}
            # neg_cases = {'x': [], 'y': [], 'z': []}
            # for i in xrange(len(error_dicts)):
            #     _gamma = sess.run(subspace_vector,
            #                       feed_dict={X: error_dicts[i]['X'],
            #                                  keep_prob: 0.5})

            #     # x, y, z = _gamma[0]
            #     # diff_avg = ((x-y)+(x-z)+(y-z))/3
            #     # print 'Avg diff for subspace plot: {}'.format(diff_avg)

            #     if error_dicts[i]['label'] == 1:
            #         pos_cases['x'].append(_gamma[0, 0])
            #         pos_cases['y'].append(_gamma[0, 1])
            #         pos_cases['z'].append(_gamma[0, 2])
            #     else:
            #         neg_cases['x'].append(_gamma[0, 0])
            #         neg_cases['y'].append(_gamma[0, 1])
            #         neg_cases['z'].append(_gamma[0, 2])

            # ax.scatter(pos_cases['x'], pos_cases['y'], pos_cases['z'], c='g')
            # ax.scatter(neg_cases['x'], neg_cases['y'], neg_cases['z'], c='r')
            # ax.set_title('Subspace Mapping of Input Vectors')

            fig = plt.figure()

            pos_cases = {'x': [], 'y': []}
            neg_cases = {'x': [], 'y': []}
            for i in xrange(len(error_dicts)):
                _gamma = sess.run(subspace_vector,
                                  feed_dict={X: error_dicts[i]['X'],
                                             keep_prob: 1.0})

                if error_dicts[i]['label'] == 1:
                    pos_cases['x'].append(_gamma[0, 0])
                    pos_cases['y'].append(_gamma[0, 1])
                else:
                    neg_cases['x'].append(_gamma[0, 0])
                    neg_cases['y'].append(_gamma[0, 1])

            plt.plot(neg_cases['x'], neg_cases['y'], 'ro')
            plt.plot(pos_cases['x'], pos_cases['y'], 'go')
            plt.title('SubSpace Mapping of Input Vectors')

            # get bounds
            _sigma = sigma.eval()
            _xi = xi.eval()
            _mu = mu.eval()

            # Plot bounds
            p1 = _mu - 2*_sigma - 2*_xi
            p2 = _mu - 2*_sigma + 2*_xi
            p3 = _mu + 2*_sigma + 2*_xi
            p4 = _mu + 2*_sigma - 2*_xi

            plt.plot([p1[0], p2[0], p3[0], p4[0], p1[0]],
                     [p1[1], p2[1], p3[1], p4[1], p1[1]],
                     'b', linewidth=2)

            accCount = check_bounds(_sigma, _xi, _mu, zip(pos_cases['x'],
                                                       pos_cases['y']))
            badCount = check_bounds(_sigma, _xi, _mu, zip(neg_cases['x'],
                                                       neg_cases['y']))

            accCount += len(neg_cases['x']) - badCount

            print '\tss_accuracy={}'.format(100 * float(accCount) / testing_size)
            # if pos_size != 0:
            #     print '\tss_false_neg_rate={},'.format(100 * float(false_neg_count) /  pos_size)
            # if neg_size != 0:
            #     print '\tss_false_pos_rate={}'.format(100 * float(false_pos_count) /  neg_size)
            print '}'


            plt.show()
