import tensorflow as tf
from AutoEncoder_v2 import AutoEncoder

sizes = [10, 2, 10, 4]
act_funcs = [tf.nn.sigmoid,
             tf.nn.relu,
             tf.nn.sigmoid,
             tf.nn.relu]
net_params = [{'size': sizes[i],
               'act_function': act_funcs[i]}
              for i in range(len(sizes))]

ae = AutoEncoder(4, net_params)
