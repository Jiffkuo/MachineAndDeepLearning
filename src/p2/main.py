# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
n_hidden = [100, 200, 300, 400]
n_layers = [1, 2, 3, 4]
print ("Start to run with")

# Fix number of hidden layer nodes with differnt number of hidden layers
for layer in n_layers:
    # Single hidden layer with different number of hidden layer nodes
    for nodes in n_hidden:
        cmd = "/anaconda2/bin/python tf_mnist_dense_m_layer_n_nodes_04c.py "+str(layer)+" "+str(nodes)
        os.system("echo " + cmd)
        os.system(cmd)
