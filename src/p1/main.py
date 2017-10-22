# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import os
batch_sizes = [25, 50, 75]
learning_rates = [0.001, 0.01, 0.1, 1]
print ("Start to run with")

# Fix learning rate with different batch size
for batch in batch_sizes:
    # Fix batch size with different learning rate
    for rate in learning_rates:
        cmd = "/anaconda2/bin/python tf_mnist_linear_m_batch_n_learning_rate_03b.py "+str(batch)+" "+str(rate)
        os.system("echo " + cmd)
        os.system(cmd)
