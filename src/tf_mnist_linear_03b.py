# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
#from IPython.display import clear_output, Image, display, HTML

# Common imports
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = "<stripped %d bytes>"%size
    return strip_def

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

print ('Train data shape: ', X_train.shape)
print ('Train labels shape: ', y_train.shape)
print ('Test data shape: ', X_test.shape)
print ('Test labels shape: ', y_test.shape)


n_inputs = 28*28  # MNIST
n_outputs = 10

reset_graph()

#X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
#y = tf.placeholder(tf.int64, shape=(None), name="y")
X = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.int64, [None])
W = tf.Variable(tf.zeros([n_inputs, n_outputs]))
b = tf.Variable(tf.zeros([n_outputs]))

with tf.name_scope("dnn"):
    #logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    logits = tf.nn.softmax(tf.matmul(X, W) + b)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

#learning_rate = 1
#with tf.name_scope("train"):
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#   training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_sizes = [25, 50, 75]
learning_rates = [0.001, 0.01, 0.1, 1]
n_batches = 50

z = tf.placeholder(tf.float32)
avg = tf.reduce_mean(z)

# Iteration learning rate and batch size
with tf.Session() as sess:
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            # create new train node with updated learning rate
            with tf.name_scope("train"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                training_op = optimizer.minimize(loss)
            print("Batch size = ", batch_size, " and Learning rate = ", learning_rate)
            init.run()
            iter_acc_name = "accuracy(batch_size:'%s', learning_rate:'%s')" %(batch_size,learning_rate)
            iter_loss_name = "loss(batch_size:'%s', learning_rate:'%s')" %(batch_size,learning_rate)
            traintest_accuracy = tf.summary.scalar(iter_acc_name, accuracy)
            merged = tf.summary.merge([traintest_accuracy])
            train_writer = tf.summary.FileWriter("/tmp/tkuo/tensorboard/train", sess.graph)
            test_writer = tf.summary.FileWriter("/tmp/tkuo/tensorboard/test", sess.graph)
            loss_writer = tf.summary.FileWriter("/tmp/tkuo/tensorboard/loss", sess.graph)
            loss_avg = tf.summary.scalar(iter_loss_name, avg)
            loss_merged = tf.summary.merge([loss_avg])

            for epoch in range(n_epochs):
                loss_values=[]
                for iteration in range(mnist.train.num_examples // batch_size):
                    X_batch, y_batch = mnist.train.next_batch(batch_size)
                    _, loss_value = sess.run([training_op,loss], feed_dict={X: X_batch, y: y_batch})
                    loss_values.append(loss_value)
                #acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
                train_result, acc_train = sess.run([merged, accuracy], feed_dict={X: X_batch, y: y_batch})
                #acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
                test_result, acc_test = sess.run([merged, accuracy], feed_dict={X: mnist.test.images, y: mnist.test.labels})
                #loss_value = np.mean(np.array(loss_values))
                loss_value, loss_result = sess.run([avg, loss_merged], feed_dict={z: loss_values})
                train_writer.add_summary(train_result, epoch)
                test_writer.add_summary(test_result, epoch)
                loss_writer.add_summary(loss_result, epoch)
                print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test, "Loss value:", loss_value)
    save_path = saver.save(sess, "./my_model_final.ckpt")

show_graph(tf.get_default_graph())
