


import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
from skimage.transform import resize
from glob import glob
from matplotlib import pyplot as plt


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_layer(input_tensor, num_in, num_out, kernel_size=[3, 3], ):
    weights = weight_variable([kernel_size[0], kernel_size[1], num_in, num_out])
    bias = bias_variable([num_out])
    output_tensor = tf.nn.relu(tf.nn.conv2d(input_tensor, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
    num_in = num_out
    return output_tensor, num_in


def max_pool(x, pool_count):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'), 2*num_out, pool_count + 1


def dense_layer(c_in, num_in, num_out, pool_count):
    num_neurons = x_len // pool_count * y_len//pool_count * num_in
    W_fc1 = weight_variable([num_neurons, num_out])
    b_fc1 = bias_variable([num_out])
    h_pool2_flat = tf.reshape(c_in, [-1, num_neurons])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    return tf.nn.dropout(h_fc1, keep_prob)


def import_data(file_loc):
    files = glob(file_loc + '/**/*.tif', recursive=True)
    labels = []
    data =[]
    i = len(files)
    for file in files:
        image = plt.imread(file)
        image = resize(image, (512, 512))
        image = (image - np.amin(image))/np.amax(image)
        label = 'no_fish' not in file
        data.append(image)
        labels.append(label)
        if i % 50 == 0:
            print(i)
        i -= 1
    # Convert the int numpy array into a one-hot matrix.
    labels_np = np.array(labels).astype(dtype=np.uint8)
    labels = (np.arange(2) == labels_np[:, None]).astype(np.float32)
    return data, labels

fileloc = '/media/teddy/Stephen Dedalus/HT-triggering_data'


data, labels = import_data(fileloc)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels)

num_epochs = 3
BATCH_SIZE = 1
l_rate = .0001
kernels_in_first_layer = 32


sess = tf.InteractiveSession()


# placeholder for our data input and output
x_len = 512
y_len = 512
data_size = 512 * 512
p_count = 0
num_in = 2
num_out = kernels_in_first_layer
x = tf.placeholder(tf.float32, shape=[None, data_size])
x_image = tf.reshape(x, [-1, x_len, y_len, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)
# Get the shape of the training data.
train_size = len(train_data)
#first layer
c1, num_in = conv_layer(x_image, num_in, num_out)
p1, num_out, p_count = max_pool(c1, p_count)
# second layer
p1.get_shape()
c2, num_in = conv_layer(p1, num_in, num_out)
p2, num_out, p_count = max_pool(c2, p_count)


# dense layer
dense = dense_layer(p2, num_in, num_out, p_count)
# dropout
# softmax
weight = weight_variable([num_out, 2])
bias = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(dense, weight) + bias)

#  TRAIN
# num_epochs = FLAGS.num_epochs
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

print(str(num_epochs * train_size // BATCH_SIZE) + ' steps')
ac_list = []
for step in range(num_epochs * train_size // BATCH_SIZE):
    offset = (step * BATCH_SIZE) % train_size
    batch_data = train_data[offset:(offset + BATCH_SIZE)]
    batch_data = [np.array(i).flatten() for i in batch_data]
    batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
    if step % 500 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_data, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (step, train_accuracy))
        ac_list.append(train_accuracy)
    train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

#  Test accuracy
print("test set accuracy %g" % accuracy.eval(feed_dict={
    x: test_data, y_: test_labels, keep_prob: 1.0}))

plt.plot(ac_list)
