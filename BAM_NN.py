import matplotlib.pyplot as plt   # pip install matplotlib
import multiprocessing            # pip install multiprocessing
import numpy as np                # pip install numpy
import tensorflow as tf           # pip install tensorflow-gpu==1.15
from keras.datasets import mnist  # pip install pip install keras
###############################
np.random.seed(1000)
tf.set_random_seed(1000)
###############################
width = 28
height = 28
batch_size = 10
cicle_num = 300
graph = tf.Graph()
###############################
(X_train, Y_train), (X_test, Y_test) = mnist.load_data('C:/mnist/t10k-images.idx3-ubyte')
###############################
X_source = X_train[0:50]
Y_source = Y_train[0:50]
X_source = X_source[:, :, :, np.newaxis]
Y_source = Y_source[:, np.newaxis]
X_dest = X_source.copy()
np.random.shuffle(X_dest)
###############################
def encoder(encoder_input):
    conv1 = tf.layers.conv2d(inputs=encoder_input, filters=32, kernel_size=(3, 3),
                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=tf.nn.tanh)

    conv_output = tf.contrib.layers.flatten(conv1)

    d_layer_1 = tf.layers.dense(inputs=conv_output, units=1024, activation=tf.nn.tanh)

    code_layer = tf.layers.dense(inputs=d_layer_1, units=1024, activation=tf.nn.tanh)

    return code_layer

###############################
def decoder(code_sequence, bs):
    d_layer_1 = tf.layers.dense(inputs=code_sequence, units=1024, activation=tf.nn.tanh)

    code_output = tf.layers.dense(inputs=d_layer_1, units=(height - 2) * (width - 2) * 3, activation=tf.nn.tanh)

    deconv_input = tf.reshape(code_output, (bs, height - 2, width - 2, 3))
    deconv1 = tf.layers.conv2d_transpose(inputs=deconv_input, filters=3, kernel_size=(3, 3),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         activation=tf.sigmoid)

    output_batch = tf.cast(tf.reshape(deconv1, (bs, height, width, 3)) * 255.0, tf.uint8)

    return deconv1, output_batch

###############################
def create_batch(l):
    X = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    Y = np.zeros((batch_size, height, width, 3), dtype=np.float32)

    if l < X_source.shape[0] - batch_size:
        tmax = l + batch_size
    else:
        tmax = X_source.shape[0]

    for k, image in enumerate(X_source[l:tmax]):
        X[k, :, :, :] = image / 255.0

    for k, image in enumerate(X_dest[l:tmax]):
        Y[k, :, :, :] = image / 255.0

    return X, Y

###############################
with graph.as_default():
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False)

    with tf.device('/gpu:0' if True else '/cpu:0'):
        input_images = tf.placeholder(tf.float32, shape=(None, height, width, 3))
        output_images = tf.placeholder(tf.float32, shape=(None, height, width, 3))
        t_batch_size = tf.placeholder(tf.int32, shape=())
        code_layer = encoder(encoder_input=input_images)
        deconv_output, output_batch = decoder(code_sequence=code_layer,
                                              bs=t_batch_size)

        loss = tf.nn.l2_loss(output_images - deconv_output)
        learning_rate = tf.train.exponential_decay(learning_rate=0.00025, global_step=global_step,
                                                   decay_steps=int(X_source.shape[0] / (2 * batch_size)),
                                                   decay_rate=0.9, staircase=True)

        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_step = trainer.minimize(loss)

###############################
def prediction(X, bs=1):
    feed_dict = {
        input_images: X.reshape((1, height, width, 3)) / 255.0,
        output_images: np.zeros((bs, height, width, 3), dtype=np.float32),
        t_batch_size: bs
    }

    return session.run([output_batch], feed_dict=feed_dict)[0]

###############################
if __name__ == '__main__':

    config = tf.ConfigProto(intra_op_parallelism_threads=multiprocessing.cpu_count(),
                            inter_op_parallelism_threads=multiprocessing.cpu_count(),
                            allow_soft_placement=True,
                            device_count={'CPU': 1,
                                          'GPU': 1 if True else 0})

    session = tf.InteractiveSession(graph=graph, config=config)

    tf.global_variables_initializer().run()

    for i in range(cicle_num):
       total_loss = 0.0

       for t in range(0, X_source.shape[0], batch_size):
           X, Y = create_batch(t)

           feed_dict = {
               input_images: X,
               output_images: Y,
               t_batch_size: batch_size
           }

           _, t_loss = session.run([training_step, loss], feed_dict=feed_dict)
           total_loss += t_loss

       print('Цикл обучения {} из {} - Неточности: {}'. format(i + 1, cicle_num, total_loss / float(X_train.shape[0])))

    for i in range(20):
        restored_images = np.zeros(shape=(2, height, width, 3), dtype=np.uint8)
        restored_images[0, :, :, :] = X_source[i]

        predicted = prediction(restored_images[0])[0]

        fig, ax = plt.subplots()
        ax.imshow(predicted)
        plt.show()
