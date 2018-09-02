import tensorflow as tf
import numpy as np

from config import optimizer, lr_rate, epochs, num_layers, model_name, batch_size, max_sequence_length, num_units
from load_preprocessed_data import get_test_data, get_train_data, get_val_data, get_embeddings

input_features_dim = 300

with tf.name_scope("data_placeholder"):
    x = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length])
    y = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_length])


def get_batch(data, batch_no):
    """
    returns data in batches of particular batch time
    :param data:
    :param batch_no:
    :return: data of certain batch
    """
    return data[batch_size*batch_no: batch_size*(batch_no+1)]


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def model_architecture():

    embedding_vector = get_embeddings()
    x_vector = tf.nn.embedding_lookup(embedding_vector, x)

    with tf.name_scope("encoder_RNN_cell"):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        length_seq = length(x_vector)

        output, state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell,x_vector,
            dtype=tf.float64,
            sequence_length=length(x_vector))
        concat_output = tf.concat(output, 2)
        print state
        print concat_output

        return concat_output, state,length_seq


def run_model():

    train_text, train_labels = get_train_data()
    num_batches = int(len(train_text)/batch_size)
    output_rnn, state_rnn,length_seq = model_architecture()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):

            for batch_no in range(num_batches):
                x_train = get_batch(train_text, batch_no)
                y_train = get_batch(train_labels, batch_no)

                out_rnn, state_rnn_out = sess.run([output_rnn, state_rnn], feed_dict={x: x_train, y:y_train})

                # print output_rnn.shape

if __name__ == "__main__":
    run_model()