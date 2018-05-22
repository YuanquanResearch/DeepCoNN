'''
model_train
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.
'''

import numpy as np
import tensorflow as tf
import math
from tensorflow.contrib import learn
import datetime

import pickle
import DeepCoNN
import sys

# tf.flags.DEFINE_string("word2vec", "../data/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")

datafile = "CDs"
datafile, keep, seed = sys.argv[1:]
keep = float(keep)
seed = int(seed)
tf.flags.DEFINE_string("train_data", "../data/%s/train" % datafile, "Data for training")
tf.flags.DEFINE_string("valid_data","../data/%s/val" % datafile, " Data for validation")
tf.flags.DEFINE_string("test_data","../data/%s/test" % datafile, " Data for validation")
tf.flags.DEFINE_string("para_data", "../data/%s/para" % datafile, "Data parameters")


# ==================================================

# Model Hyperparameters
#tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 80, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", keep, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda")
tf.flags.DEFINE_float("l2_reg_V", 0, "L2 regularizaion V")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 40, "Number of training epochs ")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps ")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, step, loss, accuracy, mae = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()

    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae


def dev_step(u_batch, i_batch, uid, iid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]

if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")

    pkl_file = open(FLAGS.para_data, 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    user_length = para['user_length']
    item_length = para['item_length']
    # print user_length, item_length
    # exit(0)
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    val_length = para['val_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    with tf.Graph().as_default():

        np.random.seed(seed)
        tf.set_random_seed(seed)

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)


        with sess.as_default():
            deep = DeepCoNN.DeepCoNN(
                user_num=user_num,
                item_num=item_num,
                user_length=user_length,
                item_length=item_length,
                num_classes=1,
                user_vocab_size=vocabulary_user,
                item_vocab_size=vocabulary_item,
                embedding_size=FLAGS.embedding_dim,
                fm_k=8,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                l2_reg_V=FLAGS.l2_reg_V,
                n_latent=32)
            
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # optimizer = tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=1e-8).minimize(deep.loss)

            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)
            '''optimizer=tf.train.RMSPropOptimizer(0.002)
            grads_and_vars = optimizer.compute_gradients(deep.loss)'''
            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            if FLAGS.word2vec:
                # initial matrix with random uniform
                u = 0
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0

                        if word in vocabulary_user:
                            u = u + 1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)
                sess.run(deep.W1.assign(initW))
                initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))

                item = 0
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0
                        if word in vocabulary_item:
                            item = item + 1
                            idx = vocabulary_item[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

                sess.run(deep.W2.assign(initW))

            l = (train_length / FLAGS.batch_size) + 1
            print "train_length/batchsize: ", l
            ll = 0
            # epoch = 1
            # best_mae = 5
            # best_rmse = 5
            # train_mae = 0
            train_rmse = 0

            pkl_file = open(FLAGS.train_data, 'rb')

            train_data = pickle.load(pkl_file)

            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(FLAGS.valid_data, 'rb')
            val_data = pickle.load(pkl_file)
            val_data = np.array(val_data)
            pkl_file.close()

            pkl_file = open(FLAGS.test_data, 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_val = len(val_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size
            ll = int(len(train_data) / batch_size)

            best_val = float("inf")
            pre_val = float("inf")
            best_test = float("inf")

            marked_train = 0.0
            marked_test = 0.0

            marked_epoch = -1
            overfitting = 0

            from time import time
            for epoch in range(50):

                t1 = time()
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):
                    print "epoch %d batch num %d" % (epoch, batch_num)
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, y_batch = zip(*data_train)

                    u_batch = []
                    i_batch = []
                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)

                    t_rmse, t_mae = train_step(u_batch, i_batch, uid, iid, y_batch, batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse


                print("\nepoch %d: Evaluation after one iteration, bug:" % epoch)
                print "train: mse %.4f" % ((train_rmse / ll) ** 2)
                train_rmse = 0

                val_loss = 0
                val_mse = 0

                ll_val = int(len(val_data) / batch_size) + 1
                for batch_num in range(ll_val):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_val)
                    data_val = val_data[start_index:end_index]

                    userid_valid, itemid_valid, y_valid = zip(*data_val)
                    u_valid = []
                    i_valid = []
                    for i in range(len(userid_valid)):
                        u_valid.append(u_text[userid_valid[i][0]])
                        i_valid.append(i_text[itemid_valid[i][0]])
                    u_valid = np.array(u_valid)
                    i_valid = np.array(i_valid)

                    loss, accuracy, mae = dev_step(u_valid, i_valid, userid_valid, itemid_valid, y_valid)
                    val_loss = val_loss + len(u_valid) * loss
                    val_mse = val_mse + len(u_valid) * accuracy

                val_mse /= val_length
                val_loss /= val_length
                print ("valid: mse %.4f, loss %.4f" % (val_mse, val_loss))

                test_loss = 0
                test_mse = 0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]

                    userid_test, itemid_test, y_test = zip(*data_test)
                    u_test = []
                    i_test = []
                    for i in range(len(userid_test)):
                        u_test.append(u_text[userid_test[i][0]])
                        i_test.append(i_text[itemid_test[i][0]])
                    u_test = np.array(u_test)
                    i_test = np.array(i_test)

                    loss, accuracy, mae = dev_step(u_test, i_test, userid_test, itemid_test, y_test)
                    test_loss = test_loss + len(u_test) * loss
                    test_mse = test_mse + len(u_test) * accuracy

                test_mse /= test_length
                test_loss /= test_length
                print ("test: mse %.4f, loss %.4f" % (test_mse, test_loss))
                print "epoch %d done, time %.1f: " % (epoch, time()-t1)

                if val_mse < best_val:
                    marked_epoch = epoch
                    marked_train = (train_rmse / ll) ** 2
                    best_val = val_mse
                    marked_test = test_mse

                best_test = min(best_test, test_mse)

                if val_mse < pre_val:
                    overfitting = 0
                else:
                    overfitting += 1
                pre_val = val_mse
                if overfitting >= 3:
                    break

            print "last. epoch %d, train %.4f, val %.4f, test %.4f(%.4f)" % \
                  (marked_epoch, marked_train, best_val, marked_test, best_test)

    # print 'end'
