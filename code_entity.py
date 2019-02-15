# -*- coding:utf-8 -*-
import tensorflow as tf 
from tensorflow import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator import batch_generator
from keras.layers import MaxPooling2D
from glove_embedding import glove_embedding
from attentive_reader import attentive_reader
from attention_gru import *
import numpy as np
import pickle
import time
import os

save_model = False
load_model = False
write_log = False

class Config():
    def __init__(self):
        pass

config = Config()

query_length = 126# query padding length, n
embedding_dim = 100# word embedding dim , d
dialog_length =  25# k
dialog_length2 = 25
utter_length = 92 # m
utter_length2 = 20
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1 
batch_size = 32
hidden_length = 32
training_step = 500000
learning_rate = 0.001
speaker_embedding_dim = 50
attention_length = 128

config.query_length = query_length
config.embedding_dim = embedding_dim
config.dialog_length = dialog_length
config.dialog_length2 = dialog_length2
config.utter_length = utter_length
config.utter_length2 = utter_length2
config.filter_sizes = filter_sizes
config.filter_num = filter_num
config.length_1D = length_1D 
config.window_length = window_length
config.batch_size = batch_size
config.hidden_length = hidden_length
config.training_step = training_step
config.learning_rate = learning_rate
config.speaker_embedding_dim = speaker_embedding_dim
config.attention_length = attention_length

if not os.path.exists("checkpoint/"):
    os.mkdir("checkpoint/")



#产生U,Q和labels
#此处U和Q仅仅是句子的集合，还未转化为word embedding形式
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class = data_helper('trn_cleaned_after.json', 'dev_cleaned_after.json', 'tst_cleaned_after.json')
#label的个数
num_classes = len(label2id)
print(num_classes)

#使用skipgram训练词向量
embed, word2id, id2word = glove_embedding(U + U_dev + U_tst, Q + Q_dev + Q_tst, embedding_dim)
#多加一行，代表全零的词，留给padding的位置用
embedding_matrix = tf.Variable(np.array(embed).astype(np.float32), trainable = True)
zero = tf.zeros([1, embedding_dim], tf.float32)
embedding_matrix_converted = tf.concat([embedding_matrix, zero], 0, name = "embed")

vocab_size = embedding_matrix_converted.shape.as_list()[0]
config.vocab_size = vocab_size

print("word embedding finished.")
"""
产生一个batch对象，用next()方法就可以产生一个batch的数据
在产生batch的时候会自动将U,Q处理成所需形式，传入的utter_length和query_length会对
utterance或者query最大的长度进行限制
"""
batch = batch_generator(U, Q, labels, word2id, list(label2id.values()), word_class, config)
batch_dev = batch_generator(U_dev, Q_dev,labels_dev, word2id, list(label2id.values()), word_class, config)
batch_tst = batch_generator(U_tst, Q_tst,labels_tst, word2id, list(label2id.values()), word_class, config)

labels = tf.placeholder(tf.float32, [batch_size, num_classes]) #[bs,nc]
query_id = tf.placeholder(tf.int32, [batch_size, query_length])
utter_id = tf.placeholder(tf.int32, [batch_size, dialog_length, utter_length])
utter2_id = tf.placeholder(tf.int32, [batch_size, dialog_length2, utter_length2])
query = tf.nn.embedding_lookup(embedding_matrix_converted, query_id)#Q:[bs,n,d]
utter = tf.nn.embedding_lookup(embedding_matrix_converted, utter_id) #U:[bs,k,m,d]
utter_long = tf.nn.embedding_lookup(embedding_matrix_converted, utter2_id) #U:[bs,k,m,d]
utter_long = tf.reshape(utter_long, [batch_size, dialog_length2 * utter_length2, embedding_dim])

print("shape of the query:")
print(query.shape.as_list())
print("shape of the utterance:")
print(utter.shape.as_list())

#dropout
keep_prob = tf.placeholder(tf.float32)

def raw_lstm(hidden_length, x):
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_length)
    hidden_state = tf.zeros([batch_size, hidden_length])
    current_state = tf.zeros([batch_size, hidden_length])
    state = hidden_state, current_state
    bilstm_result = []
    length = x.shape.as_list()[1]
    for i in range(length):
        output, state = lstm(x[:, i], state)
        bilstm_result.append(output)
    bilstm_result = tf.transpose(tf.stack(bilstm_result), [1, 0, 2])
    return bilstm_result

forward_lstm = raw_lstm(hidden_length, utter_long)
backward_lstm = raw_lstm(hidden_length, tf.reverse(utter_long, axis = [1]))
bilstm_result = tf.concat([forward_lstm, backward_lstm], axis = -1)


with tf.name_scope("get_v_layer"):
    def sim(t1, t2):
        re = tf.matmul(t1, t2, transpose_b = True) #[m,n]
        tmp1 = tf.reshape(tf.reduce_sum(tf.square(t1), axis = 1), [dialog_length*utter_length, 1])
        tmp2 = tf.reshape(tf.reduce_sum(tf.square(t2), axis = 1), [1,query_length])
        e1 = tf.matmul(tmp1, tf.ones([1, query_length]))
        e2 = tf.matmul(tf.ones([dialog_length*utter_length, 1]), tmp2)
        return 1/(1+tf.sqrt(e1+e2+(-2*re) + 0.01))

    #tf.unstack
    tmp_utter = tf.unstack(tf.reshape(utter, [batch_size, dialog_length*utter_length, embedding_dim]))
    tmp_query = tf.unstack(query)
    s = [sim(tmp_utter[i], tmp_query[i]) for i in range(batch_size)]
    sim_w = tf.reshape(tf.stack(s), [batch_size*dialog_length*utter_length, query_length], name = "S") #S for cal

    utter_attention = tf.Variable(tf.truncated_normal([query_length, embedding_dim], -1, 1)) #A

    utter2 = tf.reshape(tf.matmul(sim_w, utter_attention), [batch_size, dialog_length, utter_length, embedding_dim],name = "U2") 

    v = tf.concat([tf.expand_dims(utter, -1), tf.expand_dims(utter2, -1)], -1, name = "V")
    #v = tf.reshape(utter, [batch_size, dialog_length, utter_length, embedding_dim, 1], name = "v")

with tf.name_scope("conv2d_layer"):
    conv_output = []
    for filter_size in filter_sizes:
        output = TimeDistributed(
            Conv2D(filter_num, (filter_size, embedding_dim)
            , padding = "VALID"
            , activation="relu"))(v)
        tmp = TimeDistributed(MaxPooling2D((utter_length-filter_size+1, 1), padding = "VALID"))(output)
        conv_output.append(tf.nn.dropout(tmp, keep_prob))
dialog = tf.reshape(tf.concat(conv_output, axis=-1),[batch_size, dialog_length, 4*filter_num], name="D") #D

# Normalization towards dialog
dia_mean, dia_var = tf.nn.moments(dialog, [2])
dia_mean = tf.expand_dims(dia_mean, 2)
dia_var = tf.expand_dims(dia_var, 2)
dialog = (dialog - dia_mean) / tf.sqrt(dia_var + 10 ** -6)


with tf.name_scope("dialog_attention_layer"):
    query2 = Conv1D(length_1D, window_length, activation = "relu", name = "Q2")(query) #Q2
    dialog2 = Conv1D(length_1D, window_length, activation = "relu", name = "D2")(dialog) #D2
    dialog_attention = tf.matmul(query2, dialog2, transpose_b = True, name = "P") #P
    da_column =tf.reshape(tf.reduce_sum(dialog_attention, axis = -1), [batch_size, query_length - window_length + 1, 1],name="pc") #pc
    da_row = tf.reshape(tf.reduce_sum(dialog_attention, axis = -2), [batch_size, 1, dialog_length - window_length + 1],name="pr") #pr

with tf.name_scope("concat_layer"):
    hd = Bidirectional(LSTM(hidden_length))(dialog)
    hq = Bidirectional(LSTM(hidden_length))(query)
    hd = tf.nn.dropout(hd, keep_prob)
    hq = tf.nn.dropout(hq, keep_prob)
    r = attentive_reader(bilstm_result, hq, hidden_length, attention_length, batch_size)
    aq = tf.squeeze(tf.matmul(da_column, query2, transpose_a = True), name = "aq")
    ad = tf.squeeze(tf.matmul(da_row, dialog2), name= "ad")
    hidden_layer = tf.concat([hd, hq, ad, aq, r], -1, name="hidden_layer") 
    hidden_size = int(hidden_layer.shape[1])

print(hidden_size)

"""
这一部分是之前调试的时候感觉可能出问题的部分，所以手写一个softmax层并且加了对prediction的标准化
"""
W = tf.Variable(tf.truncated_normal([hidden_size, num_classes], -1, 1))
b = tf.Variable(tf.truncated_normal([batch_size, num_classes], -1, 1))
prediction = tf.matmul(hidden_layer, W) + b
res = tf.argmax(prediction, axis = 1)

mean = tf.reshape(tf.reduce_mean(prediction, axis = 1), [-1, 1])
var = tf.reshape(tf.reduce_mean((prediction - mean) ** 2, axis = 1), [-1, 1])
prediction = (prediction - mean) / tf.sqrt(var)

correct_prediction = tf.equal(tf.argmax(prediction,axis = 1),tf.argmax(labels,axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

if write_log:
    if not os.path.exists("log/"):
        os.mkdir("log/")
    ummary_waiter = tf.summary.FileWriter("log/",tf.get_default_graph())
    ummary_waiter.close()


saver = tf.train.Saver(max_to_keep = 4)

with tf.name_scope("optimizer_and_loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    loss = loss + reg
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

if save_model:
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    savename="checkpoint/"+now+'/'
    if not os.path.exists(savename):
        os.mkdir(savename)

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary = tf.summary.merge_all()

epoch_size = int(len(batch.train_u) / batch_size)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if load_model:
        saver.restore(sess, "./checkpoint/2018-08-16-10_09_47/-16200")
    if save_model:
        writer = tf.summary.FileWriter(savename, sess.graph)
    for i in range(training_step):
        u, q, y, u2 = batch.next()
        _, los, acc, summary, r, pred, dia, emb = sess.run([optimizer, loss, accuracy, merged_summary, res, prediction, dialog, embedding_matrix], feed_dict = {query_id:q, utter_id:u, labels:y, utter2_id:u2, keep_prob:0.5})
        print("loss of step", i, los, "accuracy:", acc)
        if save_model:
            writer.add_summary(summary, i)
        if i % epoch_size == 0:
            index = list(range(0, len(batch.train_u)))
            np.random.shuffle(index)
            batch.train_u = np.array(batch.train_u)[index]
            batch.train_q = np.array(batch.train_q)[index]
            batch.train_y = np.array(batch.train_y)[index]
        if i % 300 == 0:
            if save_model:
                saver.save(sess, savename, global_step = i)
            acc_sum = 0
            los_sum = 0
            loops = int(len(batch_dev.train_u) / batch_size)
            for _ in range(loops):
                u, q, y, u2 = batch_dev.next()
                acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1, utter2_id:u2})
                acc_sum += acc
                los_sum += los
            acc = float(acc_sum) / loops
            los = float(los_sum) / loops
            print("===== accuracy for dev set: ", acc, "loss", los, "========")

            acc_sum = 0
            los_sum = 0
            loops = int(len(batch_tst.train_u) / batch_size)
            for _ in range(loops):
                u, q, y, u2= batch_tst.next() 
                acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1, utter2_id:u2})
                acc_sum += acc
                los_sum += los
            acc = float(acc_sum) / loops
            los = float(los_sum) / loops
            print("===== accuracy for test set: ", acc, "loss", los, "========")

