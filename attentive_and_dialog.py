# -*- coding:utf-8 -*-
import tensorflow as tf 
from tensorflow import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator_long import batch_generator
from keras.layers import MaxPooling2D
from glove_embedding import glove_embedding
from aoa import get_aoa_attention
from attentive_reader import attentive_reader
from attention_gru import *
import numpy as np
import pickle
import time
import os

save_model = True
load_model = False
write_log = False

query_length = 126# query padding length, n
embedding_dim = 100# word embedding dim , d
utter_length = 500 # m
dialog_length = 25
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1 
batch_size = 32
hidden_length = 64
training_step = 500000
learning_rate = 0.001
attention_length = 100

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
print("word embedding finished.")
"""
产生一个batch对象，用next()方法就可以产生一个batch的数据
在产生batch的时候会自动将U,Q处理成所需形式，传入的utter_length和query_length会对
utterance或者query最大的长度进行限制
"""
batch = batch_generator(U, Q, labels, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class)
batch_dev = batch_generator(U_dev, Q_dev, labels_dev, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class)
batch_tst = batch_generator(U_tst, Q_tst, labels_tst, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class)

#batch产生的u,q里面对应每个词是词在字典中的id，使用embedding_lookup就可以转化为词向量表示
dialog_pos = np.zeros([batch_size, dialog_length], np.int32)
labels = tf.placeholder(tf.float32, [batch_size, num_classes]) #[bs,nc]
query_id = tf.placeholder(tf.int32, [batch_size, query_length])
utter_id = tf.placeholder(tf.int32, [batch_size, utter_length])
query = tf.nn.embedding_lookup(embedding_matrix_converted, query_id)#Q:[bs,n,d]
utter = tf.nn.embedding_lookup(embedding_matrix_converted, utter_id) #U:[bs,k,m,d]
print("shape of the query:")
print(query.shape.as_list())
print("shape of the utterance:")
print(utter.shape.as_list())

#bilstm_result为双向lstm产生的所有结果，size为bs * (dialog_length * utter_length) * hidden_length
def raw_lstm(hidden_length, x):
    lstm = tf.contrib.rnn.BasicLSTMCell(hidden_length)
    hidden_state = tf.zeros([batch_size, hidden_length])
    current_state = tf.zeros([batch_size, hidden_length])
    state = hidden_state, current_state
    bilstm_result = []
    length = x.shape.as_list()[1]
    for i in range(length):
        output, state = lstm(utter[:, i], state)
        bilstm_result.append(output)
    bilstm_result = tf.transpose(tf.stack(bilstm_result), [1, 0, 2])
    return bilstm_result

forward_lstm = raw_lstm(hidden_length, utter)
backward_lstm = raw_lstm(hidden_length, tf.reverse(utter, axis = [1]))
bilstm_result = tf.concat([forward_lstm, backward_lstm], axis = -1)

#dialog
dialog = [[bilstm_result[i][j] for j in range(dialog_length)] for i in range(batch_size)]
dialog = tf.reshape(dialog, [batch_size, dialog_length, 2 * hidden_length])
print("shape of dialog:", dialog.shape.as_list())

#dropout
keep_prob = tf.placeholder(tf.float32)
"""
with tf.name_scope("concat_layer"):
	hq = Bidirectional(LSTM(hidden_length))(query)
	hq = tf.nn.dropout(hq, keep_prob)
"""

"""
Newly updated.
获取attention_over_attention中的attention向量
此处需要query过biLSTM的各个中间状态
"""
query_forward = raw_lstm(hidden_length, query)
query_backward = raw_lstm(hidden_length, tf.reverse(query, axis = [1]))
query_bilstm = tf.concat([query_forward, query_backward], axis = -1)

"""
这里是attentive_reader的工作，输入是utterance经过bilstm的各个隐藏层，以及query的bilstm最后一层
输出是utterance经过attention之后的最终输出
"""
with tf.name_scope("attention_layer"):
    r = attentive_reader(bilstm_result, query_bilstm[:, -1], hidden_length, attention_length, batch_size)
    att_gru = Attention_Gru(dialog, dialog)
    d = att_gru.get_result(return_sequences = True)
    att_gru2 = Attention_Gru(query_bilstm, query_bilstm)
    q = att_gru2.get_result(return_sequences = True)
    att_gru3 = Attention_Gru(d, q)
    res = att_gru3.get_result()
    hidden_layer = tf.concat([r, q[:, -1], res], axis = -1)
    print("hidden_layer", hidden_layer.shape.as_list())


hidden_size = hidden_layer.shape.as_list()[1]
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
writeFile = open(savename + "test_log.txt", "w")

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	if load_model:
		saver.restore(sess, "./checkpoint/2018-08-16-10_09_47/-16200")
	writer = tf.summary.FileWriter(savename, sess.graph)
	for i in range(training_step):
		u, q, y, pos = batch.next()
		dialog_pos = pos
		_, los, acc, summary = sess.run([optimizer, loss, accuracy, merged_summary], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:0.5})
		print("loss of step", i, los, "accuracy:", acc)
		writer.add_summary(summary, i)
		if i % epoch_size == 0:
			index = list(range(0, len(batch.train_u)))
			np.random.shuffle(index)
			batch.train_u = np.array(batch.train_u)[index]
			batch.train_q = np.array(batch.train_q)[index]
			batch.train_y = np.array(batch.train_y)[index]
		if i % 300 == 0 and save_model:
			saver.save(sess, savename, global_step = i)
			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_dev.train_u) / batch_size)
			for _ in range(loops):
				u, q, y, pos = batch_dev.next()
				dialog_pos = pos
				acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += acc
				los_sum += los
			acc = float(acc_sum) / loops
			los = float(los_sum) / loops
			writeFile.write("DEV acc: " + str(acc) + " loss: " + str(los) + ' ')
			print("===== accuracy for dev set: ", acc, "loss", los, "========")
			
			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_tst.train_u) / batch_size)
			for _ in range(loops):
				u, q, y, pos = batch_tst.next() 
				dialog_pos = pos
				acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += acc
				los_sum += los
			acc = float(acc_sum) / loops
			los = float(los_sum) / loops
			writeFile.write("TST acc: " + str(acc) + " loss: " + str(los) + '\n')
			print("===== accuracy for test set: ", acc, "loss", los, "========")

