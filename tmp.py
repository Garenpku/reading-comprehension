# -*- coding:utf-8 -*-
import tensorflow as tf 
from tensorflow import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator import batch_generator
from keras.layers import MaxPooling2D
from glove_embedding import glove_embedding
import numpy as np
import pickle
import time
import os
import copy

save_model = True
load_model = False
write_log = False

query_length = 126# query padding length, n
embedding_dim = 100# word embedding dim , d
dialog_length =  25# k
utter_length = 92 # m
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1
batch_size = 32
hidden_length = 256
training_step = 500000
learning_rate = 0.001

if not os.path.exists("checkpoint/"):
	os.mkdir("checkpoint/")

writeFile = open("test_log.txt", "w")

#产生U,Q和labels
#此处U和Q仅仅是句子的集合，还未转化为word embedding形式
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class = data_helper('trn_clean.txt', 'dev_clean.json', 'tst_clean.json')
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

assert batch.k_max == batch_dev.k_max
assert batch.u_maxlen == batch_dev.u_maxlen
assert batch.q_maxlen == batch_dev.q_maxlen
assert batch_tst.k_max == batch_dev.k_max
assert batch_tst.u_maxlen == batch_dev.u_maxlen
assert batch_tst.q_maxlen == batch_dev.q_maxlen

#batch产生的u,q里面对应每个词是词在字典中的id，使用embedding_lookup就可以转化为词向量表示
labels = tf.placeholder(tf.float32, [batch_size, num_classes]) #[bs,nc]
query_id = tf.placeholder(tf.int32, [batch_size, query_length])
utter_id = tf.placeholder(tf.int32, [batch_size, dialog_length, utter_length])
query = tf.nn.embedding_lookup(embedding_matrix_converted, query_id)#Q:[bs,n,d]
utter = tf.nn.embedding_lookup(embedding_matrix_converted, utter_id) #U:[bs,k,m,d]

print("shape of the query:")
print(query.shape.as_list())
print("shape of the utterance:")
print(utter.shape.as_list())

def mylstm(input_length, hidden_length, x, max_len):
    #Input gate
    Wi = tf.Variable(tf.truncated_normal([input_length, hidden_length], -1, 1))
    Ui = tf.Variable(tf.truncated_normal([hidden_length, hidden_length], -1, 1))
    bi = tf.Variable(tf.truncated_normal([1, hidden_length], -1, 1))
    #Forget gate
    Wf = tf.Variable(tf.truncated_normal([input_length, hidden_length], -1, 1))
    Uf = tf.Variable(tf.truncated_normal([hidden_length, hidden_length], -1, 1))
    bf = tf.Variable(tf.truncated_normal([1, hidden_length], -1, 1))
    #Output gate
    Wo = tf.Variable(tf.truncated_normal([input_length, hidden_length], -1, 1))
    Uo = tf.Variable(tf.truncated_normal([hidden_length, hidden_length], -1, 1))
    bo = tf.Variable(tf.truncated_normal([1, hidden_length], -1, 1))
    #New memory cell
    Wc = tf.Variable(tf.truncated_normal([input_length, hidden_length], -1, 1))
    Uc = tf.Variable(tf.truncated_normal([hidden_length, hidden_length], -1, 1))
    bc = tf.Variable(tf.truncated_normal([1, hidden_length], -1, 1))
    #Initial states
    saved_c = tf.Variable(tf.zeros([batch_size, hidden_length]), trainable = False)
    saved_h = tf.Variable(tf.zeros([batch_size, hidden_length]), trainable = False)

    def lstm_cell(x, h, last_cell):
        nonlocal Ui, Wi, bi, Uf, Wf, bf, Uo, Wo, bo, Uc, Wc, bc
        input_gate = tf.sigmoid(tf.matmul(x, Wi) + tf.matmul(h, Ui) + bi)
        forget_gate = tf.sigmoid(tf.matmul(x, Wf) + tf.matmul(h, Uf) + bf)
        output_gate = tf.sigmoid(tf.matmul(x, Wo) + tf.matmul(h, Uo) + bo)
        new_cell = tf.tanh(tf.matmul(x, Wc) + tf.matmul(h, Uc) + bc)
        final_cell = forget_gate * last_cell + input_gate * new_cell
        final_hidden = output_gate * tf.tanh(last_cell)
        return final_hidden, final_cell

    #x is a batch_size * embedding_dim 2-D matrix
    hidden_states = []
    for i in range(max_len):
        saved_h, saved_c = lstm_cell(x[:, i], saved_h, saved_c)
        tmp_h = tf.Variable(saved_h, trainable = False)
        hidden_states.append(tmp_h)

    res = tf.transpose(tf.stack(hidden_states), [1, 0, 2])
    return res

#dropout
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope("get_v_layer"):
	def sim(t1, t2):
		re = tf.matmul(t1, t2, transpose_b = True) #[m,n]
		tmp1 = tf.reshape(tf.reduce_sum(tf.square(t1), axis = 1), [dialog_length*utter_length, 1])
		tmp2 = tf.reshape(tf.reduce_sum(tf.square(t2), axis = 1), [1,query_length])
		e1 = tf.matmul(tmp1, tf.ones([1, query_length]))
		e2 = tf.matmul(tf.ones([dialog_length*utter_length, 1]), tmp2)
		return 1/(1+tf.sqrt(e1+e2+(-2*re) + 0.01))

	#tf.unstack
	tmp_utter = tf.unstack(tf.reshape(utter, [batch_size, dialog_length*utter_length, embedding_dim])) #
	tmp_query = tf.unstack(query)
	s = [sim(tmp_utter[i], tmp_query[i]) for i in range(batch_size)] # batch * (k * m) * n
	sim_w = tf.reshape(tf.stack(s), [batch_size*dialog_length*utter_length, query_length], name = "S") 
	raw = tf.reshape(tf.stack(s), [batch_size, dialog_length, utter_length, query_length])
	raw = tf.reshape(tf.transpose(raw, [0, 1, 3, 2]), [batch_size * dialog_length * query_length, utter_length])

	utter_attention = tf.Variable(tf.truncated_normal([query_length, embedding_dim], -1, 1)) #A
	query_attention = tf.Variable(tf.truncated_normal([utter_length, embedding_dim], -1, 1))

	utter2 = tf.reshape(tf.matmul(sim_w, utter_attention), [batch_size, dialog_length, utter_length, embedding_dim],name = "U2") #u2
	q2 = tf.reshape(tf.matmul(raw, query_attention), [batch_size, dialog_length, query_length, embedding_dim]) 
	q2 = tf.reduce_mean(q2, axis = 1)
	print("q2:", q2.shape.as_list())
	#v = tf.concat([tf.expand_dims(utter, -1), tf.expand_dims(utter2, -1)], -1, name = "V")
	v = tf.reshape(utter, [batch_size, dialog_length, utter_length, embedding_dim, 1], name = "v")
	new_query = tf.concat([tf.expand_dims(query, -1), tf.expand_dims(q2, -1)], -1)
	print("new_query: ", new_query.shape.as_list())

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

input_length = dialog.shape.as_list()[2]

forward_result = mylstm(input_length, hidden_length, dialog, dialog_length)
backward_result = mylstm(input_length, hidden_length, tf.reverse(dialog, axis = [1]), dialog_length)
print("shape after lstm:")
print(forward_result.shape.as_list())
print(backward_result.shape.as_list())
bilstm_result = tf.concat([forward_result, backward_result], -1)
sim_for_dialog = tf.reduce_sum(tf.reduce_sum(tf.reshape(sim_w, [batch_size, dialog_length, utter_length, query_length]), axis = -1), axis = -1)
sim_for_dialog = tf.expand_dims(tf.nn.softmax(sim_for_dialog, axis = -1), axis = -1)
sim_for_dialog = tf.unstack(sim_for_dialog)
bilstm_result = tf.unstack(bilstm_result)
result_after_attention = tf.stack([tf.matmul(bilstm_result[i], sim_for_dialog[i], transpose_a = True) for i in range(batch_size)])
result_after_attention = tf.squeeze(result_after_attention)
result_after_attention = tf.nn.dropout(result_after_attention, keep_prob)
print("shape after attention:")
print(result_after_attention.shape.as_list())

with tf.name_scope("dialog_attention_layer"):
	#query2 = Conv1D(length_1D, window_length, activation = "relu", name = "Q2")(query) #Q2
	print("new_query: ", new_query.shape.as_list())
	conv = tf.layers.conv2d(inputs = new_query,
		filters = length_1D,
		kernel_size =  [1, embedding_dim], 
		padding = "VALID", 
		activation = tf.nn.relu)
	query2 = tf.squeeze(conv)
	print("query after conv: ", conv.shape.as_list())	
	
	dialog2 = Conv1D(length_1D, window_length, activation = "relu", name = "D2")(dialog) #D2
	dialog_attention = tf.matmul(query2, dialog2, transpose_b = True, name = "P") #P
	da_column =tf.reshape(tf.reduce_sum(dialog_attention, axis = -1), [batch_size, query_length,1],name="pc") #pc
	da_row = tf.reshape(tf.reduce_sum(dialog_attention, axis = -2), [batch_size, 1, dialog_length],name="pr") #pr

with tf.name_scope("concat_layer"):
	#hd = Bidirectional(LSTM(hidden_length))(dialog)
	hq = Bidirectional(LSTM(hidden_length))(query)
	#hd = tf.nn.dropout(hd, keep_prob)
	hq = tf.nn.dropout(hq, keep_prob)
	aq = tf.squeeze(tf.matmul(da_column, query2, transpose_a = True), name = "aq")
	ad = tf.squeeze(tf.matmul(da_row, dialog2), name= "ad")
	hidden_layer = tf.concat([result_after_attention, hq, ad, aq], -1, name="hidden_layer") 
	hidden_length= int(hidden_layer.shape[1])

print(hidden_length)

"""
这一部分是之前调试的时候感觉可能出问题的部分，所以手写一个softmax层并且加了对prediction的标准化
"""
W = tf.Variable(tf.truncated_normal([hidden_length, num_classes], -1, 1))
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

writeFile = open(savename + "test_log0.txt", "w")
epoch_size = int(len(batch.train_u) / batch_size)

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	if load_model:
		saver.restore(sess, "checkpoint/2018-08-06-10_36_54/-51000")
	writer = tf.summary.FileWriter(savename, sess.graph)
	for i in range(training_step):
		u, q, y = batch.next()
		_, los, acc, summary, r, pred, dia, emb = sess.run([optimizer, loss, accuracy, merged_summary, res, prediction, dialog, embedding_matrix], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:0.5})
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
				u, q, y = batch_dev.next()
				acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += acc
				los_sum += los
			acc = float(acc_sum) / loops
			los = float(los_sum) / loops
			writeFile.write("DEV acc: " + str(acc) + " loss: " + str(los))
			print("===== accuracy for dev set: ", acc, "loss", los, "========")

			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_tst.train_u) / batch_size)
			for _ in range(loops):
				u, q, y = batch_tst.next()
				acc, los = sess.run([accuracy, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += acc
				los_sum += los
			acc = float(acc_sum) / loops
			los = float(los_sum) / loops
			writeFile.write("TST acc: " + str(acc) + " loss: " + str(los))
			print("===== accuracy for test set: ", acc, "loss", los, "========")

		if i % 10000 == 0:
			writeFile.close()
			writeFile = open(savename + "test_log" + str(i) + ".txt", "w")
