# -*- coding:utf-8 -*-
import tensorflow as tf 
import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator import batch_generator
from keras.layers import MaxPooling2D
from glove_embedding import glove_embedding
import numpy as np
import pickle
import json
import time
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

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
utter_length = 92 # m
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1 
batch_size = 32
hidden_length = 32
training_step = 500000
learning_rate = 0.001
speaker_embedding_dim = 50

config.query_length = query_length
config.embedding_dim = embedding_dim
config.dialog_length = dialog_length
config.utter_length = utter_length
config.filter_sizes = filter_sizes
config.filter_num = filter_num
config.length_1D = length_1D 
config.window_length = window_length
config.batch_size = batch_size
config.hidden_length = hidden_length
config.training_step = training_step
config.learning_rate = learning_rate
config.speaker_embedding_dim = speaker_embedding_dim


if not os.path.exists("checkpoint/"):
	os.mkdir("checkpoint/")



#产生U,Q和labels
#此处U和Q仅仅是句子的集合，还未转化为word embedding形式
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class, raw_query, tmp = data_helper('trn.json', 'dev.json', 'tst.json')
#label的个数
num_classes = len(label2id)
id2label = dict(zip(label2id.values(), label2id.keys()))
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
batch_dev = batch_generator(U_dev, Q_dev, labels_dev, word2id, list(label2id.values()), word_class, config)
batch_tst = batch_generator(U_tst, Q_tst, labels_tst, word2id, list(label2id.values()), word_class, config)


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

#dropout
keep_prob = tf.placeholder(tf.float32)

###########
#填充代码
###########
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
#v = tf.expand_dims(utter, -1)

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
"""
with tf.name_scope("dialog_attention_layer"):
	query2 = Conv1D(length_1D, window_length, activation = "relu", name = "Q2")(query) #Q2
	dialog2 = Conv1D(length_1D, window_length, activation = "relu", name = "D2")(dialog) #D2
	dialog_attention = tf.matmul(query2, dialog2, transpose_b = True, name = "P") #P
	da_column =tf.reshape(tf.reduce_sum(dialog_attention, axis = -1), [batch_size, query_length - window_length + 1, 1],name="pc") #pc
	da_row = tf.reshape(tf.reduce_sum(dialog_attention, axis = -2), [batch_size, 1, dialog_length - window_length + 1],name="pr") #pr
"""

with tf.name_scope("concat_layer"):
	hd = Bidirectional(LSTM(hidden_length))(dialog)
	hq = Bidirectional(LSTM(hidden_length))(query)
	hd = tf.nn.dropout(hd, keep_prob)
	hq = tf.nn.dropout(hq, keep_prob)
	#aq = tf.squeeze(tf.matmul(da_column, query2, transpose_a = True), name = "aq")
	#ad = tf.squeeze(tf.matmul(da_row, dialog2), name= "ad")
	print("hi", dialog.shape, query.shape)
	hidden_layer = tf.concat([hd, hq], -1, name="hidden_layer") 
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

#####################、

pred_answers = tf.argmax(prediction, axis = 1)
ground_truth = tf.argmax(labels, axis = 1)
correct_prediction = tf.equal(tf.argmax(prediction,axis = 1),tf.argmax(labels,axis = 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
right_num = tf.cast(correct_prediction, "float") 
batch_last = batch_size

if write_log:
	if not os.path.exists("log/"):
		os.mkdir("log/")
	ummary_waiter = tf.summary.FileWriter("log/",tf.get_default_graph())
	ummary_waiter.close()


#saver = tf.train.Saver(max_to_keep = 4)

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

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
savename="checkpoint/"+now+'/'
epoch_size = int(len(batch.train_u) / batch_size)
#writeFile = open(savename + "test_log.txt", "w")

acc_record = 0

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	if load_model:
		saver.restore(sess, "./checkpoint/2018-08-16-10_09_47/-16200")
	writer = tf.summary.FileWriter(savename, sess.graph)
	for i in range(training_step):
		u, q, y = batch.next()
		_, los, acc, summary= sess.run([optimizer, loss, accuracy, merged_summary], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:0.5})
		#print("loss of step", i, los, "accuracy:", acc)
		writer.add_summary(summary, i)
		if i % epoch_size == 0:
			index = list(range(0, len(batch.train_u)))
			np.random.shuffle(index)
			batch.train_u = np.array(batch.train_u)[index]
			batch.train_q = np.array(batch.train_q)[index]
			batch.train_y = np.array(batch.train_y)[index]
		if i % 100 == 0:
			#saver.save(sess, savename, global_step = i)
			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_dev.train_u) / batch_size) + 1
			wrong_answer = []
			for k in range(loops):
				u, q, y = batch_dev.next()
				if k == loops - 1:
					batch_last = len(batch_dev.train_u) % batch_size
				right, los, pred, l = sess.run([right_num, loss, pred_answers, ground_truth], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				for j, ans in enumerate(right):
					if ans == 0 and batch_last == batch_size:
						wrong_answer.append(raw_query[batch_dev.cur_pos - batch_size + j])
						wrong_answer[-1]['Predicted'] = id2label[pred[j]]
						wrong_answer[-1]['GroundTruth'] = id2label[l[j]]

				acc_sum += np.sum(right[:batch_last])
				los_sum += los
				batch_last = batch_size

			acc = float(acc_sum) / len(batch_dev.train_u)
			los = float(los_sum) / loops
			batch_dev.cur_pos = 0
			print("step:", i)
			print("===== accuracy for dev set: ", acc, "loss", los, "========")
			if acc > acc_record:
				ans_writer = open("wrong_answers_cnnlstm.txt", "w")
				acc_record = acc
				json.dump(wrong_answer, ans_writer)
				ans_writer.close()

			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_tst.train_u) / batch_size)
			for k in range(loops):
				u, q, y = batch_tst.next()
				if k == loops - 1:
					batch_last = len(batch_tst.train_u) % batch_size
				right, los = sess.run([right_num, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += np.sum(right[:batch_last])
				los_sum += los
				batch_last = batch_size
			acc = float(acc_sum) / len(batch_tst.train_u)
			los = float(los_sum) / loops
			batch_tst.cur_pos = 0
			print("===== accuracy for test set: ", acc, "loss", los, "========")


