# -*- coding:utf-8 -*-
import tensorflow as tf 
import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator_long import batch_generator
from keras.layers import MaxPooling2D
from glove_embedding import glove_embedding
import numpy as np
import pickle
import time
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

print_result_flag = False
save_model = True
load_model = False
write_log = False

query_length = 126# query padding length, n
embedding_dim = 100# word embedding dim , d
dialog_length =  25# k
utter_length = 500 # m
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1 
batch_size = 32
hidden_length = 32
training_step = 500000
learning_rate = 0.001

if not os.path.exists("checkpoint/"):
	os.mkdir("checkpoint/")


#产生U,Q和labels
#此处U和Q仅仅是句子的集合，还未转化为word embedding形式
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class = data_helper('trn.json', 'dev.json', 'tst.json')
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

#batch_dev = batch_generator(U[split_point:], Q[split_point:], labels[split_point:], word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class)

#batch产生的u,q里面对应每个词是词在字典中的id，使用embedding_lookup就可以转化为词向量表示
labels = tf.placeholder(tf.float32, [batch_size, num_classes]) #[bs,nc]
query_id = tf.placeholder(tf.int32, [batch_size, query_length])
utter_id = tf.placeholder(tf.int32, [batch_size, utter_length])
query = tf.nn.embedding_lookup(embedding_matrix_converted, query_id)#Q:[bs,n,d]
utter = tf.nn.embedding_lookup(embedding_matrix_converted, utter_id) #U:[bs,k,d]

print("shape of the query:")
print(query.shape.as_list())
print("shape of the utterance:")
print(utter.shape.as_list())

#dropout
keep_prob = tf.placeholder(tf.float32)

#####################

with tf.name_scope("concat_layer"):
	hd = Bidirectional(LSTM(hidden_length))(utter)
	hq = Bidirectional(LSTM(hidden_length))(query)
	hd = tf.nn.dropout(hd, keep_prob)
	hq = tf.nn.dropout(hq, keep_prob)

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

'''
if write_log:
	
	ummary_waiter = tf.summary.FileWriter("log/",tf.get_default_graph())
	#ummary_waiter.close()
'''
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
savename="checkpoint/"+now+'/'
epoch_size = int(len(batch.train_u) / batch_size)
result_cnt = 0
last_eval_max = 0.83
import pickle as pk
#writeFile = open(savename + "test_log.txt", "w")
batch_last = batch_size

if not os.path.exists("log/"):
		os.mkdir("log/")

saver = tf.train.Saver(max_to_keep = 4)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('log/', sess.graph)
	if load_model:
		saver.restore(sess, "./checkpoint/2018-08-16-10_09_47/-16200")
	else:
		init = tf.global_variables_initializer()
		sess.run(init)
	trn_acc = []
	for i in range(training_step):
		u, q, y, di = batch.next()
		_, los, acc, summary= sess.run([optimizer, loss, accuracy, merged_summary], feed_dict = {dialog_index:di,query_id:q, utter_id:u, labels:y, keep_prob:0.5})
		#print(_dialog_test)
		#assert False
		#print("loss of step", i, los, "accuracy:", acc)
		trn_acc.append(acc)
		writer.add_summary(summary, i)
		if i % epoch_size == 0:
			index = list(range(0, len(batch.train_u)))
			np.random.shuffle(index)
			batch.train_u = np.array(batch.train_u)[index]
			batch.train_q = np.array(batch.train_q)[index]
			batch.train_y = np.array(batch.train_y)[index]
		if i % 100 == 0:
			if i%1000 == 0 and i >= 40000:
				saver.save(sess, savename, global_step = i)
			acc_sum = 0
			los_sum = 0
			dev_res_pr = []
			dev_res_tr = []
			tst_res_pr = []
			tst_res_tr = []
			loops = int(len(batch_dev.train_u) / batch_size) + 1
			for k in range(loops):
				u, q, y, di = batch_dev.next()
				if k == loops - 1:
					batch_last = len(batch_dev.train_u) % batch_size
				#dialog_index = di
				right, acc, los, prlb, trlb = sess.run([right_num, accuracy, loss, pred_answers, ground_truth], feed_dict = {dialog_index:di,query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += np.sum(right[:batch_last])
				los_sum += los
				batch_last = batch_size
				dev_res_pr.append(prlb)
				dev_res_tr.append(trlb)
			acc = float(acc_sum) / len(batch_dev.train_u)
			los = float(los_sum) / loops
			batch_dev.cur_pos = 0
			print("step:", i)
			print("acc for train set: ", sum(trn_acc)*1.0/len(trn_acc))
			trn_acc = []
			#writeFile.write("DEV acc: " + str(acc) + " loss: " + str(los) + ' ')
			print("===== accuracy for dev set: ", acc, "loss", los, "========")
			dev_acc = acc

			acc_sum = 0
			los_sum = 0
			loops = int(len(batch_tst.train_u) / batch_size) + 1
			for k in range(loops):
				u, q, y, di = batch_tst.next()
				if k == loops - 1:
					batch_last = len(batch_dev.train_u) % batch_size
				#dialog_index = di 
				right, acc, los, prlb, trlb = sess.run([right_num, accuracy, loss, pred_answers, ground_truth], feed_dict = {dialog_index:di,query_id:q, utter_id:u, labels:y, keep_prob:1})
				acc_sum += np.sum(right[:batch_last])
				los_sum += los
				batch_last = batch_size
				tst_res_pr.append(prlb)
				tst_res_tr.append(trlb)
			acc = float(acc_sum) / len(batch_dev.train_u)
			los = float(los_sum) / loops
			batch_dev.cur_pos = 0
			#writeFile.write("TST acc: " + str(acc) + " loss: " + str(los) + '\n')
			print("===== accuracy for test set: ", acc, "loss", los, "========")

			if print_result_flag == True:
				if dev_acc > last_eval_max:
					eval_res = {"dev_acc":dev_acc, "tst_acc":acc,"dev_prediction": dev_res_pr, "dev_true": dev_res_tr, "tst_prediction": tst_res_pr, "tst_true": tst_res_tr}
					with open("eval_result-"+str(i)+'-'+str(dev_acc), "wb") as f:
						pk.dump(eval_res, f)
					result_cnt += 1
					if result_cnt >= 3:
						last_eval_max += 0.02
