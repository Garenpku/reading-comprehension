# -*- coding:utf-8 -*-
import tensorflow as tf 
import json
import math
import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper_coreference import data_helper
from word2vec import word2vec
from batch_generator_coreference import batch_generator
from keras.layers import MaxPooling2D
from attentive_reader import attentive_reader
from glove_embedding import glove_embedding
from transformer import transformer
import numpy as np
import pickle
import time
import os

config = tf.ConfigProto(allow_soft_placement = True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth=True
#config.allow_soft_placement = config.getboolean(‘gpu’, ‘allow_soft_placement’)

save_model = False
load_model = False
write_log = False

query_length = 126# query padding length, n
embedding_dim = 100# word embedding dim , d
dialog_length =  25# k
utter_length = 300 # m
filter_sizes = [2,3,4,5]
filter_num = 50 #f
length_1D =  50 #e
window_length = 1 
batch_size = 16
hidden_length = 32
training_step = 500000
learning_rate = 0.001
attention_length = 64
layer_num = 4
entity_max_num = 80
pronoun_max_num = 50

if not os.path.exists("checkpoint/"):
    os.mkdir("checkpoint/")

batch_last = batch_size


#产生U,Q和labels
#此处U和Q仅仅是句子的集合，还未转化为word embedding形式
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class, raw_query, tmp = data_helper('trn.json', 'dev.json', 'tst.json')
#label的个数
num_classes = len(label2id)
print(num_classes)
print(label2id)
id2label = dict(zip(label2id.values(), label2id.keys()))

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
batch = batch_generator(U, Q, labels, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class, pronoun_max_num, entity_max_num)
batch_dev = batch_generator(U_dev, Q_dev, labels_dev, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class, pronoun_max_num, entity_max_num)
batch_tst = batch_generator(U_tst, Q_tst, labels_tst, word2id, vocab_size, list(label2id.values()), batch_size, query_length, utter_length, dialog_length, word_class, pronoun_max_num, entity_max_num)

print(labels_dev[:10])
print(batch_dev.train_y[:10])

#batch产生的u,q里面对应每个词是词在字典中的id，使用embedding_lookup就可以转化为词向量表示
labels = tf.placeholder(tf.float32, [batch_size, num_classes]) #[bs,nc]
query_id = tf.placeholder(tf.int32, [batch_size, query_length])
utter_id = tf.placeholder(tf.int32, [batch_size, utter_length])
query = tf.nn.embedding_lookup(embedding_matrix_converted, query_id)#Q:[bs,n,d]
utter = tf.nn.embedding_lookup(embedding_matrix_converted, utter_id) #U:[bs,k,d]

all_entity = ["@ent0" + str(i) for i in range(0, 10)] + ["@ent" + str(i) for i in range(10, num_classes)]
all_entity = [word2id[word] for word in all_entity]
ent = tf.nn.embedding_lookup(embedding_matrix_converted, all_entity) # [nc, d]

print("shape of the query:")
print(query.shape.as_list())
print("shape of the utterance:")
print(utter.shape.as_list())

#dropout
keep_prob = tf.placeholder(tf.float32)

#####################
dialog_index = np.zeros([batch_size, dialog_length], np.int32)

#此处打算尝试指代消解
# 1.将utter过一个双向lstm
# 2.取所有entity组成一个矩阵；取所有代词组成一个矩阵
# 3.将两个矩阵相乘，在entity维度做softmax
# 4.对于向量中的每一个数值，将对应的@ent序号位置加上这个值
entity_pos = np.zeros([batch_size, entity_max_num], np.int32)
pronoun_pos = np.zeros([batch_size, pronoun_max_num], np.int32)

weight_matrix = np.zeros([batch_size, pronoun_max_num, entity_max_num])
for i in range(batch_size):
    for j in range(pronoun_max_num):
        for k in range(entity_max_num):
            wight_matrix[i][j][k] = 10 / math.abs(pronoun_pos[i][j] - entity_pos[i][k])

entity_class = np.zeros([batch_size, num_classes, entity_max_num], np.float32)
with tf.device('/cpu:0'):
	dialog_before = Bidirectional(LSTM(64, return_sequences = True))(utter)

entity_matrix = tf.stack([tf.stack([dialog_before[i, entity_pos[i, j], :] for j in range(entity_max_num)]) for i in range(batch_size)]) #batch_size * entity_max_num * 64
pronoun_matrix = tf.stack([tf.stack([dialog_before[i, pronoun_pos[i, j], :] for j in range(pronoun_max_num)]) for i in range(batch_size)]) # batch_size * pronoun_max_num * 64

assert entity_matrix.shape.as_list() == [batch_size, entity_max_num, 128]
assert pronoun_matrix.shape.as_list() == [batch_size, pronoun_max_num, 128]

pronoun_mask = np.ones([batch_size, utter_length, 1], np.float32)
for i in range(batch_size):
    for position in pronoun_pos[i]:
        pronoun_mask[i][position][0] = 0.

mutual_matrix = tf.matmul(entity_matrix, pronoun_matrix, transpose_b = True) # bs * entity * pronoun
mutual_matrix = tf.transpose(mutual_matrix, [0, 2, 1]) #bs * pronoun *entity 
mutual_matrix = tf.nn.softmax(mutual_matrix, axis = -1)
mutual_matrix = tf.multiply(mutual_matrix, weight_matrix)

score_list = []
for i in range(num_classes):
    entity_mask = entity_class[:, i, :]
    entity_mask = tf.expand_dims(entity_mask, axis = 1)
    score_one_class = tf.reduce_sum(entity_mask * mutual_matrix, axis = -1)
    score_list.append(score_one_class)
entity_score = tf.stack(score_list, axis = -1)
print("shape of entity_score:", entity_score.shape.as_list())
"""
entity_score = tf.Variable(tf.zeros([batch_size, pronoun_max_num, num_classes]))
for i in range(batch_size):
    for j in range(entity_max_num):
        for k in range(pronoun_max_num):
            entity_score[i, k, entity_class[i][j]] += mutual_matrix[i][k][j]
"""
utter = utter * pronoun_mask
pronoun_embedding = [[] for i in range(batch_size)]
for i in range(batch_size):
    for j in range(utter_length):
        if j in pronoun_pos[i]:
            score = tf.expand_dims(entity_score[i, j, :], axis = 0) # 1 * num_classes 
            replaced_embedding = tf.squeeze(tf.matmul(score, ent))
            pronoun_embedding[i].append(replaced_embedding)
        else:
            pronoun_embedding[i].append(tf.zeros([100], tf.float32))
pronoun_embedding = tf.stack([tf.stack(p) for p in pronoun_embedding])
print("pronoun embedding shape: ", pronoun_embedding.shape.as_list())
utter = utter + pronoun_embedding

"""
for i in range(batch_size):
    for j in range(pronoun_max_num):
        if pronoun_pos[i][j] != -1:
            score = tf.expand_dims(entity_score[i, j, :], axis = 0) # 1 * num_classes 
            replaced_embedding = tf.squeeze(tf.matmul(score, ent))
            print(replaced_embedding.shape.as_list())
            utter[pronoun_pos[i][j]] += replaced_embedding
"""            

##尝试 1 stat
with tf.device('/gpu:1'):
	dialog_raw = Bidirectional(LSTM(64, return_sequences = True))(utter)
##尝试 1 end
'''

##尝试2 stat
from attention_gru import Attention_BiGru
dialog_raw = Attention_BiGru(utter, query, 32, return_sequences = True)
##尝试2 end
'''
"""
##尝试 3 start
dialog_raw = utter
for _ in range(layer_num):
    dialog_raw = transformer(dialog_raw, utter_length, batch_size, 8, 64, embedding_dim)
print("dialog_raw:", dialog_raw.shape.as_list())
"""
shape = dialog_index.shape
print(shape)
dialog = [dialog_raw[i,dialog_index[i,j],:] for i in range(shape[0]) for j in range(shape[1])]
dialog = tf.reshape(tf.stack(dialog), [batch_size, dialog_length, -1])


with tf.name_scope("dialog_align_layer"):
	query2 = Conv1D(length_1D, window_length, activation = "relu", name = "Q2")(query) #Q2
	dialog2 = Conv1D(length_1D, window_length, activation = "relu", name = "D2")(dialog) #D2
	dialog_attention = tf.matmul(query2, dialog2, transpose_b = True, name = "P") #P
	query2 = tf.matmul(dialog_attention, query2, transpose_a=True)
	'''
	da_column =tf.reshape(tf.reduce_sum(dialog_attention, axis = -1), [batch_size, query_length - window_length + 1, 1],name="pc") #pc
	da_row = tf.reshape(tf.reduce_sum(dialog_attention, axis = -2), [batch_size, 1, dialog_length - window_length + 1],name="pr") #pr
	'''

with tf.name_scope("multimodal_attention_layer"):
	def get_weight(x):
		shape_x = x.shape.as_list()
		#[bs, dialog_length, length_1D]
		y = TimeDistributed(Dense(64, activation = "tanh"))(x)
		v = tf.Variable(tf.truncated_normal([1,1,64]), dtype = tf.float32)
		e = tf.reduce_sum(y*v, axis = -1)
		a = tf.nn.softmax(e)
		return a
	wei_dialog = get_weight(dialog2)
	wei_query = get_weight(query2)
	h = TimeDistributed(Dense(length_1D))(tf.concat([dialog2, query2], axis = -1))
	wei_s = (wei_dialog+wei_query)/2
	wei_h = get_weight(h)
	wei_u = tf.expand_dims((wei_s+wei_h)/2, axis = -1)
	#简化版，省略cnn
	hma = tf.reduce_sum(h*wei_u, axis = 1)



with tf.name_scope("self_attention_layer"):
	from attention_gru import *
	att_gru =  Attention_Gru(dialog, dialog)
	dialog = att_gru.get_result(return_sequences = True)
	att_gru_query = Attention_Gru(query, query)
	query = att_gru_query.get_result(return_sequences = True)


with tf.name_scope("concat_layer"):
	hd = Bidirectional(LSTM(hidden_length))(dialog)
	hq = Bidirectional(LSTM(hidden_length))(query)
	hd = tf.nn.dropout(hd, keep_prob)
	hq = tf.nn.dropout(hq, keep_prob)
	'''
	aq = tf.squeeze(tf.matmul(da_column, query2, transpose_a = True), name = "aq")
	ad = tf.squeeze(tf.matmul(da_row, dialog2), name= "ad")
	'''
	print("hi", dialog.shape, query.shape)
	att_gru_qd = Attention_Gru(dialog, query)

	hidden_layer = tf.concat([hd, hq, hma, att_gru_qd.get_result()], -1, name="hidden_layer") 
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

with tf.Session(config = config) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    if load_model:
        saver.restore(sess, "./checkpoint/2018-08-16-10_09_47/-16200")
    writer = tf.summary.FileWriter(savename, sess.graph)
    for i in range(training_step):
        u, q, y, di, pronoun_pos, entity_pos, entity_class = batch.next()
        dialog_index = di
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
                u, q, y, di, pronoun_pos, entity_pos, entity_class = batch_dev.next()
                if k == loops - 1:
                    batch_last = len(batch_dev.train_u) % batch_size
                dialog_index = di
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
                ans_writer = open("wrong_answers.txt", "w")
                acc_record = acc
                json.dump(wrong_answer, ans_writer)
                ans_writer.close()

            acc_sum = 0
            los_sum = 0
            loops = int(len(batch_tst.train_u) / batch_size)
            for k in range(loops):
                u, q, y, di, pronoun_pos, entity_pos, entity_class = batch_tst.next()
                if k == loops - 1:
                    batch_last = len(batch_tst.train_u) % batch_size
                dialog_index = di 
                right, los = sess.run([right_num, loss], feed_dict = {query_id:q, utter_id:u, labels:y, keep_prob:1})
                acc_sum += np.sum(right[:batch_last])
                los_sum += los
                batch_last = batch_size
            acc = float(acc_sum) / len(batch_tst.train_u)
            los = float(los_sum) / loops
            batch_tst.cur_pos = 0
            print("===== accuracy for test set: ", acc, "loss", los, "========")

