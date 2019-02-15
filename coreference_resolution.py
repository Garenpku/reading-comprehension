import tensorflow as tf 
from keras.layers import Conv2D, TimeDistributed, Dense, Conv1D, MaxPooling1D
import numpy as np 

'''
直接调用single_softmax()得到得到分类可能性的softmax结果， [bs, mentions_num, num_classes]
通过比较计算相似度聚类还是有问题，我再想一下
其他离散特征都需要另外的神经网络进行训练而且用到很多人工特征，所以这里的离散特征单独指speaker的embedding
'''

k = 4 #embedding feature map num
m = 10 #embedding length
n = 20 #embedding dim
filter_num = 50 #conv1 feature num
df_dim = 20 #discrete feature dim
filter_sizes = [1,2,3]
mpe_dim = 30 #mention pair embedding dim， 暂时用不上
pf_dim = 15
bs = 32
mentions_num = 50 #scenes 里mention的最大值

def get_mention_embedding(_feature_map, dis_fea):
	def conv1_cell(_x, scope):
		with tf.variable_scope(scope, reuse = tf.AUTO_REUSE) as scope:
			x = tf.reshape(_x, [-1, m, n])
			outputs = []
			for filter_size in filter_sizes:
				conv1d = Conv1D(filter_num, filter_size)(x)
				outputs.append(MaxPooling1D(pool_size = conv1d.shape.as_list()[1])(conv1d))
			return tf.concat(outputs, axis = 1)
	with tf.variable_scope("mention_embed", reuse = tf.AUTO_REUSE) as scope:
		with tf.name_scope("conv1"):
			feature_map = tf.reshape(_feature_map, [-1, k, m, n])
			dis_fea = tf.reshape(dis_fea,[-1, df_dim])
			conv1_cells = [conv1_cell(feature_map[:,i,:,:], "conv1_cell"+str(i)) for i in range(k)]
			conv1 = tf.stack(conv1_cells, axis = -1)
		with tf.name_scope("conv2"):
			shape = conv1.shape.as_list()
			conv2 = tf.reshape((Conv2D(1, (shape[1], 1))(conv1)), [-1, filter_num])
		return tf.concat([conv2, dis_fea], axis = -1)

def cal_sim(_m1, _m2, _dis_fea1, _dis_fea2, _pair_fea = None):
	with tf.variable_scope("pair_men_embed") as scope:
		men_embed1 = get_mention_embedding(m1, dis_fea1)
		men_embed2 = get_mention_embedding(m2, dis_fea2)
	with tf.name_scope("conv3"):
		conv3_input = tf.expand_dims(tf.stack([men_embed1, men_embed2], axis = -2), axis = -1)
		conv3 = Conv2D(1, (2,1), activation= "tanh")(conv3_input)
		conv3 = tf.reshape(conv3, [bs, -1])
	#re = tf.concat([conv3, _pair_fea], axis = -1)
	re = Dense(64, activation = "tanh")(conv3)
	re = Dense(1, activation = "sigmoid")
	return re

def single_softmax():
	#需要每个scene的mention padding到指定数目
	#只需要每一个mention对应类别的可能性的话，需要传入feature_maps 和 discrete features
	feature_maps = tf.placeholder(tf.float32, [bs, mentions_num, k, m, n])
	_dis_feas = tf.placeholder(tf.int32, [bs, mentions]) #speaker_feature
	speaker_embed_matrix = tf.truncated_normal([num_classes, df_dim])
	dis_feas = tf.nn.embedding_lookup(speaker_embed_matrix, _dis_feas)

	embed = get_mention_embedding(feature_maps, dis_feas)
	embed = Dense(num_classes, activation = "tanh")(embed)
	re = Dense(num_classes, activation = "softmax")(embed)
	re = tf.reshape(re, [bs, mentions_num, num_classes])
	return re







	