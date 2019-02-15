from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
import re

"""
Word2Vec Interface Usage:

Parameters:
raw_text : A list of sentences, format like: [['I', 'like', 'apples'], ['Ice-creams', 'are', 'tasty']]
batch_size
embedding_size : dimension of word representation
window_size : number of neighbors to predict in skip-gram model
loop : number of steps for training
learning_rate
num_sampled : Similar to number of samples to fetch in negetive sampling

Returns:
1.Embedding Matrix, each row corresponds to a word
2.id2word, mapping from row index in embedding matrix to words
3.word2id, mapping from words to row indexes in the embedding matrix
"""
def word2vec(U, Q, 
	batch_size = 128, 
	embedding_size = 128, 
	window_size = 3, 
	loop = 5000, 
	learning_rate = 0.1,
	num_sampled = 64):

	data, word2id, id2word, vocab_size = build_dataset(U, Q, window_size)
	batch = batch_generator(data, batch_size)
	print("dataset buliding finished.")
	train_x = tf.placeholder(tf.int32, [batch_size])
	train_y = tf.placeholder(tf.int32, [batch_size, 1])

	embedding_matrix = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1, 1))
	embed = tf.nn.embedding_lookup(embedding_matrix, train_x)

	#NCE parameters（Just treat this as negetive sampling）
	nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], -1, 1))
	nce_bias = tf.Variable(tf.truncated_normal([vocab_size], -1, 1))

	loss = tf.reduce_mean(
		tf.nn.nce_loss(
			weights = nce_weights,
			biases = nce_bias,
			labels = train_y,
			inputs = embed,
			num_sampled = num_sampled,
			num_classes = vocab_size))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:
		sess.run(init)
		for i in range(loop):
			x, y = batch.next()
			feed_dict = {train_x: x, train_y: y}
			_, los = sess.run([optimizer, loss], feed_dict = feed_dict)
			if i % 100 == 0:
				print("loss of step ", i, ":", los)
		word_represent = sess.run([embedding_matrix])
		return word_represent[0], word2id, id2word

class batch_generator():
    def __init__(self, data, batch_size):
        self.data = data
        self.train_x = np.array(data)[:, 0]
        self.train_y = np.reshape(np.array(data)[:, 1], [-1, 1])
        self.length = len(self.train_x)
        self.cur_pos = 0
        self.batch_size = batch_size
    def next(self):
        if self.cur_pos + self.batch_size < self.length:
            x = self.train_x[self.cur_pos:self.cur_pos + self.batch_size]
            y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
            self.cur_pos += self.batch_size
            return x, y
        else:
            record = self.cur_pos
            self.cur_pos = self.batch_size - self.length + self.cur_pos
            if self.cur_pos == 0:
                return self.train_x[record:self.length], self.train_y[record:self.length]
            try:
                return np.concatenate((self.train_x[record:self.length], self.train_x[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos]))
            except:
                print(np.array(self.train_x[record:self.length]).shape)
                print(np.array(self.train_x[:self.cur_pos]).shape)
                raise ValueError

def build_dataset(U, Q, window_size):
	words = []
	sentences = [line.split(' ') for line in Q]
	for utterances in U:
		for line in utterances:
			sentences.append(line.split(' '))
	for sent in sentences:
		for word in sent:
			words.append(word)
	words = set(words)
	word2id = {}
	id2word = {}
	vocab_size = len(words)
	print("vocab size:", vocab_size)
	for i, word in enumerate(words):
		word2id[word] = i
		id2word[i] = word

	data = []
	for sentence in sentences:
		for index, word in enumerate(sentence):
			for nb_word in range(max(0, index - window_size), min(index + window_size + 1, len(sentence))):
				if nb_word != index:
					data.append([word2id[word], word2id[sentence[nb_word]]])
	return data, word2id, id2word, vocab_size



