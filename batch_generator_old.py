# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class batch_generator():
	def __init__(self, U, Q, labels, word2id, vocab_size, all_label, batch_size, query_length, utter_length, dialog_length, word_class):
		tmp_U = [[[word for word in line.split(' ')] for line in query] for query in U]
		tmp_Q = [[word for word in line.split(' ')] for line in Q]
		for i, query in enumerate(tmp_U):
			for j, line in enumerate(query):
				if len(line) > utter_length:
					tmp_U[i][j] = [word for word in line if word_class[word] > 0]
					if len(tmp_U[i][j]) > utter_length:
						tmp_U[i][j] = [word for word in tmp_U[i][j] if word_class[word] > 1]
		for i, line in enumerate(tmp_Q):
			if len(line) > query_length:
				tmp_Q[i] = [word for word in line if word_class[word] > 0]
				if len(tmp_Q[i] > query_length):
					tmp_Q[i] = [word for word in tmp_Q[i] if word_class[word] > 1]
		self.train_u = [[[word2id[word] for word in line] for line in query] for query in tmp_U]
		self.train_q = [[word2id[word] for word in line] for line in tmp_Q]
		self.q_maxlen = query_length
		self.u_maxlen = utter_length
		self.k_max = dialog_length
		self.batch_size = batch_size
		self.length = len(self.train_u)
		self.cur_pos = 0
		ohe = OneHotEncoder().fit(np.reshape(all_label, [-1, 1]))
		self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()	# sample_num * label_size
		for i in range(len(self.train_q)):
			if len(self.train_q[i]) > query_length:
				train_q[i] = train_q[i][:query_length]
			for _ in range(len(self.train_q[i]), self.q_maxlen):
				self.train_q[i].append(vocab_size - 1)
		for i in range(len(self.train_u)):
			if len(self.train_u[i]) > self.k_max:
				self.train_u[i] = self.train_u[i][:self.k_max]
			for j in range(len(self.train_u[i])):
				if len(self.train_u[i][j]) > utter_length:
					self.train_u[i][j] = self.train_u[i][j][:utter_length]
				for _ in range(len(self.train_u[i][j]), self.u_maxlen):
					self.train_u[i][j].append(vocab_size - 1)
			for j in range(len(self.train_u[i]), self.k_max):
				self.train_u[i].append([vocab_size - 1] * self.u_maxlen)
		print(np.array(self.train_u).shape)
		print(np.array(self.train_q).shape)

	def next(self):
		if self.cur_pos + self.batch_size < self.length:
			u = self.train_u[self.cur_pos:self.cur_pos + self.batch_size]
			q = self.train_q[self.cur_pos:self.cur_pos + self.batch_size]			
			y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
			self.cur_pos += self.batch_size
			return np.array(u), np.array(q), np.array(y)
		else:
			record = self.cur_pos
			self.cur_pos = self.batch_size - self.length + self.cur_pos
			if self.cur_pos == 0:
				return np.array(self.train_u[record:self.length]), np.array(self.train_q[record:self.length]), np.array(self.train_y[record:self.length])
			return np.concatenate((self.train_u[record:self.length], self.train_u[:self.cur_pos])), np.concatenate((self.train_q[record:self.length], self.train_q[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos]))
