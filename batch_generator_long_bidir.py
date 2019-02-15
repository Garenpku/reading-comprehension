# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def process(utter):
	res = []
	pos1 = []
	pos2 = []
	for query in utter:
		tmp = []
		tmp_pos1 = []
		tmp_pos2 = []
		record = 0
		last = 0
		for line in query:
			tmp += line
			record += len(line)
			tmp_pos1.append(record)
			tmp_pos2.append(record)
			last += len(line)
		res.append(tmp)
		pos1.append(tmp_pos1)
		pos2.append(tmp_pos2)
	return res, pos1, pos2

def get_length(utter):
	return sum([len(query) for query in utter])

class batch_generator():
	def __init__(self, U, Q, labels, word2id, vocab_size, all_label, batch_size, query_length, utter_length, dialog_length, word_class):
		tmp_U = [[[word for word in line.split(' ')] for line in query] for query in U]
		#tmp_U = process(tmp_U)
		tmp_Q = [[word for word in line.split(' ')] for line in Q]
		for i, query in enumerate(tmp_U):
			cnt = 0
			while get_length(tmp_U[i]) > utter_length:
				tmp_U[i] = [[word for word in line if word_class[word] > cnt] for line in query]
				cnt += 1
		for i, line in enumerate(tmp_Q):
			cnt = 0
			while len(tmp_Q[i]) > query_length:
				tmp_Q[i] = [word for word in line if word_class[word] > cnt]
				cnt += 1
		tmp_U, pos1, pos2 = process(tmp_U)
		self.dialog_pos1 = pos1
		self.dialog_pos2 = pos2
		self.train_u = [[word2id[word] for word in line] for line in tmp_U] 
		imax = max([len(line) for line in tmp_U])
		print("max length of utterances:", imax)
		self.train_q = [[word2id[word] for word in line] for line in tmp_Q]
		self.q_maxlen = query_length
		self.u_maxlen = utter_length
		self.batch_size = batch_size
		self.length = len(self.train_u)
		self.cur_pos = 0
		ohe = OneHotEncoder().fit(np.reshape(all_label, [-1, 1]))
		self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()	# sample_num * label_size
		for i in range(len(self.train_q)):
			for _ in range(len(self.train_q[i]), self.q_maxlen):
				self.train_q[i].append(vocab_size - 1)
		for i in range(len(self.train_u)):
			for _ in range(len(self.train_u[i]), self.u_maxlen):
				self.train_u[i].append(vocab_size - 1)
		for i in range(len(self.train_u)):
			for _ in range(len(self.dialog_pos1[i]), dialog_length):
				self.dialog_pos1[i].append(-1)
				self.dialog_pos2[i].append(0)

	def next(self):
		if self.cur_pos + self.batch_size < self.length:
			u = self.train_u[self.cur_pos:self.cur_pos + self.batch_size]
			q = self.train_q[self.cur_pos:self.cur_pos + self.batch_size]			
			y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
			pos1 = self.dialog_pos1[self.cur_pos:self.cur_pos + self.batch_size]
			pos2 = self.dialog_pos2[self.cur_pos:self.cur_pos + self.batch_size]
			self.cur_pos += self.batch_size
			return np.array(u), np.array(q), np.array(y), pos1, pos2
		else:
			record = self.cur_pos
			self.cur_pos = self.batch_size - self.length + self.cur_pos
			if self.cur_pos == 0:
				return np.array(self.train_u[record:self.length]), np.array(self.train_q[record:self.length]), np.array(self.train_y[record:self.length]), np.array(self.dialog_pos1[record:self.length]), np.array(self.dialog_pos2[record:self.length])

			return np.concatenate((self.train_u[record:self.length], self.train_u[:self.cur_pos])), np.concatenate((self.train_q[record:self.length], self.train_q[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos])), np.concatenate((self.dialog_pos1[record:self.length], self.dialog_pos1[:self.cur_pos])), np.concatenate((self.dialog_pos2[record:self.length], self.dialog_pos2[:self.cur_pos]))
