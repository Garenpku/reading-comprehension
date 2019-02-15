# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def test(query, query_length, utter, utter_length):
	for line in query:
		if len(line) > query_length:
			return False
	for q in utter:
		for line in q:
			if len(line) > utter_length:
				return False
	return True


class batch_generator():
	def __init__(self, U, Q, labels, word2id, all_label, word_class, config):
		tmp_U = [[[word for word in line.split(' ')] for line in query] for query in U]
		tmp_Q = [[word for word in line.split(' ')] for line in Q]
		for i, query in enumerate(tmp_U):
			for j, line in enumerate(query):
				cnt = 0
				while len(tmp_U[i][j]) > config.utter_length:
					tmp_U[i][j] = [word for word in line if word_class[word] > cnt]
					cnt += 1
		for i, line in enumerate(tmp_Q):
			cnt = 0
			while len(tmp_Q[i]) > config.query_length:
				tmp_Q[i] = [word for word in line if word_class[word] > cnt]
				cnt += 1
		self.train_u = [[[word2id[word] for word in line] for line in query] for query in tmp_U]
		self.train_q = [[word2id[word] for word in line] for line in tmp_Q]
		self.batch_size = config.batch_size
		self.length = len(self.train_u)
		self.cur_pos = 0
		
		ohe = OneHotEncoder().fit(np.reshape(all_label, [-1, 1]))
		self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()	# sample_num * label_size
		
		if test(self.train_q, config.query_length, self.train_u, config.utter_length):
			print("Not truncated.")
		else:
			print("Truncated.")
		
		padding_vocab_id = config.vocab_size - 1
		padding_speaker_id = len(all_label)-1

		for i in range(len(self.train_q)):
			if len(self.train_q[i]) > config.query_length:
				train_q[i] = train_q[i][:query_length]

			for _ in range(len(self.train_q[i]), config.query_length):
				self.train_q[i].append(padding_vocab_id)
			
			#self.train[i] = self.train[i] + [padding_vocab_id]*(self.q_maxlen-len(self.train_q[i]))

		for i in range(len(self.train_u)):
			if len(self.train_u[i]) > config.dialog_length:
				self.train_u[i] = self.train_u[i][:config.dialog_length]
			for j in range(len(self.train_u[i])):
				if len(self.train_u[i][j]) > config.utter_length:
					self.train_u[i][j] = self.train_u[i][j][:config.utter_length]
				for _ in range(len(self.train_u[i][j]), config.utter_length):
					self.train_u[i][j].append(padding_vocab_id)
			for j in range(len(self.train_u[i]), config.dialog_length):
				self.train_u[i].append([padding_vocab_id] * config.utter_length)


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
