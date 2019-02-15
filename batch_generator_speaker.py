# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class batch_generator():
	def __init__(self, U, Q, Speaker, labels, word2id, all_label, word_class, config):
		tmp_U = [[[word for word in line.split(' ')] for line in query] for query in U]
		tmp_Q = [[word for word in line.split(' ')] for line in Q]

		#按照停用词表和频率去除过长的句子的部分词
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
		self.Speaker = Speaker
		self.batch_size = config.batch_size
		self.length = len(self.train_u)
		self.cur_pos = 0

		#将all_label里面的标签从小到大排列代表每一列，labels里面将数字重合的那一位标位1，其余是0
		ohe = OneHotEncoder().fit(np.reshape(all_label, [-1, 1]))
		self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()	# sample_num * label_size
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

		for i,speakers in enumerate(self.Speaker):
			if len(speakers) > config.dialog_length:
				self.Speaker[i] = self.Speaker[i][:config.dialog_length]
			else:
				self.Speaker[i] = self.Speaker[i] + [padding_speaker_id]*(config.dialog_length-len(speakers))

		print("total utterances data shape:", np.array(self.train_u).shape)
		print("total queries data shape:", np.array(self.train_q).shape)
		print("total speakers data shape:", np.array(self.Speaker).shape)

	def next(self):
		if self.cur_pos + self.batch_size < self.length:
			u = self.train_u[self.cur_pos:self.cur_pos + self.batch_size]
			q = self.train_q[self.cur_pos:self.cur_pos + self.batch_size]			
			y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
			s = self.Speaker[self.cur_pos:self.cur_pos + self.batch_size]
			self.cur_pos += self.batch_size
			return np.array(u), np.array(q), np.array(y), np.array(s)
		else:
			record = self.cur_pos
			self.cur_pos = self.batch_size - self.length + self.cur_pos
			if self.cur_pos == 0:
				return (np.array(self.train_u[record:self.length])
					 , np.array(self.train_q[record:self.length])
					 , np.array(self.train_y[record:self.length])
					 , np.array(self.Speaker[record:self.length]))
			return (np.concatenate((self.train_u[record:self.length], self.train_u[:self.cur_pos]))
				 , np.concatenate((self.train_q[record:self.length], self.train_q[:self.cur_pos]))
				 , np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos]))
				 , np.concatenate((self.Speaker[record:self.length], self.Speaker[:self.cur_pos])))
