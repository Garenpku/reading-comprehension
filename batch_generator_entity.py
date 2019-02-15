# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def process(utter, label):
	res = []
	pos = []
	label_res = []
	for query in utter:
		label_line = [[] for i in range(len(label))]
		tmp = []
		tmp_pos = []
		record = 0
		last = 0
		for line in query:
			for i, word in enumerate(line):
				if word in label and i != 0:
					label_line[label.index(word)].append(record + i)
			tmp += line
			record += len(line)
			tmp_pos.append(int((record + last) / 2))
			last += len(line)
		label_res.append(label_line)		
		res.append(tmp)
		pos.append(tmp_pos)
	return res, pos, label_res

def get_length(utter):
	return sum([len(query) for query in utter])

class batch_generator():
	def __init__(self, U, Q, labels, word2id, vocab_size, all_label, batch_size, query_length, utter_length, dialog_length, word_class, label2id):
		tmp_U = [[[word for word in line.split(' ')] for line in query] for query in U]
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
		#得到label的位置
		label = []
		for one in sorted(label2id.items(), key = lambda x : x[1]):
			label.append(one[0])

		tmp_U, pos, label_pos = process(tmp_U, label)
		"""
		for line in tmp_U:
			pos_line = []
			for char in label:
				pos_char = []
				for position, word in enumerate(line):
					if word == char:
						pos_char.append(position)
				pos_line.append(pos_char)
			label_pos.append(pos_line)
		"""
		self.label_pos = label_pos
		print('label pos:', label_pos[0])
		for i, label in enumerate(label_pos[0]):
			print(i)
			for posit in label:
				print(label2id[tmp_U[0][posit]])

		self.dialog_pos = pos
		self.train_u = [[word2id[word] for word in line] for line in tmp_U] 
		imax = max([len(line) for line in tmp_U])
		print("max length of utterances:", imax)
		self.train_q = [[word2id[word] for word in line] for line in tmp_Q]
		self.batch_size = batch_size
		self.length = len(self.train_u)
		self.cur_pos = 0
		ohe = OneHotEncoder().fit(np.reshape(all_label, [-1, 1]))
		self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()	# sample_num * label_size
		for i in range(len(self.train_q)):
			for _ in range(len(self.train_q[i]), query_length):
				self.train_q[i].append(vocab_size - 1)
		for i in range(len(self.train_u)):
			for _ in range(len(self.train_u[i]), utter_length):
				self.train_u[i].append(vocab_size - 1)
		for i in range(len(self.train_u)):
			for _ in range(len(self.dialog_pos[i]), dialog_length):
				self.dialog_pos[i].append(-1)

	def next(self):
		if self.cur_pos + self.batch_size < self.length:
			u = self.train_u[self.cur_pos:self.cur_pos + self.batch_size]
			q = self.train_q[self.cur_pos:self.cur_pos + self.batch_size]			
			y = self.train_y[self.cur_pos:self.cur_pos + self.batch_size]
			pos = self.dialog_pos[self.cur_pos:self.cur_pos + self.batch_size]
			l = self.label_pos[self.cur_pos:self.cur_pos + self.batch_size]
			self.cur_pos += self.batch_size
			return np.array(u), np.array(q), np.array(y), pos, l
		else:
			record = self.cur_pos
			self.cur_pos = self.batch_size - self.length + self.cur_pos
			if self.cur_pos == 0:
				return np.array(self.train_u[record:self.length]), np.array(self.train_q[record:self.length]), np.array(self.train_y[record:self.length]), np.array(self.dialog_pos[record:self.length]), np.array(self.label_pos[record:self.length])

			return np.concatenate((self.train_u[record:self.length], self.train_u[:self.cur_pos])), np.concatenate((self.train_q[record:self.length], self.train_q[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos])), np.concatenate((self.dialog_pos[record:self.length], self.dialog_pos[:self.cur_pos])), np.concatenate((self.label_pos[record:self.length], self.label_pos[:self.cur_pos]))
