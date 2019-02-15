# -*- coding:utf-8 -*-
import json
import numpy as np

def data_helper(trn_file, dev_file, tst_file):
	sentences, Q, U, labels, label2id = generate_sentences(trn_file)
	s, Q_dev, U_dev, labels_dev, tmp = generate_sentences(dev_file)
	ss, Q_tst, U_tst, labels_tst, tt = generate_sentences(tst_file)
	init_num = len(label2id)
	for label in tmp:
		if label not in label2id:
			label2id[label] = init_num
			init_num += 1
	for label in tt:
		if label not in label2id:
			label2id[label] = init_num
			init_num += 1
	print(label2id)
	labels = [label2id[label] for label in labels]
	labels_dev = [label2id[label] for label in labels_dev]
	labels_tst = [label2id[label] for label in labels_tst]
	sentences = sentences + s + ss
	word_ratio = {}
	sum_words = sum([len(line) for line in sentences])
	for line in sentences:
		for word in line:
			if word in word_ratio:
				word_ratio[word] += 1
			else:
				word_ratio[word] = 1
	word_class = {}
	record = 0
	split_0 = 0.05 * len(word_ratio)
	split_1 = 0.3 * len(word_ratio)
	#生成5%，30%对应的词，作剪枝用
	for word, ratio in sorted(word_ratio.items(), key = lambda x : x[1], reverse = True):
		record += 1
		if record < split_0:
			word_class[word] = 0
		elif record < split_1:
			word_class[word] = 1
		else:
			word_class[word] = 2

	return U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class

def generate_sentences(file):
	data = json.load(open(file))
	Q = []
	U = []
	labels = []
	label2id = {}
	for query in data:
		utterances = [ut['speakers'] + ' ' + ut['tokens']for ut in query['utterances']]
		U.append(utterances)
		Q.append(query['query'])
		labels.append(query['answer'])
	for i, label in enumerate(list(set(labels))):
		label2id[label] = i
	sentences = [line.split(' ') for line in Q]
	for utterances in U:
		for line in utterances:
			sentences.append(line.split(' '))
	return sentences, Q, U, labels, label2id


