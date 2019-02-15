# -*- coding:utf-8 -*-
import json
import numpy as np

def data_helper(trn_file, dev_file, tst_file):
	sentences, Q, U, labels = generate_sentences(trn_file)
	s, Q_dev, U_dev, labels_dev = generate_sentences(dev_file)
	ss, Q_tst, U_tst, labels_tst = generate_sentences(tst_file)
	
	label2id = {}
	for i in range(10):
		lab = "@ent0"+ str(i)
		label2id[lab] = i
	for i in range(10, 16):
		lab = "@ent"+str(i)
		label2id[lab] = i

	labels = [label2id[label] for label in labels]
	labels_dev = [label2id[label] for label in labels_dev]
	labels_tst = [label2id[label] for label in labels_tst]
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

	#优先过滤测试集中没有在训练集里出现的词
	for line in s:
		for word in line:
			if not word in word_class:
				word_class[word] = 0
	for line in ss:
		for word in line:
			if not word in word_class:
				word_class[word] = 0

	for i in range(10):
		word_class["@ent0"+str(i)] = 2
	for i in range(10,16):
		word_class["@ent"+str(i)] = 2

	word_class["@placeholder"] = 2


	return U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class

def generate_sentences(file):
	data = json.load(open(file))
	Q = []
	U = []
	labels = []
	for query in data:
		utterances = [ut['speakers'] + ' ' + ut['tokens']for ut in query['utterances']]
		U.append(utterances)
		Q.append(query['query'])
		labels.append(query['answer'])
	sentences = [line.split(' ') for line in Q]
	for utterances in U:
		for line in utterances:
			sentences.append(line.split(' '))
	return sentences, Q, U, labels


