import numpy as np

def glove_embedding(U, Q, embedding_dim = 100):
	src = open("./glove.6B/glove.6B." + str(embedding_dim) + "d.txt", encoding = 'utf-8').readlines()
	embed_dict = {}
	for line in src:
		info = line[:-1].split(' ')
		embed_dict[info[0]] = info[1:]
	word2id, id2word, vocab_size = build_dataset(U, Q)
	embedding_matrix = []
	cnt = 0
	for word in word2id:
		if word in embed_dict:
			embedding_matrix.append(embed_dict[word])
		else:
			embedding_matrix.append(list(np.random.rand(embedding_dim)))
			cnt += 1
	print(cnt, "words not in dictionary")
	return embedding_matrix, word2id, id2word

def build_dataset(U, Q):
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

	return word2id, id2word, vocab_size