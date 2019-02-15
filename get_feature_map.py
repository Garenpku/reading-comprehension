import numpy as np 
'''
这个函数是用来处理得到mention的feature map， 很多地方因为需要统一化进行了padding， 先用的全零，但是为了更合理应当换成所有
词向量的平均值
映射的时候，padding出来的mention暂时映射到了字典中的0号条目， 可以通过nobody_entity的值进行修改

character_dict那里就用你开始简单跑的时候的人名到id的映射吧，统一一下，但是我没有那段代码不太清楚映射方式，需要麻烦你加一下
utterances的内容与json data里面的结构完全相同，后面需要整合到batch_generator里面就先没有写外面调用函数的循环代码
以及 word2id（）函数， 就是把一个词 转换成它的嵌入向量（不是编号）， 函数名字起得有点迷， 
这里我没有太看懂你的embedding那里的代码， 所以这个函数麻烦加一下啦
'''

#将人名映射到id的那个表
character_dict = dict()
mentions_num = 50
nobody_entity = 0 # padding 的条目对应的entity在字典里的编号

sentence_length = embedding_dim = 50
padding_sentence_embedding = padding_word_embedding = [0]*embedding_dim #这里应当换成所有vocab的embedding的平均值的那个向量
padding_feature_map = [[padding_sentence_embedding]*3]*4

#对一个scene中的utterance进行处理
def get_feature_map(utterances):
	def tokens2tensor(sentence):
		re = [word2id(obj) for obj in sentence]
		return np.mean(re, axis = 0)
	def utter2tensor(utter):
		re = [tokens2tensor(tokens) for tokens in utter]
		return np.mean(re, axis = 0)
	feature_maps = []
	labels = []
	for utter_id, utter in enumerate(utterances):
		sentences = utter["tokens"]
		for senten_id, mentions in enumerate(utter['character_entities']):
			senten_token = sentences[senten_id]
			for mention_id, mention in enumerate(mentions):
				mention_embed = []
				mention_stat, mention_end, character  = mention
				fm1 = [word2id(senten_token(i)) for i in range(mention_stat, min(mention_end, mention_stat+3))]
				fm1 = fm1 + [padding_word_embedding]*(3-len(fm1))

				fm21 = [word2id(senten_token[i]) for i in range(max(mention_stat-3, 0), mention_stat)]
				fm21 = np.mean(fm21 + [padding_word_embedding]*(3-len(fm21)), axis = 0)
				fm22 = [word2id(senten_token[i]) for i in range(mention_end, min(len(senten_token), mention_end+3))]
				fm22 = np.mean(fm22 + [padding_word_embedding]*(3-len(fm22)), axis = 0)
				fm23 = np.mean([word2id(senten_token[i]) for i in range(mention_stat, mention_end)])

				fm2 = [fm21, fm22, fm23]

				fm31 = [tokens2tensor(sentences[i]) for i in range(max(0, senten_id-3), senten_id)]
				fm31 = np.mean(fm31+[padding_sentence_embedding]*(3-len(fm31)))
				if senten_id+1 == len(sentences):
					fm32 = padding_sentence_embedding
				else:
					fm32 = tokens2tensor(sentences[senten_id+1])
				fm33 = tokens2tensor(sentences[senten_id])
				fm3 = [fm31, fm32, fm33]

				fm41 = [utter2tensor(utterances[i]["tokens"]) for i in range(max(0, utter_id-3), utter_id)]
				fm41 = np.mean(fm41+[padding_sentence_embedding]*(3-len(fm41)))
				if utter_id+1 == len(utterances):
					fm42 = padding_sentence_embedding
				else:
					fm42 = utter2tensor(utterances[senten_id+1]["tokens"])
				fm43 = utter2tensor(utterances[senten_id]"tokens")
				fm4 = [fm41, fm42,fm43]
				feature_maps.append([fm1, fm2, fm3, fm4])

				labels.append(character_dict[character])

	feature_maps = feature_maps + [padding_feature_map]*(mentions_num - len(feature_maps))
	labels = labels + [character_dict[nobody_entity]]*(mentions_num - len(labels))
	return feature_maps, labels






