# -*- coding:utf-8 -*-
from sklearn.preprocessing import OneHotEncoder
import numpy as np


mention_list = ["I", "you", "You", "he", "He", "She", "she"]
def process(utter):
    res = []
    pos = []
    for query in utter:
        tmp = []
        tmp_pos = []
        record = 0
        last = 0
        for line in query:
            tmp += line
            record += len(line)
            tmp_pos.append(int((record + last) / 2))
            last += len(line)
        res.append(tmp)
        pos.append(tmp_pos)
    return res, pos

def get_length(utter):
    return sum([len(query) for query in utter])

def generate_score(data):
    mentions = []
    entities = []
    entity_class = []
    for query in data:
        sent = []
        mention_pos = []
        entity_pos = []
        start_pos = []
        entity_one_query = []
        for u in query:
            start_pos.append(len(sent))
            sent += u
        for i, word in enumerate(sent):
            if word in mention_list:
                mention_pos.append(i)
            #将speaker去掉
            if "@ent" in word and i not in start_pos:
                entity_pos.append(i)
                entity_one_query.append(int(word[-2:]))
        mentions.append(mention_pos)
        entities.append(entity_pos)
        entity_class.append(entity_one_query)
    
    score = []
    for i, query in enumerate(data):
        sent = []
        score_one_query = []
        for u in query:
            sent += u
        diction = {}
        for j, word in enumerate(sent):
            if "@ent" in word:
                try:
                    diction[word].append(j)
                except:
                    diction[word] = [j]
        for pos in mentions[i]:
            score_one_mention = [0] * 16
            for item in diction.items():
                index = int(item[0][-2:])
                for entity_pos in item[1]:
                    score_one_mention[index] += 10 / abs(entity_pos - pos)
            score_one_query.append(score_one_mention)
        score.append(score_one_query) 
    return mentions, entities, entity_class

def transfer_onehot(x, pronoun_max_num, num_classes):
    res = np.zeros([len(x), num_classes, pronoun_max_num])
    for i in range(len(x)):
        for j in range(len(x[i])):
            if x[i][j] != -1:
                res[i][x[i][j]][j] = 1
    return res


class batch_generator():
        def __init__(self, U, Q, labels, word2id, vocab_size, all_label, batch_size, query_length, utter_length, dialog_length, word_class, pronoun_max_num, entity_max_num):
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
                mentions, entities, entity_class = generate_score(tmp_U)
                tmp_U, pos = process(tmp_U)
                print("max number of mentions: ", max([len(mention) for mention in mentions]))
                print("max number of entities: ", max([len(entity) for entity in entities]))
                for i in range(len(mentions)):
                    for j in range(len(mentions[i]), pronoun_max_num):
                        mentions[i].append(-1)
                    mentions[i] = mentions[i][:pronoun_max_num]
                for i in range(len(entities)):
                    for j in range(len(entities[i]), entity_max_num):
                        entities[i].append(-1)
                    entities[i] = entities[i][:entity_max_num]
                for i in range(len(entity_class)):
                    for j in range(len(entity_class[i]), entity_max_num):
                        entity_class[i].append(-1)
                    entity_class[i] = entity_class[i][:entity_max_num]

                self.mentions = mentions
                self.entities = entities
                self.entity_class = transfer_onehot(entity_class, entity_max_num, len(all_label))
                print(np.array(mentions).shape)
                print(np.array(entities).shape)
                print(np.array(entity_class).shape)

                self.dialog_pos = pos
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
                self.train_y = ohe.transform(np.reshape(labels, [-1, 1])).toarray()     # sample_num * label_size
                for i in range(len(self.train_q)):
                        for _ in range(len(self.train_q[i]), self.q_maxlen):
                                self.train_q[i].append(vocab_size - 1)
                for i in range(len(self.train_u)):
                        for _ in range(len(self.train_u[i]), self.u_maxlen):
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
                        mentions = self.mentions[self.cur_pos:self.cur_pos + self.batch_size]
                        entities = self.entities[self.cur_pos:self.cur_pos + self.batch_size]
                        entity_class = self.entity_class[self.cur_pos:self.cur_pos + self.batch_size]
                        self.cur_pos += self.batch_size
                        return np.array(u), np.array(q), np.array(y), np.array(pos), np.array(mentions), np.array(entities), np.array(entity_class)
                else:
                        record = self.cur_pos
                        self.cur_pos = self.batch_size - self.length + self.cur_pos
                        if self.cur_pos == 0:
                                return np.array(self.train_u[record:self.length]), np.array(self.train_q[record:self.length]), np.array(self.train_y[record:self.length]), np.array(self.dialog_pos[record:self.length]), np.array(self.mentions[record:self.length]), np.array(self.entities[record:self.length]), np.array(self.entity_class[record:self.length])

                        return np.concatenate((self.train_u[record:self.length], self.train_u[:self.cur_pos])), np.concatenate((self.train_q[record:self.length], self.train_q[:self.cur_pos])), np.concatenate((self.train_y[record:self.length], self.train_y[:self.cur_pos])), np.concatenate((self.dialog_pos[record:self.length], self.dialog_pos[:self.cur_pos])), np.concatenate((self.mentions[record:self.length], self.mentions[:self.cur_pos])), np.concatenate((self.entities[record:self.length], self.entities[:self.cur_pos])), np.concatenate((self.entity_class[record:self.length], self.entity_class[:self.cur_pos]))
