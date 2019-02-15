import json
import numpy as np

data = json.load(open("dev.json"))
mention_list = ["I", "you", "You", "he", "He", "She", "she"]
mentions = []
for query in data:
    sent = []
    mention_pos = []
    for u in query["utterances"]:
        sent += u["tokens"].split(' ')
    for i, word in enumerate(sent):
        if word in mention_list:
            mention_pos.append(i)
    mentions.append(mention_pos)
print(mentions[:10])
score = []
for i, query in enumerate(data):
    sent = []
    score_one_query = []
    for u in query["utterances"]:
        sent += u["tokens"].split(' ')
    diction = {}
    for j, word in enumerate(sent):
        if "@ent" in word:
            try:
                diction[word].append(j)
            except:
                diction[word] = [j]
    if i == 0:
        print(diction)
    for pos in mentions[i]:
        score_one_mention = [0] * 16
        for item in diction.items():
            index = int(item[0][-2:])
            if i == 0:
                print(index)
            for entity_pos in item[1]:
                score_one_mention[index] += 10 / abs(entity_pos - pos)
        score_one_query.append(score_one_mention)
    score.append(score_one_query) 
print(score[0])
