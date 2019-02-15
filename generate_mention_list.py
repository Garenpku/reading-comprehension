import json
import re

raw = []
for i in range(1, 5):
    data = json.load(open('./raw/friends_season_0' + str(i) + '.json'))
    raw.append(data)

def get_index(s):
    index = re.findall('s(\d*)_e(\d*)_c(\d*)', s)[0]
    return index

def token_merge(tokens, character, cur_pos):
    cur = []
    char_pos = []
    for i, token_single in enumerate(tokens):
        try:
            for char in character[i]:
                char_pos.append((char[0] + len(cur) + cur_pos, char[1] + len(cur) + cur_pos))
        except:
            pass
        cur += token_single
    return char_pos, cur_pos + len(cur), cur
    """
    else:
        note = tokens_with_note[:len(tokens_with_note[0]) - len(tokens[0])]
        cur += note
        for i, token_single in enumerate(tokens):
            for char in character[i]:
                char_pos.append((char[0] + len(cur), char[1] + len(cur)))
            cur += token_single
    """

data = json.load(open('tst_my.json'))

data_cleaned = []
mention_list = []
for query in data:
    index = get_index(query['scene_id'])
    print(index)
    season_id = int(index[0]) - 1
    episode_id = int(index[1]) - 1
    scene_id = int(index[2]) - 1
    scene = raw[season_id]['episodes'][episode_id]['scenes'][scene_id]
    cur_pos = 0
    mention_scene = []
    all_tokens = []
    for utter in scene['utterances']:
        if utter['tokens'] != []:
            char_pos, cur_pos, cur = token_merge(utter['tokens'], utter['character_entities'], cur_pos) 
        else:
            char_pos, cur_pos, cur = token_merge(utter['tokens_with_note'], utter['character_entities'], cur_pos) 

        mention_scene += char_pos
        all_tokens += cur
    mention_list += mention_scene
    for mention in mention_scene:
        print(" ".join(all_tokens[mention[0]: mention[1]]))
    #test
    all_tokens = []
    for utter in query['utterances']:
        all_tokens += utter['tokens'].split(' ')
    print("======")
    for mention in mention_scene:
        print(" ".join(all_tokens[mention[0]: mention[1]]))
    print("======")
"""
data = json.load(open('dev_cleaned_after.json'))
for query in data:
    index = get_index(query['scene_id'])
    season_id = int(index[0]) - 1
    episode_id = int(index[1]) - 1
    scene_id = int(index[2]) - 1
    if season_id >= 4:
        continue
    scene = raw[season_id]['episodes'][episode_id]['scenes'][scene_id]
    #extra work
    for i, utter in enumerate(scene['utterances']):
        if utter['tokens'] != []:
            tmp = []
            for token in utter['tokens']:
                tmp += token
            query['utterances'][i]['tokens'] = ' '.join(tmp)
        else:
            tmp = []
            for token in utter['tokens_with_note']:
                tmp += token
            query['utterances'][i]['tokens'] = ' '.join(tmp)

    data_cleaned.append(query)
json.dump(data_cleaned, open("dev_my.json", "w"))
"""
