import json
import re


def substitude_utterances(entry, raw):
	for i, utter in enumerate(entry['utterances']):
		#print(raw['utterances'][i])
		#print(entry['utterances'][i])
		if raw['utterances'][i]['tokens_with_note'] != None:
			transcript = ' '.join(concat(raw['utterances'][i]['tokens_with_note']))
		else:
			transcript = ' '.join(concat(raw['utterances'][i]['tokens']))
		entry['utterances'][i]['tokens'] = transcript
		if len(raw['utterances'][i]['speakers']) > 0:
			entry['utterances'][i]['speakers'] = raw['utterances'][i]['speakers'][0]
		else:
			entry['utterances'][i]['speakers'] = ''

def substitude_query_and_answer(entry, raw):
	query = entry['query']
	entities = re.findall('(@ent\d*)', query)
	for entity in entities:
		character = map_character(entity, entry, raw)
		if character != None:
			entry['query'] = re.sub(entity, character, entry['query'])
		else:
			print("No match!", entity, query)
	character = map_character(entry['answer'], entry, raw)
	if not character:
		print("===============\nAnswer not annotated as entity!\n================")
		return False
	entry['answer'] = re.sub(entry['answer'], character, entry['answer'])
	return True

def map_character(entity, entry, raw): # entry:dev.json里的一个query, raw:原数据里对应的一个scene
	#print(entity)
	for i, utter in enumerate(entry['utterances']):
		if entity == utter['speakers']: 
			res = raw['utterances'][i]['speakers'][0]
			#print(res)
			return res
		#print(utter['tokens'])
		if entity in utter['tokens']:
			raw_text = find_raw_text(entity, utter, raw['utterances'][i])
			result = select_entity(raw_text, raw['utterances'][i])
			if result:
				return result

def find_raw_text(entity, query_utter, raw_utter):
	#print("find raw text!")
	query_sent = query_utter['tokens'].split(' ')
	if raw_utter['tokens_with_note']:
		raw_sent = concat(raw_utter['tokens_with_note'])
	else:
		raw_sent = concat(raw_utter['tokens'])
	start = 0
	#print(query_sent)
	#print(raw_sent)
	for i, word in enumerate(query_sent):
		if '@ent' in word:
			cnt = 1
			if i + 1 == len(query_sent):
				cnt = len(raw_sent) - start
			while cnt + start < len(raw_sent):
				if raw_sent[start + cnt] == query_sent[i + 1]: 
					end = start + cnt
					break
				cnt += 1
			end = start + cnt
			if word == entity:
				#print(raw_sent[start:end])
				return raw_sent[start : end]
			start += cnt
		else:
			start += 1
	#print("oops, not found!")

def select_entity(raw_words, raw_utter):
	raw_sent = []
	entity_pos = []
	cur_len = 0
	for i, line in enumerate(raw_utter['tokens']):
		raw_sent += line
		for j, entity in enumerate(raw_utter['character_entities'][i]):
			raw_utter['character_entities'][i][j][0] += cur_len
			raw_utter['character_entities'][i][j][1] += cur_len
		entity_pos += raw_utter['character_entities'][i]
	#print(raw_sent)
	word_num = len(raw_words)
	record = -1
	for i in range(len(raw_sent) - word_num + 1):
		if raw_sent[i : i + word_num] == raw_words:
			record = i
	if record == -1:
		return None
	for pos in entity_pos:
		if pos[0] == record and pos[1] == record + word_num:
			print("entity found! ", pos[2])
			return pos[2]

def concat(sent_list):
	res = []
	for l in sent_list:
		res += l
	return res

trn = json.load(open("tst.json"))
raw = []
final = []
count = 0
for i in range(1, 5):
	data = json.load(open('./raw/friends_season_0' + str(i) + '.json'))
	raw.append(data)
for entry in trn:
	scene_id = entry['scene_id']
	index = re.findall('s(\d*)_e(\d*)_c(\d*)', scene_id)[0]
	#print(index)
	season = int(index[0]) - 1
	if season >= 4:
		break
	episode = int(index[1]) - 1
	scene = int(index[2]) - 1
	raw_dialog = raw[season]['episodes'][episode]['scenes'][scene]
	flag = substitude_query_and_answer(entry, raw_dialog)
	substitude_utterances(entry, raw_dialog)
	if flag:
		count += 1
		final.append(entry)
print(len(final), "queries left.")

json.dump(final, open("tst_after.json", 'w'))



