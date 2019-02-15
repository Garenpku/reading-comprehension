import json

data = json.load(open('tst.json', encoding = 'utf-8'))
stop_list = []
with open('stopwords.txt', encoding = 'utf-8') as stop:
	for stop_word in stop:
		stop_list.append(stop_word[:-1])
word_dict = {}
print(len(data))
for i, query in enumerate(data):
	for j, utterance in enumerate(query['utterances']):
		words = utterance['tokens'].split(' ')
		words = [word for word in words if word not in stop_list]
		data[i]['utterances'][j]['tokens'] = ' '.join(words)
	if i % 100 == 0:
		print(i)
json.dump(data, open('tst_clean.json', 'w'))