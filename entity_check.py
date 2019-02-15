import json

a = json.load(open("tst.json"))
cnt = 0
for query in a:
	speaker_list = []
	for utter in query["utterances"]:
		speaker_list.append(utter["speakers"])
	if query["answer"] not in speaker_list:
		print(query["answer"])
		print(set(speaker_list))
		print("========")
		cnt += 1
print(cnt)