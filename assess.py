import json

data = json.load(open("wrong_answers_dev.txt"))
print("length: ", len(data))
for i in range(len(data)):
    print(data[i]['scene_id'])
    for ut in data[i]['utterances']:
        print(ut['speakers'], end = ",")
        print(ut['tokens'])
    print(data[i]['query'])
    print(data[i]['answer'])
    print(data[i]['Predicted'])
    print(data[i]['GroundTruth'])
    ch = input("continue?")
    if ch != '1':
        break
