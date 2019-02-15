import json
import pickle

char_dict = {}
for i in range(1, 5):
    if i < 10:
        string = '0' + str(i)
    else:
        string = str(i)
    data = json.load(open('./raw/friends_season_' + string +  '.json'))
    for episode in data['episodes']:
        for scene in episode['scenes']:
            for utterance in scene['utterances']:
                if "character_entities" not in utterance:
                    continue
                for j, token in enumerate(utterance['tokens']):
                    for character in utterance['character_entities'][j]:
                        if character[2] not in char_dict:
                            char_dict[character[2]] = [' '.join(token[character[0]:character[1]])] 
                        else:
                            char_dict[character[2]].append(' '.join(token[character[0]:character[1]]))

    print("season" + str(i) + "finished")
for item in char_dict:
    char_dict[item] = list(set(char_dict[item]))
pickle.dump(char_dict, open("entity", "wb"))
                        

