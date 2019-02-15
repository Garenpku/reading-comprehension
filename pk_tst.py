# -*- coding:utf-8 -*-
import pickle as pk 

with open("label2id.txt", 'rb') as f:
	label_dict = pk.load(f)
tmp = zip(label_dict.values(), label_dict.keys())
labelid = list(sorted(tmp))

re = {}
for obj in labelid:
	tdic = {"name":obj[1], "all_cnt":0,"trn_cnt": 0, "dev_cnt":0, "tst_cnt":0}
	re[obj[0]] = tdic

def fff(filename, index):
	with open(filename+".txt", 'rb') as f:
		label_dict = pk.load(f)
	s = 0
	for obj in label_dict:
		re[obj][index] = label_dict[obj]
		s += label_dict[obj]

fff("label_distri", "all_cnt")
fff("label_distri_trn", "trn_cnt")
fff("label_distri_dev", "dev_cnt")
fff("label_distri_tst", "tst_cnt")

print(re[label_dict["by"]])
print(re[label_dict["and"]])
print(re[label_dict["for"]])
print(re[label_dict["as"]])
print(re[label_dict["tell"]])
print(re[label_dict["from"]])
print(re[label_dict[","]])
print(re[label_dict["."]])
print(re[label_dict["..."]])
print(re[label_dict["With"]])
print(re[label_dict["are"]])
print(re[label_dict["about"]])
print(re[label_dict["("]])
print(re[label_dict["["]])
print(re[label_dict["the"]])
print(re[label_dict["this"]])
print(re[label_dict["to"]])
print(re[label_dict["what"]])
print(re[label_dict["where"]])
print(re[label_dict["with"]])