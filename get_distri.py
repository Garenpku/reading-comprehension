# -*- coding:utf-8 -*-

from batch_generator_long import batch_generator
import pickle as pk 
from data_helper import data_helper
import os
U, U_dev, U_tst, Q, Q_dev, Q_tst, labels, labels_dev, labels_tst, label2id, word_class = data_helper('trn_cleaned_after.json', 'dev_cleaned_after.json', 'tst_cleaned_after.json')

def fff(setname, all_label):
	label_dict = {}
	for obj in all_label:
		if obj in label_dict:
			label_dict[obj] += 1
		else:
			label_dict[obj] = 1
	f = open("label_distri"+setname+".txt", 'wb')
	pk.dump(label_dict, f)
	f.close()

fff("_trn",labels)
fff("_dev",labels_dev)
fff("_tst",labels_tst)
fff("", labels+labels_dev+labels_tst)

f = open("label2id.txt", "wb")
pk.dump(label2id,f)
f.close()

