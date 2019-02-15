import tensorflow as tf

def attention_pos(pos, attention, batch_size, num_classes):
	#attention.shape = [batch_size, utter_length]
	#pos.shape = [batch_size, num_classes, None]
	#return shape: [batch_size, num_classes]
	res = tf.zeros([batch_size, num_classes], tf.float32)
	for i, line in enumerate(pos):
		for j, char in enumerate(line):
			for pos in char:
				if pos != 0:
					res[i][j] += attention[pos]	
	return res
