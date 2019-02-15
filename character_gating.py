import tensorflow as tf

"""
hidden_layer.shape = [batch_size, hidden_size]
"""
def character_gating(hidden_layer, gating_size, hidden_size, num_classes, batch_size):
	w1 = tf.Variable(tf.truncated_normal([gating_size, hidden_size], -1, 1))
	w2 = tf.Variable(tf.truncated_normal([gating_size, gating_size], -1, 1))
	C = tf.Variable(tf.truncated_normal([gating_size, num_classes], -1, 1))
	b = tf.Variable(tf.truncated_normal([gating_size, 1], -1, 1))
	product1 = tf.matmul(hidden_layer, w1, transpose_b = True) # batch_size * gating_size
	product2 = tf.matmul(w2, C) # gating_size * num_classes

	product1 = [tf.matmul(tf.expand_dims(line, axis = -1), tf.ones([1, num_classes])) for line in tf.unstack(product1)]
	product2 = tf.unstack(tf.expand_dims(product2, axis = 0)) * batch_size
	product3 = tf.matmul(b, tf.ones([1, num_classes]))
	print("product1:", tf.stack(product1).shape.as_list())
	print("product2:", tf.stack(product2).shape.as_list())
	
	g = [product1[i] + product2[i] + product3 for i in range(batch_size)] # bs * gs * nc
	mean = [tf.reshape(tf.reduce_mean(g[i], axis = 1), [-1, 1]) for i in range(batch_size)]
	var = [tf.reshape(tf.reduce_mean((g[i] - mean[i]) ** 2, axis = 1), [-1, 1]) for i in range(batch_size)]
	g = [tf.nn.sigmoid((g[i] - mean[i]) / tf.sqrt(var[i])) for i in range(batch_size)]
	C_gated = [g[i] * C for i in range(batch_size)] # batch_size * gating_size * num_classes

	wo = tf.Variable(tf.truncated_normal([hidden_size, gating_size], -1, 1))
	bo = tf.Variable(tf.truncated_normal([gating_size, 1], -1, 1))
	
	res1 = tf.unstack(tf.expand_dims(tf.matmul(hidden_layer, wo), axis = -1)) # batch_size * gating_size * 1
	res1 = tf.squeeze(tf.stack([tf.matmul(C_gated[i], res1[i], transpose_a = True) for i in range(batch_size)]))
	res2 = tf.squeeze(tf.stack([tf.matmul(C_gated[i], bo, transpose_a = True) for i in range(batch_size)]))
	prediction = res1 + res2
	return prediction
