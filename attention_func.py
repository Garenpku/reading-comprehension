import tensorflow as tf
class Attention(object):
	def __init__(self, attention_matrix, vec1_length, vec2_length, v_length = 64):
		shape_m = attention_matrix.shape.as_list()
		#print(shape_m)
		#attention_matrix: [bs, k, dim], bs unknown
		self.attention_matrix = attention_matrix
		self.w = tf.Variable(tf.truncated_normal([shape_m[2], v_length]), dtype = tf.float32)
		self.w1 = tf.Variable(tf.truncated_normal([vec1_length, v_length]), dtype = tf.float32)
		self.w2 = tf.Variable(tf.truncated_normal([vec2_length, v_length]), dtype = tf.float32)
		self.v = tf.Variable(tf.truncated_normal([v_length, 1]), dtype = tf.float32)
		self.k = shape_m[1]
		self.vec1_length = vec1_length
		self.vec2_length = vec2_length

		self.v_length = v_length

		self.part1 = tf.reshape(tf.matmul(tf.reshape(attention_matrix, [-1, shape_m[2]]), self.w), [-1, shape_m[1], v_length])

	def get_append(self, _vec1, _vec2):
		with tf.variable_scope("cal_next_state", reuse = tf.AUTO_REUSE):
			#_vec1: [bs, vec1_length]
			#_vec2: [bs, vec2_length]
			vec1 = tf.reshape(tf.matmul(tf.reshape(_vec1, [-1, self.vec1_length]), self.w1), [-1, 1, self.v_length])
			vec2 = tf.reshape(tf.matmul(tf.reshape(_vec2, [-1, self.vec2_length]), self.w2), [-1, 1, self.v_length])

			#这里v如果过大可能会出问题，出现溢出的话self.v改成tf.tanh(self.v)
			#coefficient： [bs, k, 1]
			coeffient = tf.reshape(tf.nn.softmax((tf.matmul(tf.reshape(tf.tanh(self.part1 + vec1 + vec2), [-1, self.v_length]), self.v))), [-1,self.k,1])
			return tf.reduce_sum((self.attention_matrix * coeffient), axis = 1)



