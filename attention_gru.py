from attention_func import *
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
attnum = 0
class Attention_Gru(object):
	def __init__(self, input_w, attention_w, hidden_size =None):
		global attnum
		self.input_shape = input_w.shape.as_list()
		self.attention_shape = attention_w.shape.as_list()
		if hidden_size == None:
			self.hidden_size = self.attention_shape[-1]
		else:
			self.hidden_size = hidden_size
		self.batch_size = self.input_shape[0]
		self.att = Attention(attention_w, self.input_shape[-1], self.hidden_size)
		self.cell = GRUCell(self.hidden_size)
		self.input_list = tf.unstack(input_w, axis = 1)
		self.cnt = 0
		attnum+=1

	def get_result(self, return_sequences = False):
		global attnum
		attention_shape = self.attention_shape
		hidden_size = self.hidden_size
		batch_size = self.batch_size

		def loop_fn_initial():
			initial_elements_finished = False
			#h1 = tf.stack([tf.Variable(tf.truncated_normal([input_shape[1]]))]*batch_size)
			h2 = tf.stack([tf.Variable(tf.truncated_normal([hidden_size]))]*batch_size)
			step_input = self.input_list[self.cnt]
			self.cnt += 1
			initial_input = tf.concat([step_input, (self.att).get_append(step_input, h2)], axis = -1)
			initial_cell_state = h2
			initial_cell_output = None
			initial_loop_state = None
			return (initial_elements_finished,
					initial_input,
					initial_cell_state,
					initial_cell_output,
					initial_loop_state)
		def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
			element_finished = (time >= self.input_shape[1])
			ori_input = self.input_list[self.cnt]
			self.cnt += 1
			inputs = tf.concat([ori_input, (self.att).get_append(ori_input, previous_state)], axis = -1)
			state = previous_state
			output = previous_output
			loop_state = None
			return (element_finished,
					inputs,
					state,
					output,
					loop_state)
		def loop_fn(time, previous_output, previous_state, previous_loop_state):
			if self.cnt == 0:
				return loop_fn_initial()
			else:
				return loop_fn_transition(time, previous_output,previous_state,previous_loop_state)
		with tf.variable_scope("raw_rnn_"+str(attnum), reuse = tf.AUTO_REUSE) as scope:
			outputs_ta, final_state,_ = tf.nn.raw_rnn(self.cell, loop_fn)
		outputs = outputs_ta.stack()
		outputs = tf.reshape(outputs, [self.input_shape[0], self.input_shape[1], self.hidden_size])

		if return_sequences:
			return outputs
		else:
			return tf.transpose(outputs, [1,0,2])[-1]

def Attention_BiGru(input_w, attention_w, hidden_size = None, return_sequences = False):
	fwd = Attention_Gru(input_w,attention_w, hidden_size).get_result(return_sequences = return_sequences)
	input_w2 = tf.reverse(input_w, axis = [-2])
	bwd = Attention_Gru(input_w2,attention_w, hidden_size).get_result(return_sequences = return_sequences)
	re = tf.concat([fwd, bwd], axis = -1)
	return re
