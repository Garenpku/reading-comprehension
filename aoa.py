import tensorflow as tf

"""
使用此函数得到attention_over_attention中的attention
d为Dialog使用bilstm/biGRU得到的矩阵，shape = [batch_size * utterance_length * hidden_length]
q为Query使用bilstm/biGRU得到的矩阵，shape = [batch_size * query_length * hidden_length]
返回的是与utterance维度相同的attention向量，shape = [batch_size * utterance_length]
"""
def get_aoa_attention(d, q, batch_size): 
    print(d.shape.as_list())
    print(q.shape.as_list())
    tmp_d = tf.unstack(d)
    tmp_q = tf.unstack(q)
    sim = [tf.matmul(tmp_d[i], tmp_q[i], transpose_b = True) for i in range(batch_size)]
    alpha = [tf.nn.softmax(sim[i], axis = 0) for i in range(batch_size)]
    beta = [tf.nn.softmax(sim[i], axis = 1) for i in range(batch_size)]
    beta = [tf.expand_dims(tf.reduce_mean(sim[i], axis = 0), axis = -1) for i in range(batch_size)]
    attention = tf.squeeze(tf.stack([tf.matmul(alpha[i], beta[i]) for i in range(batch_size)]))
    print("shape of the attention:", attention.shape.as_list())
    return attention
