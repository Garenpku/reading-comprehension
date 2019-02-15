import tensorflow as tf

def attentive_reader(utter, query, hidden_length, attention_length, batch_size):
    W_utter = tf.Variable(tf.truncated_normal([2 * hidden_length, attention_length], -1, 1))
    W_query = tf.Variable(tf.truncated_normal([2 * hidden_length, attention_length], -1, 1))
    W_ms = tf.Variable(tf.truncated_normal([attention_length, 1], -1, 1))
    hq = tf.unstack(query)
    bilstm_result = tf.unstack(utter)
    M = []
    for i in range(batch_size):
        M.append(tf.nn.tanh(tf.matmul(W_query, tf.expand_dims(hq[i], axis = -1), transpose_a = True) + tf.matmul(W_utter, bilstm_result[i], transpose_a = True, transpose_b = True)))
    S = [tf.matmul(m, W_ms, transpose_a = True) for m in M]
    S = [tf.nn.softmax(s, axis = 0) for s in S]
    r = tf.squeeze(tf.stack([tf.matmul(bilstm_result[i], S[i], transpose_a = True) for i in range(batch_size)])) #batch_size * hidden_length
    return r
