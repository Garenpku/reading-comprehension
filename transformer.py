# -*- coding:utf-8 -*-
import tensorflow as tf 
import json
import keras
from keras.layers import TimeDistributed, Dense, Conv2D, Conv1D,LSTM,Bidirectional
from data_helper import data_helper
from word2vec import word2vec
from batch_generator_long import batch_generator
from keras.layers import MaxPooling2D
from attentive_reader import attentive_reader
from glove_embedding import glove_embedding
import numpy as np
import pickle
import time
import os

def transformer(utter, utter_length, batch_size, head_num, hidden_length, embedding_dim):
    # batch_size * utter_length * 
    def one_head(utter):
        query = TimeDistributed(Dense(hidden_length, activation = "relu"))(utter)
        keys = TimeDistributed(Dense(hidden_length, activation = "relu"))(utter)
        value = TimeDistributed(Dense(hidden_length, activation = "relu"))(utter)
        att = tf.matmul(keys, value, transpose_b = True)
        att = att / 8
        att = tf.nn.softmax(att, axis = -1)
        res = tf.reshape(tf.matmul(att, value), [batch_size, utter_length, hidden_length])
        return res
    result = []
    for i in range(head_num):
        result.append(one_head(utter))
    res = tf.reshape(tf.transpose(tf.stack(result), [1, 2, 0, 3]), [batch_size, utter_length, head_num * hidden_length])
    res = tf.reshape(TimeDistributed(Dense(embedding_dim, activation = "relu"))(res), [batch_size, utter_length, embedding_dim])
    res = res + utter
    mean = tf.expand_dims(tf.reduce_mean(res, axis = -1), axis = -1)
    std = tf.sqrt(tf.expand_dims(tf.reduce_mean(tf.square((res - mean)), axis = -1), axis = -1))
    res = (res - mean) / std
    res = tf.reshape(TimeDistributed(Dense(embedding_dim, activation = "relu"))(res), [batch_size, utter_length, embedding_dim])
    print("shape of final:", res.shape.as_list())
    return res
    
