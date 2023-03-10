import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
import math
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import gensim.models
import pdb
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler,LabelEncoder
import pandas as pd
import seaborn as sns
import os
import numpy as np
import re
import tensorflow as tf
import random
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow import keras
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.preprocessing import StandardScaler
from itertools import repeat
from collections import defaultdict
import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from sklearn.preprocessing import MinMaxScaler
import pickle


tf.keras.backend.set_floatx('float32')

class FM_layer(tf.keras.layers.Layer):
    def __init__(self, num_feature, num_field, embedding_size, field_index):
        super(FM_layer, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # f: 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        # Parameters of FM Layer
        # w: capture 1st order interactions
        # V: capture 2nd order interactions
        self.w = tf.Variable(tf.random.normal(shape=[num_feature],
                                              mean=0.0, stddev=1.0), name='w')
        self.V = tf.Variable(tf.random.normal(shape=(num_field, embedding_size),
                                              mean=0.0, stddev=1.0), name='V')

    def call(self, inputs):
        x_batch = tf.reshape(inputs, [-1, self.num_feature, 1])
        # Parameter V를 field_index에 맞게 복사하여 num_feature에 맞게 늘림
        embeds = tf.nn.embedding_lookup(params=self.V, ids=self.field_index)

        # Deep Component에서 쓸 Input
        # (batch_size, num_feature, embedding_size)
        new_inputs = tf.math.multiply(x_batch, embeds)

        # (batch_size, )
        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.w, inputs), axis=1, keepdims=False)

        # (batch_size, )
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(new_inputs, [1, 2])),
            tf.reduce_sum(tf.square(new_inputs), [1, 2])
        )

        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])

        y_fm = tf.concat([linear_terms, interactions], 1)

        return y_fm, new_inputs

class DeepFM(tf.keras.Model):

    def __init__(self, num_feature, num_field, embedding_size, field_index, mode):
        super(DeepFM, self).__init__()
        self.embedding_size = embedding_size    # k: 임베딩 벡터의 차원(크기)
        self.num_feature = num_feature          # f: 원래 feature 개수
        self.num_field = num_field              # m: grouped field 개수
        self.field_index = field_index          # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        self.fm_layer = FM_layer(num_feature, num_field, embedding_size, field_index)

        self.layers1 = tf.keras.layers.Dense(units=150, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=0.2)
        self.layers2 = tf.keras.layers.Dense(units=150, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=0.2)
        self.layers3 = tf.keras.layers.Dense(units=150, activation='relu')
        if mode == 'classification':
            self.final = tf.keras.layers.Dense(units=1, activation='sigmoid')
        elif mode == 'regression':
            self.final = tf.keras.layers.Dense(units=1)

    def __repr__(self):
        return "DeepFM Model: #Field: {}, #Feature: {}, ES: {}".format(
            self.num_field, self.num_feature, self.embedding_size)

    def call(self, inputs):
        # 1) FM Component: (num_batch, 2)
        y_fm, new_inputs = self.fm_layer(inputs)
        # retrieve Dense Vectors: (num_batch, num_feature*embedding_size)
        new_inputs = tf.reshape(new_inputs, [-1, self.num_feature*self.embedding_size])
        # 2) Deep Component
        y_deep = self.layers1(new_inputs)
        y_deep = self.dropout1(y_deep)
        y_deep = self.layers2(y_deep)
        y_deep = self.dropout2(y_deep)
        y_deep = self.layers3(y_deep)


        # Concatenation
        y_pred = tf.concat([y_fm, y_deep], 1)
        y_pred = self.final(y_pred)
        y_pred = tf.reshape(y_pred, [-1, ])

        return y_pred

