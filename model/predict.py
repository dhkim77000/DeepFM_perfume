import config
import pandas as pd
import numpy as np
import seaborn as sns
from ast import literal_eval
import math
from collections import Counter
import pdb
import os
import re
import multiprocessing as mp
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import argparse
from itertools import repeat
from collections import defaultdict
import pycountry_convert as pc
import numpy as np
import pandas as pd
import warnings
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler,LabelEncoder
import config
import warnings
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
import deepfm
from train import seed_everything
import gc
from numba import cuda 
import torch
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession


def create_user_df(user_row, length):
    user_df = user_row
    result = pd.DataFrame()

    for column in tqdm(user_df.columns):
        if column != 'user_id':
            result[column] = list(user_row[column]) * length
    #for i in tqdm(range(length)):
    #    user_df = user_df.append(user_row)
    print('-------Done-------')
    #user_df.drop('user_id', axis = 1, inplace = True)
    result.reset_index(drop = False, inplace = True)
    return result

def binary(x):
    if x > 0.95: x= 1
    else: x= 0

def sort_pre(predict, mode):
    if mode == 'classification':
        df = pd.DataFrame({
        'index': range(len(predict)),
        'result': predict}, 
        columns=['index', 'result'])

        df = df[df['result'] > 0.95]
        df['rank'] = df['result'].rank(method = 'min', ascending = False)
        df = df.sort_values(by=['rank'])
        return df
    elif mode == 'regression':
        df = pd.DataFrame({
        'index': range(len(predict)),
        'result': predict}, 
        columns=['index', 'result'])

        df = df[df['result'] > 8.5]
        df['rank'] = df['result'].rank(method = 'min', ascending = False)
        df = df.sort_values(by=['rank'])
        return df


if __name__ =='__main__':
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = InteractiveSession(config = tf_config)

    mode = sys.argv[-2]
    if sys.argv[-1] == 'full': detail = True
    elif sys.argv[-1] == 'short': detail = False
    
 
    num_core = os.cpu_count()
    all_filed = config.ALL_FIELDS
    cat_field = config.CAT_FIELDS
    cont_fields = config.CONT_FIELDS
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    print('-------Reading user data...-------')
    user_info = pd.read_csv(config.user_info_path ,encoding ='utf-8-sig')
    user_info = user_info[user_info['update'] == 1]
    user_index = list(user_info.index)
    user_dummy = pd.read_csv(config.user_dummy_path, encoding ='utf-8-sig').iloc[user_index, :]
    print('-------Done-------')


    print('-------Reading item data...-------')
    item_dummy = pd.read_csv(config.item_dummy_path, encoding ='utf-8-sig')
    fd_user = defaultdict(list)
    fd_item = defaultdict(list)
    user_code = {}
    with open(config.field_dict_user_path,'rb') as f1:
        fd_user = pickle.load(f1)
    with open(config.field_dict_item_path,'rb') as f2:
        fd_item = pickle.load(f2)
    with open(config.user_code_path,'rb') as f3:
        user_code = pickle.load(f3)

    fd_user.update(fd_item)
    field_dict = fd_user
    field_index = []
    for i,column in enumerate(all_filed):
        if column in set(cat_field):
            field_index.extend(list(repeat(i, len(field_dict[column]))))
        else: field_index.append(i)
    print('-------Done-------')


    seed_everything(config.seed)
    print('-------Setting model...-------')
    if mode == 'classification':
        checkpoint_filepath = config.classification_checkpoint_filepath
    elif mode == 'regression':
        checkpoint_filepath = config.regression_checkpoint_filepath
    model = deepfm.DeepFM(embedding_size=config.embedding_size, num_feature=len(field_index),
                    num_field=len(field_dict), field_index=field_index, mode = mode)
    print('-------Done-------')
    model.load_weights(checkpoint_filepath)

    size_indicator = np.array_split(np.zeros(len(item_dummy)), num_core)

    for i in user_index:
        user = user_dummy.loc[i,'user_id']
        user_code = user_info.loc[i,'code']

        print(f'-------Preparing data for {user}...-------')
        user_row =  user_dummy.iloc[i:i+1,:]
        with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
            results = parallel(delayed(create_user_df)(user_row, len(size_indicator[i])) for i in range(num_core))
        for i,result in enumerate(results):
            if i == 0:
                user_df = result
            else:
                user_df  = pd.concat([user_df , result], axis = 0)
        user_df.reset_index(inplace = True)
        user_df.drop(['level_0', 'index'], axis = 1, inplace = True)

        print('-------Concating data...-------')
        input = pd.concat([user_df, item_dummy] , axis = 1).dropna()
        del user_df
        del result
        print('-------Done-------')

        print('-------Predicting...-------')
        pdb.set_trace()
        predict = model.predict(input)
        predict = sort_pre(predict, mode)
        del input

        if detail:
            predict.to_csv('/home/dhkim/Fragrance/predict/{i}_predict.csv', index = False)
        else:
            with open('/home/dhkim/Fragrance/predict/{i}_predict.pkl','wb') as f:
                pickle.dump(predict['index'].values, f)
        del predict
        torch.cuda.empty_cache() 
        gc.collect()


    
