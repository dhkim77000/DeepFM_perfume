from deepfm import DeepFM
import numpy as np
import pandas as pd
from time import perf_counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
import pandas as pd
import config
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow import keras
import deepfm
import sys
import torch
import config
import pdb
import random
from numba import cuda 
import os
import gc
from train import train_model
from train import seed_everything


def clear_gpu():
    device = cuda.get_current_device()
    device.reset()

if __name__ == '__main__':
    mode = sys.argv[-1]
    print(mode)
    item_table_path = config.item_table_path
    rating_table_path = config.rating_table_path
    notes_info_path = config.notes_info_path
    X_path = config.X_path
    Y_path = config.Y_path
    field_dict_path = config.field_dict_path
    field_index_path = config.field_index_path
    if mode == 'classification':
        checkpoint_filepath = config.classification_checkpoint_filepath
    elif mode == 'regression':
        checkpoint_filepath = config.regression_checkpoint_filepath
    types_list = config.types_list
    ALL_FIELDS = config.ALL_FIELDS
    types_dict = config.types_dict
    test_size = config.test_size 
    epochs = config.epochs
    scheduler_type = config.scheduler_type 
    embedding_size = config.embedding_size
    lr = config.lr
    batch_size = config.batch_size

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*8)])
        except RuntimeError as e:
            print(e)

    print('-------Reading file...-------')
    X = pd.read_csv(X_path, encoding ='utf-8-sig')
    Y = pd.read_csv(Y_path, encoding ='utf-8-sig')
    with open(field_dict_path, 'rb') as f:
        field_dict = pickle.load(f)
    with open(field_index_path, 'rb') as f:
        field_index = pickle.load(f)
    print('-------Done-------')

    best = 0
    best_seed = -1

    logs = pd.DataFrame(columns =['seed', 'result'])
    failed_seed = set()
    #with open(f'{config.seed_dir}/{mode}_fail.pickle', 'rb') as f:
    #    failed_seed = pickle.load(f)
    for seed in range(100):
        if seed not in failed_seed:
            print(f'best seed {best_seed }: {best}')

            result = train_model(tensor_board = False, X = X, Y = Y, seed = seed, 
                                test_size = test_size, embedding_size = embedding_size, lr = lr, 
                                field_dict= field_dict, field_index = field_index, 
                                scheduler = scheduler_type, checkpoint_filepath = checkpoint_filepath, 
                                epochs = 10, batch_size = batch_size, mode = mode, save = True)
            if result[0] < 1:
                result = train_model(tensor_board = True, X = X, Y = Y, seed = seed, 
                                test_size = test_size, embedding_size = embedding_size, lr = lr, 
                                field_dict= field_dict, field_index = field_index, 
                                scheduler = scheduler_type, checkpoint_filepath = checkpoint_filepath, 
                                epochs = 100, batch_size = batch_size, mode = mode, save = True)[-1]
                if (result > best) & (result > 0.7): 
                    best_seed = seed
                    best = result
                    log = {}
                    log['seed'] = best_seed
                    log['result'] = result
                    logs.append(log, ignore_index = True)
                    logs.to_csv('/home/dhkim/Fragrance/model/log/log.csv', index = False)
                    print(seed)
            else:
                failed_seed.add(seed)
                with open(f'{config.seed_dir}/{mode}_fail.pickle', 'wb') as f:
                    pickle.dump(failed_seed, f)
                    #clear_gpu()
            torch.cuda.empty_cache()
            print(f'seed {seed }: {result}')
    print(best_seed)
