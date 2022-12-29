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
import config
import random
import os


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)





 
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
    embedding_size = config.embedding_size
    lr = config.lr
    batch_size = config.batch_size

    seed_everything(config.seed)
    print('-------Reading file...-------')
    X = pd.read_csv(X_path, encoding ='utf-8-sig')
    Y = pd.read_csv(Y_path, encoding ='utf-8-sig')
    if mode == 'classification':
        Y = pd.Series([1 if (int(x)>= 8) else 0 for x in Y.iloc[:,0]])
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y)

    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    with open(field_dict_path, 'rb') as f:
        field_dict = pickle.load(f)
    with open(field_index_path, 'rb') as f:
        field_index = pickle.load(f)
    print('-------Done-------')

    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            for i in gpu:
                tf.config.experimental.set_memory_growth(i, True)
        except RuntimeError as e:
            print(e)

    print('-------Setting model...-------')
    model = deepfm.DeepFM(embedding_size=embedding_size, num_feature=len(field_index),
                    num_field=len(field_dict), field_index=field_index, mode = mode)
    print('-------Done-------')

    print('-------Setting environment...-------')
    checkpoint_dir = os.path.dirname(checkpoint_filepath)

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    if mode == 'classification':
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_auc',
            mode='max',
            save_best_only=True)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    elif mode == 'regression':
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        model.compile(loss='mse', optimizer=optimizer, metrics=tf.keras.metrics.RootMeanSquaredError())
    print('-------Done-------')

    model.fit(X_train, Y_train, 
                epochs=epochs, 
                verbose=1, 
                validation_split=0.2, 
                shuffle=True, 
                batch_size = batch_size, 
                callbacks = [model_checkpoint_callback])

    model.load_weights(checkpoint_filepath)
    model.evaluate(X_test, Y_test)

