from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt
from nltk.stem import WordNetLemmatizer
import nltk
import re
import gc
import pandas as pd
from tqdm import tqdm
import numpy as np
import pdb
import ast
from scipy.stats import rankdata
import config
import sys
import os
import multiprocessing as mp
from joblib import Parallel, delayed
import sys
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler,LabelEncoder
import config
import warnings
import pickle
import multiprocessing as mp



def clean(text):
    text = str(text)
    if text != 'nan':
        text = re.sub('[^a-zA-Z]',' ',text).lower()
        text = word_tokenize(text)
        return text
    else: return None

def lemmatizer(text):
    word_lemmatizer = WordNetLemmatizer()
    lemm_text = [word_lemmatizer.lemmatize(word) for word in text]
    return lemm_text

def add_text_column():
    item_table = pd.read_csv(config.item_table_path, encoding ='utf-8-sig')
    text = pd.read_csv('data/text.csv')
    text_data = pd.DataFrame(columns = ['link','text'])
    text_dic = {}
    datas = []
    visited = set()
    text_data = pd.DataFrame(columns = ['link','text'])
    text_dic = {}
    datas = []
    visited = set()
    for i in tqdm(range(len(text))):
        link = text.loc[i, 'link']
        t = text.loc[i,'text']
        if link in visited:
            text_dic[link] += t
        else:
            text_dic[link] = t
            visited.add(link)
    text_data['link'] = text_dic.keys()
    text_data['text'] = text_dic.values()
    item_table = pd.merge(item_table, text_data, how = 'left',left_on = 'url',right_on = 'link')
    item_table.drop('link',axis = 1, inplace = True)
    item_table = item_table.rename(columns={'text_x':'text','text_y':'review_text'})
    item_table.to_csv(config.item_dummy_path, encoding ='utf-8-sig', index= False)
    return item_table

def text_preprocessing(data):
    tqdm.pandas()
    data['review_text'] = data['review_text'].progress_apply(lambda x: clean(x))
    data = data[data['review_text'].notna() == True]
    data['review_text'] = data['review_text'].progress_apply(lambda x: lemmatizer(x))
    return data


def tokenization(df):
    tqdm.pandas()
    try: df['review_text'] = df['review_text'].apply(lambda x: nltk.pos_tag(ast.literal_eval(x)))
    except: df['review_text'] = df['review_text'].apply(lambda x: nltk.pos_tag(x))
    df['noun'] = df['review_text'].progress_apply(lambda x: [w[0] for w in x if (w[1]=='NN') or (w[1]=='NNPS') or (w[1]=='NNP')]) 
    df['adj'] = df['review_text'].progress_apply(lambda x: [w[0] for w in x if  (w[1]=='JJ') or (w[1]=='JJR') or (w[1]=='JJS')])

    return df

def parallel_tokenization(data):
    num_core = os.cpu_count()
    data_chunks = np.array_split(data, num_core)
    print(f"Parallelizing with {num_core} cpus")
    with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
        results = parallel(delayed(tokenization)(data_chunks[i]) for i in range(num_core))
    for i,df in enumerate(results):
        if i == 0:
            result = df
        else:
            result = pd.concat([result, df], axis = 0)
    return result

if __name__ =='__main__':
    mode = sys.argv[-1]
    
    if mode == 'full':
        item_table = add_text_column()
        data = item_table.loc[:,['name','review_text']]
        data = text_preprocessing(data)
        data = parallel_tokenization(data)
        data.to_csv(config.token_path, index = False)

    elif mode == 'pre':
        item_table = pd.read_csv(config.item_table_path)
        data = item_table.loc[:,['name','review_text']]
        data = text_preprocessing(data)
        data.to_csv(config.fragrance_text_path , index = False)
    elif mode == 'tok':
        data = pd.read_csv(config.fragrance_text_path)
        data = parallel_tokenization(data)
        data.to_csv(config.token_path, index = False)
    elif mode == 'train':
        data = pd.read_csv(config.text_path)
        data= data.rename(columns={'text':'review_text'})
        data = text_preprocessing(data)
        data = parallel_tokenization(data)
        data.to_csv(config.text_train_path, index = False)


        
        