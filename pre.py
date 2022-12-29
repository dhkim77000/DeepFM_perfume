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

def get_continent(country):
    try:
        country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        return  pc.country_alpha2_to_continent_code(country_code)
    except:
        return 'ETC'

def user_id_dummies(user_info):
  df = []
  for user in user_info.index:
    dummy = [0] * len(user_info)
    dummy[user_info.loc[user,'code']] = 1
    df.append(dummy)
  df = np.array(df)
  return pd.DataFrame(df).add_prefix('u_')


def count_dummies(user_info,len):
    df = []
    user_info['count'].fillna(0)

def count_dummies(user_info,len, threshold = config.threshold):
    df = []
    user_info['count'].fillna(0)
    count_bin = pd.cut(user_info['count'], len, labels=False)
    for user in user_info.index:
        dummy = [0] * len
        index = user_info.loc[user,'code']
        dummy[count_bin[index]] = 1
        df.append(dummy)
    df = np.array(df)

    return pd.DataFrame(df).add_prefix('c_')


def gender_dummies(user_info):
  user_gender_df = user_info['gender'].str.get_dummies()
  user_gender_df.set_index(user_info['code'], inplace = True)
  return user_gender_df

def nation_dummies(user_info):
  user_info['continent'] = user_info['nation'].apply(lambda x : get_continent(x))
  nation_df = user_info['continent'].str.get_dummies()
  nation_df.set_index(user_info['code'], inplace = True)

  return nation_df


def get_user_dummmies(user_info, field_dict):
    index = {}

    count_df = count_dummies(user_info, 4)
    index['count'] = (0,len(count_df.columns))
    last = len(count_df.columns)
    field_dict['count'] = list(count_df.columns)

    gender_df = gender_dummies(user_info)
    index['gender'] = (last, last+len(gender_df.columns))
    last += len(gender_df.columns)
    field_dict['user_gender'] = list(gender_df.columns)

    nation_df = nation_dummies(user_info)
    index['nation'] = (last, last + len(nation_df.columns))
    last += len(nation_df.columns)
    field_dict['nation'] = list(nation_df.columns)

    user_id_df = user_id_dummies(user_info)
    index['user_id'] = (last, last + len(user_id_df.columns))
    field_dict['user_id'] = list(user_id_df.columns)

    result = pd.concat([count_df, gender_df, nation_df, user_id_df], axis = 1)
    result['user_id'] = user_info.index
    print(f'before drop: {len(result)}')
    result.drop(user_info[user_info['count'] < config.threshold].code, inplace=True)
    result.reset_index(inplace = True)
    print(f'after drop: {len(result)}')
    result.drop('index',axis = 1, inplace = True)
    return result, index, field_dict



def user_dummy(user_info, rating_table, field_dict):
  user_df, _, field_dict = get_user_dummmies(user_info, field_dict)
  rating_table.drop(['gender','nation','brand','fragrance','url'], axis = 1, inplace = True)

  result = pd.merge(user_df, rating_table, on = 'user_id',how = 'inner')
  result.drop('user_id', axis = 1, inplace = True)
  return result, field_dict

class Bayesian_adj:
    def __init__(self, df, column, count_column):
        self.df = df
        self.column = self.df[column]
        self.count = self.df[count_column]
        self.N = self.count.sum(skipna=True)
        self.NR = (self.column * self.count).sum()

    def bayesian_rating(self, n, r):
        return (self.NR  + n * r) / (self.N + n)

def convert_data(datas):
    result = []
    for i in datas:
        try:
            result.extend(literal_eval(i))
        except Exception:
            continue
    return result

def get_unique_values_list_data(column):
    data_list = [x for x in column.values]
    return list(set(sum(data_list, [])))

def get_unique_values_string_data(column):
    data_list = [x for x in column.values]
    return list(set(data_list))

def key_value(unique_value_list):
    result = {}
    for i in range(len(unique_value_list)):
        result[unique_value_list[i]] = i
    return result   

class Encoder:
    def __init__(self, df, columns_list, name, sig):
        self.columns = columns_list
        self.sig = sig
        self.df = df[self.columns]
        
        self.count = df[name+'_count']
        self.avg_count = df[name+'_count'].mean(skipna = True)

        self.count =  df[name+'_count'].fillna(0)
        for column in columns_list:
            self.df[column] = self.df[column].fillna(0)
                    
    def custom_sigmoid(self, x):
        return 1 / (1 +np.exp(-x/self.avg_count))

    def encoder(self):
        result = []
        for i in tqdm(self.df.index):
            input = self.df.loc[i,:]
            row = np.zeros(len(input))
            count = self.count[i]
            if count == 0: result.append(row)

            else:
                for i in range(len(input)):
                    row[i] = input[i]
                zero = [i for i, e in enumerate(row) if e == 0]

                if self.sig == True: 
                    row = row * self.custom_sigmoid(count)
                    left = 100 - np.nansum(row)
                    if left != 0:
                        left = left / len(zero)
                        for i in zero:
                            row[i] = left
                        result.append(row)
                    else: result.append(row)
                elif self.sig == False: result.append(row)

        result =  pd.DataFrame(result, columns = self.columns[:])
        result.index = self.df.index
        return result

def get_common(item_table,rating_table):

 
    #rating_table.drop('brand', axis = 1, inplace = True)
    #fragrance_name=np.array(list(item_table['name']))

    #Encoding user id
    #encoder = LabelEncoder()
    #encoder.fit(rating_table['user_id'])
    #rating_table['user_id'] = encoder.transform(rating_table['user_id'])

    #merge
    #rating_table.drop('name', axis = 1, inplace = True)
    result = pd.merge(rating_table, item_table, on = 'img_url', how = 'inner')
    result.drop(['img_url','url','Unnamed: 39','name','continent'], axis = 1, inplace = True)
    #result['name'] = encoder.fit_transform(result['name'])
    #result.rename(columns={'name':'fragrance'}, inplace = True)

    #result['fragrance'] = result['fragrance'].astype(str)
    #result['user_id'] = result['user_id'].astype(str)
    print('total ' +str(len(result))+'data')
    return result

def get_continent(country):
    try:
        country_code = pc.country_name_to_country_alpha2(country, cn_name_format="default")
        return  pc.country_alpha2_to_continent_code(country_code)
    except:
        return 'ETC'


def find_parent_notes(notes_info, note):
    return notes_info[notes_info['child note']== note].index.values

def get_parent(notes_info, note_list):
    note_list = literal_eval(note_list)
    if len(note_list) != None:
        result = []
        for note in note_list:
            parents = find_parent_notes(notes_info, note)
            for parent in parents:
                parent = parent[0].replace(" ", "")
                result.append(parent)
    return (' ').join(result)
def to_liter(list_data):
    list_data = literal_eval(list_data)
    for i in range(len(list_data)):
        list_data[i] = list_data[i].replace(" ", '')
    return (' ').join(list_data)

def parallel_notes(item_table, notes_info):
    item_table['top_notes'] = item_table['top_notes'].fillna('[]')
    item_table['base_notes'] = item_table['base_notes'].fillna('[]')
    item_table['heart_notes'] = item_table['heart_notes'].fillna('[]')
    item_table['top_literal'] = item_table['top_notes'].apply(lambda x: get_parent(notes_info, x))
    item_table['base_literal'] = item_table['base_notes'].apply(lambda x: get_parent(notes_info, x))
    item_table['heart_literal'] = item_table['heart_notes'].apply(lambda x: get_parent(notes_info, x))
    #item_table['notes_literal'] =  item_table['top_literal'] + " "+ item_table['base_literal'] + " "+ item_table['heart_literal']

    print('--------------------done--------------------')
    return top_notes_df, base_notes_df, heart_notes_df


def train_preprocessing(rating_table, item_table, user_info, notes_info, field_dict, mode, user = True, Binary = False):
    
    field_index = []
    all_filed = config.ALL_FIELDS
    cat_field = config.CAT_FIELDS
    cont_fields = config.CONT_FIELDS

    if mode == 'train':
        rating_table, field_dict = user_dummy(user_info, rating_table, field_dict)

    field_dict['rating'] = ['rating']
    field_dict['Scent'] = ['Scent']
    field_dict['Sillage'] = ['Sillage']
    field_dict['Longevity'] = ['Longevity']
    field_dict['Value for money'] = ['Value for money']

    seasons = ['Spring','Summer','Fall','Winter']
    occasions = ['Leisure','Daily','Night Out','Business','Sport','Evening']
    audiences = ['Old','Young','Men','Women']
    types_list = config.types_list

    for columns in [seasons, audiences, occasions, types_list]:
        for column in columns:
            field_dict[column] = [column]


    #rating_table.drop('user_id', axis = 1, inplace = True)


    print('--------------------item categorical field--------------------')
    #Gender_df
    print('--------------------'+ 'processing gender'+'--------------------')
    gender_df = item_table.gender.str.get_dummies()
    item_table.drop('gender', axis = 1, inplace = True)
    item_table = pd.concat([item_table, gender_df], axis = 1)
    field_dict['gender'] = list(gender_df.columns)
    print(len(item_table))

    #year_df
    print('--------------------'+'processing year'+ '--------------------')
    item_table.year = item_table.year.astype(np.float16)
    bins = list(range(1900, 2030, 10))
    bins.append(1370)
    bins.append(0)
    bins = sorted(bins)
    labels = list(range(len(bins) - 1))
    labels = ["year_" + str(i) for i in labels]
    item_table.year = pd.cut(item_table['year'], bins=bins, right=True, labels=labels)
    year_df = pd.get_dummies(item_table.year)
    item_table = pd.concat([item_table, year_df], axis=1)
    item_table.drop("year", axis=1, inplace=True)
    field_dict['year'] = list(year_df.columns)
    print(len(item_table))

    #brands_df
    print('--------------------'+ 'processing brands'+ '--------------------')
    br_encoder = LabelEncoder()
    item_table['brand'] = br_encoder.fit_transform(item_table.brand.values)
    item_table['brand'] = item_table['brand'].astype(str)
    brands_df = item_table.brand.str.get_dummies()
    item_table.drop('brand', axis = 1, inplace = True)
    item_table = pd.concat([item_table, brands_df], axis = 1)
    field_dict['brand'] = list(brands_df.columns)
    print(len(item_table))

    #perfumer_df
    print('--------------------'+ 'processing perfumer'+ '--------------------')
    item_table['perfumer'] = item_table['perfumer'].fillna('[]')
    item_table['perfumer_literal'] = item_table['perfumer'].apply(to_liter)
    perfumer_df = item_table['perfumer_literal'].str.get_dummies(" ")
    item_table = pd.concat([item_table, perfumer_df], axis=1)
    item_table.drop(['perfumer','perfumer_literal'], axis=1, inplace=True)
    field_dict['perfumer'] = list(perfumer_df.columns)
    print(len(item_table))

    #notes 
    print('--------------------'+ 'processing notes'+ '--------------------')
    item_table['top_notes'] = item_table['top_notes'].fillna('[]')
    item_table['base_notes'] = item_table['base_notes'].fillna('[]')
    item_table['heart_notes'] = item_table['heart_notes'].fillna('[]')
    item_table['top_literal'] = item_table['top_notes'].apply(lambda x: get_parent(notes_info, x))
    item_table['base_literal'] = item_table['base_notes'].apply(lambda x: get_parent(notes_info, x))
    item_table['heart_literal'] = item_table['heart_notes'].apply(lambda x: get_parent(notes_info, x))
    #item_table['notes_literal'] =  item_table['top_literal'] + " "+ item_table['base_literal'] + " "+ item_table['heart_literal']
    #notes_df = item_table['notes_literal'].str.get_dummies(" ")
    top_notes_df = item_table['top_literal'].str.get_dummies(" ").add_prefix('t_')
    base_notes_df = item_table['base_literal'].str.get_dummies(" ").add_prefix('b_')
    heart_notes_df = item_table['heart_literal'].str.get_dummies(" ").add_prefix('h_')

   
    item_table = pd.concat([item_table, top_notes_df], axis=1)
    item_table = pd.concat([item_table, base_notes_df], axis=1)
    item_table = pd.concat([item_table, heart_notes_df], axis=1)
    item_table.drop(['top_notes','base_notes','heart_notes','top_literal','base_literal','heart_literal'], axis=1, inplace=True)

    field_dict['top_notes'] = list(top_notes_df.columns)
    field_dict['base_notes'] = list(base_notes_df.columns)
    field_dict['heart_notes'] = list(heart_notes_df.columns)


    if mode == 'train':
        input = get_common(rating_table, item_table)
        if Binary == True:  
            Y = pd.Series([1 if (int(x)>= int(config.rating_cut)) else 0 for x in input['user_rating']])
        else:
            Y = input['user_rating']

        input.drop(['user_rating'], axis = 1, inplace = True)

        field_index = []
        for i,column in enumerate(all_filed):
            if column in set(cat_field):
                #if i == 42: pdb.set_trace()
                field_index.extend(list(repeat(i, len(field_dict[column]))))
            else: field_index.append(i)

        X = input

        return X, Y, field_dict, field_index

    elif mode == 'predict':
        item_table.drop(['img_url','Unnamed: 39','name'], axis = 1, inplace = True)
        return item_table, field_dict
        


    

def Bayesian_rating(item_table):
    rating = Bayesian_adj(item_table, 'rating','rating_count')
    item_table['rating'] = rating.bayesian_rating(item_table.rating_count, item_table.rating)
    item_table.drop("rating_count", axis=1, inplace=True)
    item_table['rating'] = item_table['rating'].fillna(item_table['rating'].mean(skipna = True))

    scent = Bayesian_adj(item_table, 'Scent','Scent_count')
    item_table['Scent'] = scent.bayesian_rating(item_table.Scent_count, item_table.Scent)
    item_table.drop("Scent_count", axis=1, inplace=True)
    item_table['Scent'] = item_table['Scent'].fillna(item_table['Scent'].mean(skipna = True))

    longevity = Bayesian_adj(item_table, 'Longevity','Longevity_count')
    item_table['Longevity'] = longevity.bayesian_rating(item_table.Longevity_count, item_table.Longevity)
    item_table.drop("Longevity_count", axis=1, inplace=True)
    item_table['Longevity'] = item_table['Longevity'].fillna(item_table['Longevity'].mean(skipna = True))

    sillage = Bayesian_adj(item_table, 'Sillage','Sillage_count')
    item_table['Sillage'] = sillage.bayesian_rating(item_table.Sillage_count, item_table.Sillage)
    item_table.drop("Sillage_count", axis=1, inplace=True)
    item_table['Sillage'] = item_table['Sillage'].fillna(item_table['Sillage'].mean(skipna = True))

    price_value = Bayesian_adj(item_table, 'Value for money','Value for money_count')
    item_table['Value for money'] = price_value.bayesian_rating(item_table['Value for money_count'], item_table['Value for money'])
    item_table.drop("Value for money_count", axis=1, inplace=True)
    item_table['Value for money'] = item_table['Value for money'].fillna(item_table['Value for money'].mean(skipna = True))
    
    return item_table

def parallelize_preprocessing(item_table, rating_table, notes_info, mode):
    #nation_df

    print('--------------------'+ 'processing nation'+ '--------------------')
    rating_table['continent'] = rating_table['nation'].apply(lambda x : get_continent(x))
    

    print('--------------------'+ 'processing type'+ '--------------------')
    
    types = item_table.loc[:,'Type']
    types_list = list(set(convert_data(types)))
    types_dict = key_value(types_list)

    types_df = pd.DataFrame(columns = types_list)
    item_table['Type'] = item_table['Type'].fillna('{}')
    for i in tqdm(item_table.index):
        types_dict = literal_eval(item_table.loc[i, 'Type'])
        types_df = types_df.append(types_dict, ignore_index = True)
    
    types_df.index = item_table.index
    item_table = pd.concat([item_table, types_df], axis=1)
    item_table.drop("Type", axis=1, inplace=True)


    seasons = ['Spring','Summer','Fall','Winter']
    occasions = ['Leisure','Daily','Night Out','Business','Sport','Evening']
    audiences = ['Old','Young','Men','Women']

    if (mode == 'sim'):
        type_enc = Encoder(item_table, types_list, 'Type', False)
        season_enc = Encoder(item_table, seasons, 'Season', False)
        occasion_enc = Encoder(item_table, occasions , 'Occasion', False)
        audience_enc = Encoder(item_table, audiences, 'Audience',False)
    else: 
        type_enc = Encoder(item_table, types_list, 'Type', True)
        season_enc = Encoder(item_table, seasons, 'Season', True)
        occasion_enc = Encoder(item_table, occasions , 'Occasion', True)
        audience_enc = Encoder(item_table, audiences, 'Audience', True)

    #Type
    item_table[types_list] = type_df = type_enc.encoder()
    #Season
    print('--------------------'+ 'processing season'+ '--------------------')
    item_table[seasons] = season_enc.encoder()
    #Occasion
    print('--------------------'+ 'processing occasion'+ '--------------------')
    item_table[occasions] = occasion_enc.encoder()
    #Audience
    print('--------------------'+'processing audience'+ '--------------------')
    item_table[audiences] = audience_enc.encoder()
    item_table.drop(['Type_count','Audience_count','Season_count','Occasion_count'], axis = 1,inplace = True)

    if mode != 'sim': item_table.drop(['text'], axis = 1,inplace = True)
    elif mode == 'sim' :item_table.drop('img_url', axis = 1,inplace = True)

    if mode == 'db':
        #perfumer_df
        print('--------------------'+ 'processing perfumer'+ '--------------------')
        item_table['perfumer'] = item_table['perfumer'].fillna('[]')
        item_table['perfumer'] = item_table['perfumer'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',','|'))
        #notes 
        item_table['top_notes'] = item_table['top_notes'].fillna('[]')
        item_table['base_notes'] = item_table['base_notes'].fillna('[]')
        item_table['heart_notes'] = item_table['heart_notes'].fillna('[]')
        print('--------------------'+ 'processing top notes'+ '--------------------')
        item_table['top_literal'] = item_table['top_notes'].apply(lambda x: get_parent(notes_info, x))
        print('--------------------'+ 'processing base notes'+ '--------------------')
        item_table['base_literal'] = item_table['base_notes'].apply(lambda x: get_parent(notes_info, x))
        print('--------------------'+ 'processing heart notes'+ '--------------------')
        item_table['heart_literal'] = item_table['heart_notes'].apply(lambda x: get_parent(notes_info, x))

        t_notes_df = item_table['top_literal'].str.get_dummies(" ")
        b_notes_df = item_table['base_literal'].str.get_dummies(" ")
        h_notes_df = item_table['heart_literal'].str.get_dummies(" ")

        top_notes = pd.concat([item_table.iloc[:,:3], t_notes_df], axis=1)
        base_notes = pd.concat([item_table.iloc[:,:3], b_notes_df], axis=1)
        heart_notes = pd.concat([item_table.iloc[:,:3], h_notes_df], axis=1)

        item_table.drop(['top_notes','base_notes','heart_notes','top_literal','base_literal','heart_literal'], axis = 1, inplace = True)
        print('--------------------'+ 'done'+ '--------------------')
        return [item_table, rating_table , top_notes, base_notes, heart_notes]

    elif mode == 'train':
        return [item_table, rating_table]

    elif mode == 'sim':
        item_table['perfumer'] = item_table['perfumer'].apply(lambda x: str(x).replace('[','').replace(']','').replace(',','|'))
        item_table['top_notes'] = item_table['top_notes'].fillna('[]')
        item_table['base_notes'] = item_table['base_notes'].fillna('[]')
        item_table['heart_notes'] = item_table['heart_notes'].fillna('[]')
        print('--------------------'+ 'processing top notes'+ '--------------------')
        item_table['top_notes'] = item_table['top_notes'].apply(lambda x: get_parent(notes_info, x))
        print('--------------------'+ 'processing base notes'+ '--------------------')
        item_table['base_notes'] = item_table['base_notes'].apply(lambda x: get_parent(notes_info, x))
        print('--------------------'+ 'processing heart notes'+ '--------------------')
        item_table['heart_notes'] = item_table['heart_notes'].apply(lambda x: get_parent(notes_info, x))
        return [item_table, rating_table]

    

    

if __name__ == '__main__':

    mode = sys.argv[-1]
    print(mode)
    num_core = os.cpu_count()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    #reading fragrance DB
    item_table = pd.read_csv(config.item_table_path, encoding ='utf-8-sig')
    item_table = item_table.drop_duplicates()
    item_table.loc[item_table['year'] == 'Unknown', 'year'] = 0

    #reading rating DB
    rating_table = pd.read_csv(config.rating_table_path, encoding ='utf-8-sig')

    #reading Notes DB
    notes_info =  pd.read_csv(config.notes_info_path,index_col=[0,1])

    #reading user DB
    user_info = pd.read_csv(config.user_info_path, index_col=0 ,encoding ='utf-8-sig')

    if mode == 'DB':

        item_table_chunks = np.array_split(item_table, num_core)
        rating_table_chunks = np.array_split(rating_table, num_core)

        print('Parallelizing with ' +str(num_core)+'cores')
        with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
            results = parallel(delayed(parallelize_preprocessing)(item_table_chunks[i],rating_table_chunks[i], notes_info,'db') for i in range(num_core))

        for i,data in enumerate(results):
            if i == 0:
                item_table = data[0]
                rating_table = data[1]
                top_notes = data[2]
                base_notes = data[3]
                heart_notes = data[4]
            else:
                item_table = pd.concat([item_table, data[0]], axis = 0)
                rating_table = pd.concat([rating_table, data[1]], axis = 0)
                top_notes = pd.concat([top_notes, data[2]], axis = 0)
                base_notes = pd.concat([base_notes, data[3]], axis = 0)
                heart_notes = pd.concat([heart_notes , data[4]], axis = 0)


        item_table.to_csv('/home/dhkim/Fragrance/data/item_table.csv' ,encoding ='utf-8-sig',  index=False)
        rating_table.to_csv('/home/dhkim/Fragrance/data/rating_table.csv' ,encoding ='utf-8-sig',  index=False)
        top_notes.to_csv('/home/dhkim/Fragrance/data/top_notes.csv' ,encoding ='utf-8-sig',  index=False)
        base_notes.to_csv('/home/dhkim/Fragrance/data/base_notes.csv' ,encoding ='utf-8-sig',  index=False)
        heart_notes.to_csv('/home/dhkim/Fragrance/data/heart_notes.csv' ,encoding ='utf-8-sig',  index=False)

    elif mode == 'train':

        item_table = Bayesian_rating(item_table)
        item_table_chunks = np.array_split(item_table, num_core)
        rating_table_chunks = np.array_split(rating_table, num_core)

        print('Parallelizing with ' +str(num_core)+'cores')
        with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
            results = parallel(delayed(parallelize_preprocessing)(item_table_chunks[i],rating_table_chunks[i], notes_info, 'train') for i in range(num_core))

        for i,data in enumerate(results):
            
            if i == 0:
                item_table = data[0]
                rating_table = data[1]
            else:
                item_table = pd.concat([item_table, data[0]], axis = 0)
                rating_table = pd.concat([rating_table, data[1]], axis = 0)

        field_dict = defaultdict(list)
        X, Y, field_dict, field_index = train_preprocessing(rating_table, item_table, user_info, notes_info, field_dict, 'train')
        X = X.fillna(0)

        for col in list(X.columns):
            if 'Unnamed' in col:
                X.drop(col,axis = 1, inplace = True)

        X.to_csv('/home/dhkim/Fragrance/data/X.csv' ,encoding ='utf-8-sig',  index=False)
        Y.to_csv('/home/dhkim/Fragrance/data/Y.csv' ,encoding ='utf-8-sig',  index=False)

        with open('/home/dhkim/Fragrance/data/field_dict.pkl','wb') as f:
                pickle.dump(field_dict,f)
        with open('/home/dhkim/Fragrance/data/field_index.pkl','wb') as f:
                pickle.dump(field_index ,f)

    elif mode == 'predict':
        
        item_table = Bayesian_rating(item_table)
        item_table_chunks = np.array_split(item_table, num_core)
        rating_table_chunks = np.array_split(rating_table, num_core)

        print('Parallelizing with ' +str(num_core)+'cores')
        with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
            results = parallel(delayed(parallelize_preprocessing)(item_table_chunks[i],rating_table_chunks[i], notes_info, 'train') for i in range(num_core))

        for i,data in enumerate(results):
            if i == 0:
                item_table = data[0]
                rating_table = data[1]
            else:
                item_table = pd.concat([item_table, data[0]], axis = 0)
                rating_table = pd.concat([rating_table, data[1]], axis = 0)

        field_dict = defaultdict(list)
        item_table, field_dict = train_preprocessing(rating_table, item_table, user_info, notes_info, field_dict, 'predict')
        item_table = item_table.fillna(0)

        for col in list(item_table.columns):
            if 'Unnamed' in col:
                item_table.drop(col,axis = 1, inplace = True)

        item_table.to_csv('/home/dhkim/Fragrance/data/item_dummy.csv' ,encoding ='utf-8-sig',  index=False)
        with open('/home/dhkim/Fragrance/data/field_dict_item.pkl','wb') as f:
                pickle.dump(field_dict ,f)


    elif mode == 'sim':

        item_table_chunks = np.array_split(item_table, num_core)
        rating_table_chunks = np.array_split(rating_table, num_core)

        print('Parallelizing with ' +str(num_core)+'cores')
        with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
            results = parallel(delayed(parallelize_preprocessing)(item_table_chunks[i],rating_table_chunks[i], notes_info,'sim') for i in range(num_core))

        for i,data in enumerate(results):
            if i == 0:
                item_table = data[0]
                rating_table = data[1]
            else:
                item_table = pd.concat([item_table, data[0]], axis = 0)
                rating_table = pd.concat([rating_table, data[1]], axis = 0)


        item_table.to_csv('/home/dhkim/Fragrance/data/item_table_sim.csv' ,encoding ='utf-8-sig',  index=False)
        rating_table.to_csv('/home/dhkim/Fragrance/data/rating_table_sim.csv' ,encoding ='utf-8-sig',  index=False)

