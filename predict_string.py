import numpy as np
import pandas as pd
import warnings
import pickle
import sys
import pdb
import config

if __name__ == '__main__':

    code = int(sys.argv[-1])
    item_table_path = config.item_table_path
    user_info_path = config.user_info_path


    print('-------Reading file...-------')
    item_table = pd.read_csv(item_table_path, encoding ='utf-8-sig')
    user_info = pd.read_csv(user_info_path,index_col=0 , encoding ='utf-8-sig')
    rating_info = pd.read_csv("/home/dhkim/Fragrance/data/rating_table2.csv" , encoding ='utf-8-sig')

    with open(config.user_code_path, 'rb') as f:
        user_code = pickle.load(f)
    with open(f'/home/dhkim/Fragrance/predict/{code}_predict.pkl', 'rb') as f:
        predict = pickle.load(f)
    result = pd.DataFrame(columns = ['user','list'])

    user = user_info.iloc[code].name
    liked = rating_info.loc[rating_info['user_id'] == user].dropna()
    liked_df = pd.DataFrame({'liked': liked['fragrance'], 'url':liked['url']})

    user_pr = [item_table.loc[i,'name'] for i in predict]
    prediction_df = pd.DataFrame({'predict': user_pr})


    prediction_df.to_csv( f'/home/dhkim/Fragrance/{user}_predict.csv', encoding ='utf-8-sig')
    liked_df.to_csv( f'/home/dhkim/Fragrance/{user}_like.csv', encoding ='utf-8-sig')
    print('-------Done-------')

