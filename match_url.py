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





item_table = pd.read_csv(config.item_table_path, encoding ='utf-8-sig')
fr_url = pd.read_csv("/home/dhkim/Fragrance/data/fragrance_data2.csv", encoding ='utf-8-sig')

urls = []
i = 0
j = 0
while i < len(item_table):
    if item_table.loc[i, 'name'] == fr_url.loc[j, 'name']:
        urls.append(fr_url.loc[i, 'url'])
        i += 1
        j += 1
    else:j += 1
    if i % 1000 == 0: print('-', end = '')
item_table['url'] = urls
item_table.to_csv("/home/dhkim/Fragrance/data/DB.csv", encoding ='utf-8-sig', index = False)
