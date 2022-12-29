import pandas as pd
import csv
import os
import pickle
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_pandas
import sys
import re
from collections import defaultdict
from konlpy.tag import Kkma
import time
import pdb
import numpy as np
from konlpy.tag import Okt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import ast
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl

from collections import Counter
from tqdm import tqdm
from collections import Counter, OrderedDict
import pdb
import sys

def word_freq(data):
    others = []
    noun = []
    adj = []

    for sentence in tqdm(data['review_text']):
        try: 
            sentence = ast.literal_eval(sentence)
            for word, tag in sentence:
                if (tag=='NN') or (tag=='NNPS') or (tag=='NNP') : noun.append(word)
                elif (tag=='JJ') or (tag=='JJR') or (tag=='JJS') : adj.append(word)
                else: others.append(word)
        except: continue

    return [noun, adj, others]

def word_freq_naive(data):
    others = []
    for sentence in tqdm(data['review_text']):
        try: 
            sentence = sentence.split(' ')
            for word in sentence:
                others.append(word)
        except: continue

    return others


if __name__ == '__main__':
    name = sys.argv[-1]
    num_core = os.cpu_count()


#-----------------------------------------------naive------------------
    data = pd.read_csv(config.token_path)

    print(f"Parallelizing with {num_core} cpus")
    data_chunks = np.array_split(data, num_core)
    print(f"Parallelizing with {num_core} cpus")
    with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
        results = parallel(delayed(word_freq)(data_chunks[i]) for i in range(num_core))

    noun = []
    adj = []
    others = []
    for i,data in enumerate(results):
        noun.extend(data[0]) 
        adj.extend(data[1])
        others.extend(data[2])

    noun_counts = Counter(noun)
    adj_counts = Counter(adj)
    other_counts = Counter(others)
            
    noun = OrderedDict(noun_counts.most_common())
    adj = OrderedDict(adj_counts.most_common())
    others = OrderedDict(other_counts.most_common())


    noun_freq = pd.DataFrame(data = noun.values(), index = noun.keys(), columns=['count'])
    noun_freq.to_csv('/home/dhkim/Fragrance/text/noun.csv', encoding = 'utf-8-sig')

    adj_freq = pd.DataFrame(data = adj.values(), index = adj.keys(), columns=['count'])
    adj_freq.to_csv('/home/dhkim/Fragrance/text/adj.csv', encoding = 'utf-8-sig')

    other_freq = pd.DataFrame(data = others.values(), index = others.keys(), columns=['count'])
    other_freq.to_csv('/home/dhkim/Fragrance/text/other.csv', encoding = 'utf-8-sig')



