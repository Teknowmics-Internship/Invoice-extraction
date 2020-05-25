import os
import glob
import re
import nltk
import spacy
import string
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
pd.options.mode.chained_assignment = None
import time 
import pdb
f = glob.glob(r'C:\\Users\\HP\\Desktop\\internship\\text\\*.txt')
np_array_values =[]
for files in f:
    data = pd.read_csv(files,sep="\t",header=None)
    n=data[0].count()
    b=""
    j=0
    for i in range(n): 
        a=data[0][i]
        b=b+"  "+a
        
    np_array_values.append(b)
    print(b)
p=pd.DataFrame(data=np_array_values, index=None, columns=["invoice"],dtype=None,copy= False)
p
p['invoice'][0]
q= p.to_csv(r"C:\\Users\\HP\\Desktop\\internship\\text\\‪invoices.csv",index= False) 
df =p[["invoice"]]
df["invoice"] = df["invoice"].astype(str)
p.head()
text_list = p['invoice'].values.tolist()
text_list = [re.sub('\n', '', x) for x in text_list] # IMPT: remove some existing '\n'
text_one_long = '\n '.join(text_list)
assert len(text_one_long.split('\n ')) == len(text_list) # assert to ensure join later
# check 
text_one_long[0:500]
def fast_clean_text_keep_newline(text):
    print("0 Start")
    assert isinstance(text, str)
    text = re.sub("‘", "'", text) # weird left quote
    text = re.sub("’", "'", text) # weird right quote
    print("1")
    text = re.sub("\\B@[A-Za-z_]+", ' ', text)
    print("2")
    text = re.sub("(https?://|https?://|http//www|www)\S+", ' ', text)
    print("3")
    text = re.sub('(?<=\S) \n', '\n', text)
    print("4")
    text = re.sub("[—¡“”…{}@[]%\\/""]", ' ', text)
    print("5")
    print("DONE!")
    return text   
start = time.time() 
text_one_cleaned = fast_clean_text_keep_newline(text_one_long)
end = time.time()
print("Fast clean took: %s seconds" % (end-start))
text_cleaned_split = text_one_cleaned.split('\n ') 
text_cleaned_split 
df_cleaned = p
df_cleaned['text_cleaned'] = text_cleaned_split 
df_cleaned
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text_wo_punct"] = df["invoice"].apply(lambda text: remove_punctuation(text))
df.head()
import warnings
warnings.filterwarnings("ignore")
df = df.astype(str)
df
