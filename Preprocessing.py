#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import glob
f = glob.glob(r'C:\\Users\\HP\\Desktop\\internship\\text\\*.txt')


# In[27]:


f


# In[28]:


import pandas as pd
import numpy as np
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
        


# In[29]:


print(np_array_values)


# In[30]:


p=pd.DataFrame(data=np_array_values, index=None, columns=["invoice"],dtype=None,copy= False)
p


# In[31]:


p['invoice'][0]


# In[32]:


q= p.to_csv(r"C:\\Users\\HP\\Desktop\\internship\\text\\‪invoices.csv",index= False) 
q    


# In[47]:


import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None
import time 
import pdb


# In[34]:


df =p[["invoice"]]
df["invoice"] = df["invoice"].astype(str)
p.head()


# In[49]:


text_list = p['invoice'].values.tolist()
text_list = [re.sub('\n', '', x) for x in text_list] # IMPT: remove some existing '\n'
text_one_long = '\n '.join(text_list)
assert len(text_one_long.split('\n ')) == len(text_list) # assert to ensure join later

# check 
text_one_long[0:500]


# In[91]:


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


# In[92]:


start = time.time() 
text_one_cleaned = fast_clean_text_keep_newline(text_one_long)
end = time.time()
print("Fast clean took: %s seconds" % (end-start))


# In[93]:


text_cleaned_split = text_one_cleaned.split('\n ') 
text_cleaned_split 


# In[94]:


df_cleaned = p
df_cleaned['text_cleaned'] = text_cleaned_split 
df_cleaned


# In[36]:


def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text_wo_punct"] = df["invoice"].apply(lambda text: remove_punctuation(text))
df.head()


# In[37]:


import warnings
warnings.filterwarnings("ignore")



import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm


# In[41]:


df = df.astype(str)
df


# In[46]:





# In[ ]:





# In[ ]:




