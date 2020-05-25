#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##### import glob
a=glob.glob('/home/lenovo/Documents/ner advanved/*.tsv')
a


# In[4]:


from PIL import Image      
import os.path, sys
import matplotlib.pyplot as plt


# In[1]:


# importing pandas module 
import pandas as pd 
  
# making data frame from csv file 
data = pd.read_csv("/home/lenovo/Documents/him.csv", sep='/t', encoding = 'unicode_escape') 
  
# dropping passed columns 
# data.drop(["Unnamed: 2"], axis = 1, inplace = True) 
  
# display 
data 


# In[2]:


#save as csv 
data.to_csv(r'docs(1).csv', index = False)


# In[23]:


#csv to tsv
import csv

with open('docs(1).csv','r') as csvin, open('ner_dataset1.txt', 'w') as tsvout:
    
    csvin = csv.reader(csvin)
    tsvout = csv.writer(tsvout, delimiter=',')

    for row in csvin:
          tsvout.writerow(row)


# In[1]:


# Convert .tsv file to dataturks json format.
import json
import logging
import sys
def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            if line[0:len(line)-1]!='\t':
                
                word,entity=line.split('\t')
                s+=word+" "
                #  print(s)
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        d={}
                        d['text']=word
                        d['start']=start
                        d['end']=start+len(word)-1  
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d)
                start+=len(word)+1
            else:
                
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])):
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l)                            
                             
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp)                          
                print(data_dict)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None
tsv_to_json_format(r"ner_dataset1(3)(1).txt",r'ner_dataset11.json','abc')


# In[2]:



#spacy format
import plac
import logging
import argparse
import sys
import os
import json
import pickle

# @plac.annotations(input_file=("/content/demo.json", "option", "i", str), output_file=("/content/out.json", "option", "o", str))
def main(input_file='ner_dataset11.json', output_file='out.json'):
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        # print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None
main()


# In[3]:


from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import os
from os import listdir



# New entity labels
# Specify the new entity labels which you want to add here
LABEL = [u'invoiceno', u'invoicedate', u'customer', u'vendor', u'product', u'address', u'amount']

with open ('out.json', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

# @plac.annotations(
#     model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
#     new_model_name=("New model name for model meta.", "option", "nm", str),
#     output_dir=("Optional output directory", "option", "o", Path),
#     n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='new_model', output_dir='/home/lenovo/Documents/teknowmics/', n_iter=10):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')
    # for i in LABEL:
    #     ner.add_label(i)   # Add new entity labels to entity recognizer

    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)
        # save model to output directory
    output_dir1="/home/lenovo/Documents/him11"
    if output_dir1 is not None:
        output_dir1 = Path(output_dir1)
        if not output_dir1.exists():
            output_dir1.mkdir()
        nlp.to_disk(output_dir1)
        print("Saved model to", output_dir1)
        
main()


# In[ ]:




