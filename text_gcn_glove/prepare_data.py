#!/usr/bin/python
#-*-coding:utf-8-*-
import pandas as pd
# dataset_name = 'own'
# sentences = ['Would you like a plain sweater or something else?​', 'Great. We have some very nice wool slacks over here. Would you like to take a look?']
# labels = ['Yes' , 'No' ]
# train_or_test_list = ['train', 'test']

dataset_name = 'fakenews'
df = pd.read_csv("/4tb/nabarun/nlp/FakeNewsDeepLearning/truefake.csv")
sentences = df['text']
labels = df['label']
train_or_test_list = df['type']

# print(sentences, labels, train_or_test_list)

meta_data_list = []

for i in range(len(sentences)):
    meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
    meta_data_list.append(meta)

meta_data_str = '\n'.join(meta_data_list)

f = open('data/' + dataset_name + '.txt', 'w')
f.write(meta_data_str)
f.close()

corpus_str = '\n'.join(sentences)

f = open('data/corpus/' + dataset_name + '.txt', 'w')
f.write(corpus_str)
f.close()