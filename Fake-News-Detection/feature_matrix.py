import json
import os

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bert import run_classifier
from bert import tokenization
import spacy

from utils.UtilWordEmbedding import DocModel, DocPreprocess
import multiprocessing
import sys
from gensim.models.word2vec import Word2Vec


class FeatureMatrix:
    def __init__(self, base_path):
        self.base_path = base_path

    def get_data_from_file(self, file):
        with open(file) as json_file:
            data = json.load(json_file)
            if file.split("/")[-2] == "FakeNewsContent":
                return [data['text'], 1]
            else:
                return [data['text'], 0]

    def get_folder_data(self, folder):
        data_list = []
        for subfolder in ["FakeNewsContent", "RealNewsContent"]:
            for file in os.listdir(self.base_path + folder + "/" + subfolder):
                if file.endswith(".json"):
                    data_list.append(self.get_data_from_file(self.base_path + folder + "/" + subfolder + "/" + file))

        data_df = pd.DataFrame(data_list, columns=["text", "label"])

        return data_df

    def get_all_data(self):
        print("Fetching BuzzFeed data")
        bf_data_df = self.get_folder_data("BuzzFeed")

        # print("Fetching PolitiFact data")
        # pf_data_df = self.get_folder_data("PolitiFact")

        # all_data_df = pd.concat([bf_data_df, pf_data_df])

        return bf_data_df
    
    def loadWord2Vec(self, filename):
        """Read Word Vectors"""
        vocab = []
        embd = []
        word_vector_map = {}
        file = open(filename, 'r')
        for line in file.readlines():
            row = line.strip().split(' ')
            if(len(row) > 2):
                vocab.append(row[0])
                vector = row[1:]
                length = len(vector)
                for i in range(length):
                    vector[i] = float(vector[i])
                embd.append(vector)
                word_vector_map[row[0]] = vector
        print('Loaded Word Vectors!')
        file.close()
        return vocab, embd, word_vector_map

    def get_feature_matrix(self, dataset = "BuzzFeed"):
        if dataset in ["BuzzFeed", "PolitiFact"]:
            all_data_df = self.get_folder_data(folder=dataset)
        else:
#             print(dataset)
            all_data_df = pd.read_csv(self.base_path+"truefake.csv")
        
        all_data_df = all_data_df.sample(frac=1)

        nlp = spacy.load('en_core_web_md')
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        all_docs = DocPreprocess(nlp, stop_words, all_data_df['text'], all_data_df['label'])
        workers = multiprocessing.cpu_count()
        dm_args = {
            'dm': 1,
            'dm_mean': 1,
            'vector_size': 128,
            'window': 5,
            'negative': 5,
            'hs': 0,
            'min_count': 2,
            'sample': 0,
            'workers': workers,
            'alpha': 0.025,
            'min_alpha': 0.025,
            'epochs': 100,
            'comment': 'alpha=0.025'
        }
        dm = DocModel(docs=all_docs.tagdocs, **dm_args)
        dm.custom_train()
        features = []
        for i in range(len(dm.model.docvecs)):
            features.append(dm.model.docvecs[i])

        column_names = ["feature" + str(i) for i in range(128)]
        features_df = pd.DataFrame(features, columns=column_names)
        features_df["label"] = all_data_df['label']
    
        return features_df


if __name__ == "__main__":
    base_path = "../dataset/"

    adj = FeatureMatrix(base_path)
    res = adj.get_feature_matrix("FakeNews")
