from __future__ import print_function
import os
import pickle
import numpy as np
import torch
from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, pad_feature
from torch.utils.data import Dataset

class Mosi_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(Mosi_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataroot = os.path.join(dataroot, 'MOSI')
        self.private_set = name == 'private'
        if name == 'train':
            name = 'trainval'
        if name == 'valid':
            name = 'test'
        word_file = os.path.join(self.dataroot, name + "_sentences.p")
        audio_file = os.path.join(self.dataroot, name + "_mels.p")
        y_s_file = os.path.join(self.dataroot, name + "_sentiment.p")

        self.key_to_word = pickle.load(open(word_file, "rb"))
        self.key_to_audio = pickle.load(open(audio_file, "rb"))
        self.key_to_label = pickle.load(open(y_s_file, "rb"))
        self.set = list(self.key_to_label.keys())

        # filter y = 0 for binary task (https://github.com/A2Zadeh/CMU-MultimodalSDK/tree/master/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSI)
        if self.args.task_binary:
            for key in self.key_to_label.keys():
                if self.key_to_label[key] == 0.0:
                    print("2-class Sentiment, removing key ", key)
                    self.set.remove(key)

        for key in self.set:
            if not (key in self.key_to_word and
                    key in self.key_to_audio and
                    key in self.key_to_label):
                print("Not present everywhere, removing key ", key)
                self.set.remove(key)

        # Plot temporal dimension of feature
        # t = []
        # for key in self.key_to_word.keys():
        #     x = np.array(self.key_to_word[key]).shape[0]
        #     t.append(x)
        # plot(t)
        # sys.exit()

        # Creating embeddings and word indexes
        self.key_to_sentence = tokenize(self.key_to_word)
        if token_to_ix is not None:
            self.token_to_ix = token_to_ix
        else: # Train
            self.token_to_ix, self.pretrained_emb = create_dict(self.key_to_sentence, self.dataroot)
        self.vocab_size = len(self.token_to_ix)

        self.l_max_len = 30
        self.a_max_len = 60

    def __getitem__(self, idx):
        key = self.set[idx]
        L = sent_to_ix(self.key_to_sentence[key], self.token_to_ix, max_token=self.l_max_len)
        A = pad_feature(self.key_to_audio[key], self.a_max_len)
        V = np.zeros(1) # not using video, insert dummy

        y = self.key_to_label[key]
        if self.args.task_binary:
            c = 0 if y < 0.0 else 1
        else:
            c = int(round(y)) + 3  # from -3;3 to 0;6
        y = np.array(c)
        return key, torch.from_numpy(L), torch.from_numpy(A), torch.from_numpy(V).float(), torch.from_numpy(y)

    def __len__(self):
        return len(self.set)