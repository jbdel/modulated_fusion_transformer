from __future__ import print_function
import os
import pickle
import numpy as np
import torch
from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, pad_feature
from torch.utils.data import Dataset

class Vgaf_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(Vgaf_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataroot = os.path.join(dataroot, 'VGAF')
        self.private_set = name == 'private'

        if name == 'test':
            name = 'valid'

        word_file = os.path.join(self.dataroot, name + "_sentences.p")
        audio_file = os.path.join(self.dataroot, name + "_mels.p")
        y_file = os.path.join(self.dataroot, name + "_emotions.p")

        self.key_to_word = pickle.load(open(word_file, "rb"))
        self.key_to_audio = pickle.load(open(audio_file, "rb"))
        self.key_to_label = pickle.load(open(y_file, "rb"))
        self.set = list(self.key_to_label.keys())

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
        # print(max(t))
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
        self.a_max_len = 26

    def __getitem__(self, idx):
        key = self.set[idx]
        L = sent_to_ix(self.key_to_sentence[key], self.token_to_ix, max_token=self.l_max_len)
        A = pad_feature(self.key_to_audio[key], self.a_max_len)
        V = np.zeros(1) # not using video, insert dummy

        y = self.key_to_label[key]
        y = np.array(int(y)-1) #from 1,3 to 0,2
        return key, torch.from_numpy(L), torch.from_numpy(A), torch.from_numpy(V).float(), torch.from_numpy(y)

    def __len__(self):
        return len(self.set)