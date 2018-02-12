import pickle
import numpy as np
import random
import os

class DataLoader:

    def __init__(self,batch_size, data_dir, mode='train', max_seq_length=200, min_seq_length=5):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.mode = mode
        if mode=='train':
            self.train_list = os.listdir(os.path.join(data_dir,'train'))
            self.val_list = os.listdir(os.path.join(data_dir,'val'))
        elif mode=='val':
            self.val_list = os.listdir(os.path.join(data_dir,'val'))
        else:
            self.test_list = os.listdir(os.path.join(data_dir,'test'))
        self.load_dict()
        self.batch_count = 0
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length

    def load_dict(self):
        with open('data/dictionaries/d2i.pckl','rb') as f:
            self.d2i = pickle.load(f)
        with open('data/dictionaries/sick_converter.pckl','rb') as f:
            self.s2i = pickle.load(f)
        with open('data/dictionaries/p2i.pckl','rb') as f:
            self.p2i = pickle.load(f)

    # for each epoch, select lengths to be put into the model
    def shuffle_list(self):
        if self.mode=='train':
            random.shuffle(self.train_list)
        else:
            random.shuffle(self.test_list)

    def load_batch_file(self,file): # loads a pickle file containing lists of (list of tuples)
        with open(os.path.join(self.data_dir,self.mode,file),'rb') as f:
            self.batches = pickle.load(f)
        self.batch_count = int(np.ceil(len(self.batches)/self.batch_size)) # number of batches

    def get_batch(self): # gets a single batch
        inputs_and_labels = self.batches[:self.batch_size]
        self.batches = self.batches[self.batch_size:]
        self.inputs = []
        self.targets = []
        for tup_list in inputs_and_labels:
            tmp1 = []
            tmp2 = []
            for tup in tup_list[:self.max_seq_length]:
                date, fom, out_list, out = tup
                tmp1.append(out_list)
                tmp2.append(out)
            self.inputs.append(tmp1)
            self.targets.append(tmp2)
        return self.inputs, self.targets