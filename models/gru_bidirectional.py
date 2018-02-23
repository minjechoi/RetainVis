import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cuda_flag=False):
        super(GRU,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        emb = nn.Embedding(1400,input_size)
        self.emb = emb.weight

        self.RNN = nn.GRU(input_size,hidden_size,batch_first=True, bidirectional=True)
        self.W_out = nn.Linear(hidden_size*2,num_classes,bias=False)

    def forward(self, inputs, timestamps):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get outputs
        outputs = self.RNN(embedded) # [b, seq, hid*2]
        outputs = self.W_out(outputs.contiguous()[:,-1,:]) # [b, num_classes]
        return outputs

    def list_to_tensor(self,inputs): # deals with input preprocessing
        # ver 2
        input_tensor = Variable(torch.Tensor(len(inputs),len(inputs[0]),1400).zero_())
        if self.cuda_flag:
            input_tensor = input_tensor.cuda()
        for i,sample in enumerate(inputs):
            for j,visit in enumerate(sample):
                for item in visit:
                    input_tensor[i,j,item]=1
        return input_tensor