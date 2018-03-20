import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time

class RETAIN_EX(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
        cuda_flag=False, bidirectional=True, time_ver=1):
        super(RETAIN_EX,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        self.bidirectional = bidirectional
        self.time_ver = time_ver
        emb1 = nn.Embedding(1400,input_size)
        self.emb1 = emb1.weight
        emb2 = nn.Embedding(1400,input_size)
        self.emb2 = emb2.weight

        if self.time_ver==1:
            self.input_size += 3
        self.RNN1 = nn.LSTM(self.input_size,hidden_size,
            1,batch_first=True,bidirectional=self.bidirectional)
        self.RNN2 = nn.LSTM(self.input_size,hidden_size,
            1,batch_first=True,bidirectional=self.bidirectional)

        if self.bidirectional:
            self.wa = nn.Linear(hidden_size*2,1,bias=False)
            self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)
        else:
            self.wa = nn.Linear(hidden_size,1,bias=False)
            self.Wb = nn.Linear(hidden_size,hidden_size,bias=False)
        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)

    def forward(self, inputs, dates):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb1).view(b,seq,-1)
        embedded2 = torch.mm(inputs.view(-1,features),self.emb2).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        if self.time_ver==1:
            dates = torch.stack([dates,1/dates,1/torch.log(np.e+dates)],2) # [b x seq x 3]
            embedded = torch.cat([embedded,dates],2)
        outputs1 = self.RNN1(embedded)[0]
        outputs2 = self.RNN2(embedded)[0]
        E = self.wa(outputs1.contiguous().view(b*seq, -1)) # [b*seq x 1]
        alpha = F.softmax(E.view(b,seq),1) # [b x seq]
        if self.release:
            self.alpha = alpha
        outputs2 = self.Wb(outputs2.contiguous().view(b*seq,-1)) # [b*seq x hid]
        Beta = torch.tanh(outputs2).view(b, seq, self.hidden_size) # [b x seq x 128]
        if self.release:
            self.Beta = Beta
        return self.compute(embedded2, Beta, alpha)

    # multiply to inputs
    def compute(self, embedded, Beta, alpha):
        b,seq,_ = embedded.size()
        outputs = (embedded*Beta)*alpha.unsqueeze(2).expand(b,seq,self.hidden_size)
        outputs = outputs.sum(1) # [b x hidden]
        return self.W_out(outputs) # [b x num_classes]

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

    # fixed version of interpret
    def interpret(self,u,v,i,o):
        # u: user number, v: visit number, i: input element number, o: output sickness
        a = self.alpha[u][v] # [1]
        B = self.Beta[u][v] # [h]
        W_emb = self.emb2[i] # [h]
        W = self.W_out.weight[o] # [h]
        out = a*(B*W_emb)
        # out = a*torch.dot(W,(B*W_emb))
        return torch.dot(W,out)