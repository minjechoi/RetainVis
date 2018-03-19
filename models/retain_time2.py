import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time

class RETAIN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
        cuda_flag=False, bidirectional=True, decay_ver=1):
        super(RETAIN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        self.bidirectional = bidirectional
        self.decay_ver = decay_ver
        if decay_ver==3:
            self.decay_params = nn.Parameter(torch.Tensor([np.e,0,0]))
            # if cuda_flag:
                # self.decay_params = self.decay_params.cuda()

        emb1 = nn.Embedding(1400,input_size)
        self.emb1 = emb1.weight
        emb2 = nn.Embedding(1400,input_size)
        self.emb2 = emb2.weight

        if decay_ver==4:
            self.input_size += 3
        self.RNN1 = nn.LSTM(self.input_size,hidden_size,1,batch_first=True,bidirectional=self.bidirectional)
        self.RNN2 = nn.LSTM(self.input_size,hidden_size,1,batch_first=True,bidirectional=self.bidirectional)

        if self.bidirectional:
            self.wa = nn.Linear(hidden_size*2,1,bias=False)
            self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)
        else:
            self.wa = nn.Linear(hidden_size,1,bias=False)
            self.Wb = nn.Linear(hidden_size,hidden_size,bias=False)

        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)

    def forward(self, inputs, timestamps):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb1).view(b,seq,-1)
        embedded2 = torch.mm(inputs.view(-1,features),self.emb2).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        if self.decay_ver==4:
            dates = timestamps
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

    def decay_fn(self, dates):
        if self.decay_ver==1:
            return 1/dates
        elif self.decay_ver==2:
            return 1/torch.log(np.e+dates)
        elif self.decay_ver==3:
            a,b,c = self.decay_params
            return F.tanh(1/torch.log(a+dates*b)*c)

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
        # embedded =  torch.mm(input_tensor, self.emb) # [samples*sequences, hidden]
        # return embedded.view(len(inputs),len(inputs[0]),-1)

    # fixed version of interpret
    def interpret(self,u,v,i,o):
        # u: user number, v: visit number, i: input element number, o: output sickness
        a = self.alpha[u][v] # [1]
        B = self.Beta[u][v] # [h]
        W_emb = self.emb[i] # [h]
        W = self.W_out.weight[o] # [h]
        # b = self.W_out.state_dict()['bias'][t]
        out = a*torch.dot(W,(B*W_emb))
        return out

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cuda_flag=False, bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(input_size, hidden_size*4)
        self.U_all = nn.Linear(hidden_size, hidden_size*4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional

    def forward(self, inputs, timestamps, reverse=False):
        # inputs: [b, seq, hid]
        # h: [b, hid]
        # c: [b, hid]
        b,seq,hid = inputs.size()
        h = Variable(torch.Tensor(b,hid).zero_(), requires_grad=False)
        c = Variable(torch.randn(b,hid).zero_(), requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:,s:s+1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h)+self.U_all(inputs[:,s])
            f, i, o, c_tmp = torch.chunk(outs,4,1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.sigmoid(c_tmp)
            c = f*c_adj + i*c_tmp
            h = o*F.tanh(c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs,1)
        return outputs