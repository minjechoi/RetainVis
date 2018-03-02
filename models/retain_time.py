import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import time

class RETAIN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes,
        cuda_flag=False, bidirectional=False):
        super(RETAIN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        self.bidirectional = bidirectional

        emb = nn.Embedding(1400,input_size)
        self.emb = emb.weight

        self.RNN1 = TimeLSTM(input_size,hidden_size,cuda_flag)
        self.RNN2 = TimeLSTM(input_size,hidden_size,cuda_flag)
        if self.bidirectional:
            self.RNN1b = TimeLSTM(input_size,hidden_size,cuda_flag)
            self.RNN2b = TimeLSTM(input_size,hidden_size,cuda_flag)
        if cuda_flag:
            self.RNN1.cuda()
            self.RNN2.cuda()

        self.wa = nn.Linear(hidden_size,1,bias=False)
        self.Wb = nn.Linear(hidden_size,hidden_size,bias=False)
        if self.bidirectional:
            self.wa = nn.Linear(hidden_size*2,1,bias=False)
            self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)

        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)

    def forward(self, inputs, timestamps):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        outputs1 = self.RNN1(embedded, timestamps) # [b x seq x 128]
        outputs2 = self.RNN2(embedded, timestamps) # [b x seq x 128]
        if self.bidirectional:
            # add version where reverse is taken in
            embedded_b = embedded.data.cpu().numpy()
            embedded_b = Variable(torch.Tensor(np.flip(embedded_b,1)))
            timestamps_b = timestamps.data.cpu().numpy()
            timestamps_b = Variable(torch.Tensor(np.flip(timestamps_b,1)))
            if self.cuda_flag:
                embedded_b = embedded_b.cuda()
                timestamps_b = timestamps_b.cuda()
            outputs1b = self.RNN1b(embedded_b, timestamps_b, reverse=True)
            outputs2b = self.RNN2b(embedded_b, timestamps_b, reverse=True)
            outputs1 = torch.stack([outputs1,outputs1b],2)
            outputs2 = torch.stack([outputs2,outputs2b],2)

        E = self.wa(outputs1.contiguous().view(-1, self.hidden_size)) # [b*seq x 1]
        alpha = F.softmax(E.view(b,seq),1) # [b x seq]
        if self.release:
            self.alpha = alpha

        # get beta coefficients
        outputs2 = self.RNN2(embedded, timestamps) # [b x seq x 128]
        outputs2 = self.Wb(outputs2.contiguous().view(-1,self.hidden_size)) # [b*seq x hid]
        Beta = torch.tanh(outputs2).view(b, seq, self.hidden_size) # [b x seq x 128]
        if self.release:
            self.Beta = Beta

        return self.compute(embedded, Beta, alpha)


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