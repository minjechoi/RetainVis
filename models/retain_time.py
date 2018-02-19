import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time

class RETAIN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cuda_flag=False):
        super(RETAIN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        emb = nn.Embedding(1400,input_size)
        self.emb = emb.weight

        self.RNN1 = TimeLSTM(input_size,hidden_size,cuda_flag)
        self.RNN2 = TimeLSTM(input_size,hidden_size,cuda_flag)
        if cuda_flag:
            self.RNN1.cuda()
            self.RNN2.cuda()

        self.wa = nn.Linear(hidden_size,1,bias=False)
        self.Wb = nn.Linear(hidden_size,hidden_size,bias=False)
        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)

    def forward(self, inputs, timestamps):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        outputs1 = self.RNN1(embedded, timestamps) # [b x seq x 128]
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
    def __init__(self, input_size, hidden_size, cuda_flag=False):
        # assumes that batch_first is always true
        super(TimeLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(input_size, hidden_size*4)
        self.U_all = nn.Linear(hidden_size, hidden_size*4)
        self.W_d = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, timestamps):
        # inputs: [b, seq, hid]
        # h: [b, hid]
        # c: [b, hid]
        b,seq,hid = inputs.size()
        # print("inputs: ",inputs[0,:10])
        h = Variable(torch.randn(b,hid), requires_grad=False)
        c = Variable(torch.randn(b,hid), requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        # print("W_all: ",self.W_all.weight[0,:10])
        for s in range(seq):
            # c_s1: [b, hid]
            # c_s2: [b, hid]
            # c_l: [b, hid]
            # c_adj: [b, hid]
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:,s:s+1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            # print('c_adj: ',c_adj[0,:10])
            outs = self.W_all(h)+self.U_all(inputs[:,s])
            # print(outs.size())
            f, i, o, c_tmp = torch.chunk(outs,4,1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.sigmoid(c_tmp)
            # print('c_adj: ',c_adj.size())
            # print('f: ',f.size())
            # print('i: ',i.size())
            # print('c_tmp: ',c_tmp.size())
            c = f*c_adj + i*c_tmp
            h = o*F.tanh(c)
            outputs.append(h)
        outputs = torch.stack(outputs,1)
        return outputs