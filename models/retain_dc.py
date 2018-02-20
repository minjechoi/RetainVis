import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np

class RETAIN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, cuda_flag=False):
        super(RETAIN,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cuda_flag = cuda_flag
        self.release = False # if set to true, then we return all values in computing
        emb = nn.Embedding(1400,input_size)
        self.emb = emb.weight

        emb_trg = nn.Embedding(num_classes,num_classes)
        emb_trg.weight.data.copy_(torch.from_numpy(np.eye(num_classes)))
        self.emb_trg = emb_trg

        self.RNN1 = nn.GRU(input_size,hidden_size,1,batch_first=True,bidirectional=True) # for alpha
        self.RNN2 = nn.GRU(input_size,hidden_size,1,batch_first=True,bidirectional=True) # for Beta
        self.RNN3 = nn.GRU(input_size,hidden_size,1,batch_first=True,bidirectional=True) # for attention
        self.wa = nn.Linear(hidden_size*2,1,bias=False)
        self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)
        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)
        self.wa2 = nn.Linear(hidden_size*2, 1)
        self.w_lever = nn.Linear(hidden_size*2, 2)

    def forward(self, inputs, targets):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        outputs1 = self.RNN1(embedded) # [b x seq x 128*2]
#         print(outputs1)
        E = self.wa(outputs1[0].contiguous().view(-1, self.hidden_size*2)) # [b*seq x 1]
        alpha = F.softmax(E.view(b,seq),1) # [b x seq]
        if self.release:
            self.alpha = alpha

        # get beta coefficients
        outputs2 = self.RNN2(embedded) # [b x seq x 128]
        outputs2 = self.Wb(outputs2[0].contiguous().view(-1,self.hidden_size*2)) # [b*seq x hid]
        Beta = torch.tanh(outputs2).view(b, seq, self.hidden_size) # [b x seq x 128]
        if self.release:
            self.Beta = Beta
        outputs_calc = self.compute(embedded, Beta, alpha) # [b, num_classes]
        outputs_calc = F.softmax(outputs_calc,1)

        # get outputs obtained by copying mechanism
        outputs3 = self.RNN3(embedded) # [b, seq, 128]
        E = self.wa2(outputs3[0].contiguous().view(-1, self.hidden_size*2)) # [b*seq x 1]
        alpha2 = F.softmax(E.view(b,seq),1).unsqueeze(2) # [b, seq, 1]
        alpha2 = alpha2[:,1:,:] # [b,seq-1,1]

        # targets: list of list
        targets = Variable(torch.LongTensor(targets)[:,:-1],requires_grad=False)
        if self.cuda_flag:
            targets = targets.cuda()
        inputs2 = self.emb_trg(targets) # [b, seq-1, num_classes]
        outputs_copy = (inputs2*alpha2.expand_as(inputs2)).sum(1) # [b, num_classes]
        outputs_copy = F.softmax(outputs_copy,1)

        # for inputs, we apply another contribution score that helps add up to output
        # this way, we can obtain a vector that takes in all of the inputs

        lever = outputs3[0].sum(1) # [b, 128]
        lever = F.softmax(self.w_lever(lever),1) # [b,2]

        outputs = outputs_calc*lever[:,0:1].expand_as(outputs_calc) + outputs_copy*lever[:,1:2].expand_as(outputs_copy) # [b, num_classes]
        return outputs

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
