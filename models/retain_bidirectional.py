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

        # self.emb = Variable(torch.Tensor(1400,input_size).normal_(),requires_grad=True)
        # if cuda_flag:
        #     self.emb = self.emb.cuda()

        self.RNN1 = nn.RNN(input_size,hidden_size,1,batch_first=True,bidirectional=True)
        self.RNN2 = nn.RNN(input_size,hidden_size,1,batch_first=True,bidirectional=True)
        self.wa = nn.Linear(hidden_size*2,1,bias=False)
        self.Wb = nn.Linear(hidden_size*2,hidden_size,bias=False)
        self.W_out = nn.Linear(hidden_size,num_classes,bias=False)

    def forward(self, inputs):
        # get embedding using self.emb
        b,seq,features = inputs.size()
        embedded = torch.mm(inputs.view(-1,features),self.emb).view(b,seq,-1)
        if self.release:
            self.embedded = embedded

        # get alpha coefficients
        outputs1 = self.RNN1(embedded) # [b x seq x 128*2]
#         print(outputs1)
        E = self.wa(outputs1[0].contiguous().view(-1, self.hidden_size*2)) # [b*seq x 1]
        alpha = F.softmax(E.view(b,seq)) # [b x seq]
        if self.release:
            self.alpha = alpha

        # get beta coefficients
        outputs2 = self.RNN2(embedded) # [b x seq x 128]
        outputs2 = self.Wb(outputs2[0].contiguous().view(-1,self.hidden_size*2)) # [b*seq x hid]
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
