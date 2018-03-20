import torch
from torch import optim,nn
import torch.nn.functional as F
from torch.autograd import Variable
import gc
import numpy as np
import time
import os
import random
import pickle
import argparse
from functions import get_dates
# from sklearn.metrics import roc_auc_score as AUC
# from sklearn.metrics import average_precision_score as AUCPR

torch.manual_seed(1000)
parser = argparse.ArgumentParser()
parser.add_argument("--ver", help="which model to use", type=str)
parser.add_argument("--task", help="which data to test on", type=str)
parser.add_argument("--hid", help="hidden size of model", type=int)
parser.add_argument("--epoch", help="number of epochs", default=30, type=int)
parser.add_argument("--lr", help="learning rate size", type=float)
parser.add_argument("--time_ver", help="which time function to use (0:none, 1:add)", type=int)
parser.add_argument("--cuda", help="whether to use cuda", action="store_true")

args = parser.parse_args()
ver = args.ver.lower().strip()
task = args.task
hid = args.hid
emb = args.hid
lr = args.lr
time_ver = args.time_ver
cuda_flag = args.cuda
epochs = args.epoch

# load model
if ver=='retain':
    from models.retain_bidirectional import RETAIN
    model = RETAIN(emb, hid, 1, cuda_flag)
elif ver=='ex':
    from models.retain_ex import RETAIN_EX
    model = RETAIN_EX(emb, hid, 1, cuda_flag=cuda_flag, time_ver=time_ver)
elif ver=='gru':
    from models.gru_bidirectional import GRU
    model = GRU(emb, hid, 1, cuda_flag)
else:
    print("Error! --ver must be either 'retain', 'ex', or 'gru'")
    import sys
    sys.exit()
if cuda_flag:
    model.cuda()

# set save directories
if ver=='ex':
    # e.g. experiments/H26/ex-1_128_0.01/
    save_dir = 'experiments/%s/%s-%d_%d_%s'%(task,ver,time_fn,hid,str(lr))
else:
    # e.g. experiments/H26/gru_128_0.01/
    save_dir = 'experiments/%s/%s_%d_%s'%(task,ver,hid,str(lr))
log_dir = os.path.join(save_dir,'logs')
weight_dir = os.path.join(save_dir,'saved_weights')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(weight_dir):
    os.makedirs(weight_dir)
log_file = os.path.join(log_dir,'log.txt')
val_file =os.path.join(log_dir,'val.txt')

# load data
with open('data/%s/train.pckl'%task,'rb') as f:
    tr_data = pickle.load(f)
# with open('data/%s/val.pckl'%task,'rb') as f:
#     val_data = pickle.load(f)

# set optimizer and loss
criterion = nn.BCELoss()
opt = optim.Adam(model.parameters(), lr=lr)
stamp = 100

def print_and_save(log_file,string):
    with open(log_file,'a') as f:
        f.write(string+'\n')

def calculate(X,y,model,ver,cuda_flag,time_ver):
    date_list = []
    input_list = []
    for sample in X:
        _,dates_,inputs_,_ = zip(*sample)
        date_list.append(get_dates(dates_))
        input_list.append(list(inputs_))
    inputs = model.list_to_tensor(input_list)
    targets = Variable(torch.Tensor(np.array(y,dtype=int)))
    if cuda_flag:
        targets = targets.cuda()
    if (ver=='ex'):
        if time_ver==1:
            dates = Variable(torch.Tensor(date_list), requires_grad=False)
            if cuda_flag:
                dates = dates.cuda()
        else:
            dates = date_list
        outputs = model(inputs,dates)
    else:
        outputs = model(inputs)
    outputs = F.sigmoid(outputs.squeeze())
    return outputs, targets

cnt = 0
for epoch in range(epochs):
    # train model
    model.train()
    str1 = '========= Epoch %d ============' %(epoch+1)
    print_and_save(log_file,str1)
    random.shuffle(tr_data)
    loss_list = []
    for i in range(len(tr_data)):
        X,y = tr_data[i]
        cnt+=1
        model.zero_grad()
        outputs, targets = calculate(X,y,model,ver,cuda_flag,time_fn)
        loss = criterion(outputs,targets)
        loss.backward()
        loss_list.append(loss.data[0])
        opt.step()
        gc.collect()
        if (cnt%stamp==0):
            log_data = "Epoch %d,[%d],[%d],%1.3f" %(epoch+1,i+1,cnt,np.mean(loss_list))
            loss_list = []
            print_and_save(log_file,log_data)

    # save model
    model.cpu()
    torch.save(model.state_dict(),os.path.join(weight_dir,'%d_cpu.pckl'%(epoch+1)))
    if cuda_flag:
        model.cuda()
        torch.save(model.state_dict(),os.path.join(weight_dir,'%d_cuda.pckl'%(epoch+1)))

    # validate model
    # model.eval()
    # correct_list = []
    # score_list = []
    # for i in range(len(val_data)):
    #     X,y = val_data[i]
    #     outputs, targets = calculate(X,y,model,ver,cuda_flag)
    #     correct_list.extend(y)
    #     score_list.extend(outputs.data.cpu().tolist())

    # str_auc = "Epoch %d,AUC,%1.3f" %(epoch+1,AUC(correct_list,score_list))
    # str_aucpr = "Epoch %d,AUCPR,%1.3f" %(epoch+1,AUCPR(correct_list,score_list))
    # print_and_save(val_file,str_auc)
    # print_and_save(val_file,str_aucpr)