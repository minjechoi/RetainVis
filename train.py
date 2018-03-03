import torch
from torch import optim,nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import os
import pickle
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score


from functions import date_converter, decay_fn, get_inverted_dates


parser = argparse.ArgumentParser()
parser.add_argument("--ver", help="which model to use", default='retain',
    type=str)
parser.add_argument("--emb", help="embedding size of model", default=128,
    type=int)
parser.add_argument("--hid", help="hidden size of model", default=128,
    type=int)
parser.add_argument("--lr", help="learning rate size", default=0.001,
    type=float)
parser.add_argument("--epoch", help="number of epochs", default=30,
    type=int)
parser.add_argument("--cuda", help="whether to use cuda",
    action="store_true")

args = parser.parse_args()
args.ver = args.ver.lower().strip()
if args.ver=='retain':
    from models.retain_bidirectional import RETAIN
elif args.ver=='time':
    from models.retain_time import RETAIN
elif args.ver=='gru':
    from models.gru_bidirectional import GRU
else:
    print("Error! --ver must be either 'retain', 'time', or 'gru'")
    import sys
    sys.exit()

lr = args.lr
hid = args.hid
emb = args.emb
epochs = args.epoch
print(args)
filename = args.ver+'_'+time.ctime().replace(' ','').replace(':','')+'.txt'
file_dir = os.path.join('experiments/I50/logs',filename)
print("Saving logs at %s"%file_dir)
info = "Model: %s\nEmb size: %d\nHid size: %d\n Cuda: %s\n\
        LR: %1.5f\nEpochs: %d\n" %(args.ver,args.emb,args.hid,str(args.cuda),args.lr,args.epoch)
print(info)
with open(file_dir,'a') as f:
    f.write(info)
    f.write('\n\n')

# load data
with open('data/preprocessed/I50/list_data_2014.pckl','rb') as f1:
    L1 = pickle.load(f1)
with open('data/preprocessed/I50/list_data_2015.pckl','rb') as f1:
    L2 = pickle.load(f1)
# get training and test data
tr_data = L1[:3000]+L2[:3500]
t_data = L1[3000:]+L2[3500:]

if args.ver=='gru':
    model = GRU(emb, hid, 2, args.cuda)
else:
    model = RETAIN(emb, hid, 2, args.cuda)
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=lr)

stamp = 100
for epoch in range(epochs):
    str1 = '========= Epoch %d ============' %(epoch+1)
    with open(file_dir,'a') as f:
        f.write(str1)
        f.write('\n')

    log_list = []
    log_list.append(str1)
    loss_list = []
    correct_list = []
    predict_list = []
    score_list = []
    for i in range(len(tr_data)):
        X,y = tr_data[i]
        date_list = []
        input_list = []
        for sample in X:
            _,dates_,inputs_,_ = zip(*sample)
            date_list.append(get_inverted_dates(dates_))
            input_list.append(list(inputs_))
        inputs = model.list_to_tensor(input_list)
        dates = Variable(torch.Tensor(date_list), requires_grad=False)
        targets = Variable(torch.LongTensor(np.array(y,dtype=int)))
        if args.cuda:
            dates = dates.cuda()
            targets = targets.cuda()
        if args.ver=='time':
            outputs = model(inputs,dates)
        else:
            outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        opt.step()
        if (i%100==0) & (i!=0):
            log_data = "Epoch %d: [%d] Loss: %1.3f" %(epoch+1,i,loss.data[0])
            print(log_data)
            with open(file_dir,'a') as f:
                f.write(log_data+'\n')
        # append to lists
        correct_list.extend(y)
        score_list.extend(outputs[:,1].data.cpu().tolist())
        predict_list.extend(outputs.topk(1)[1].data.cpu().squeeze().tolist())
        loss_list.append(loss.data[0])
    torch.save(model.state_dict(),'experiments/I50/saved_weights/%s_epochs_%d.pckl'%(args.ver,epoch+1))
    str2 = '-------------------------------'
    str_loss = "Avg. loss for %d steps: %1.3f" %(i+1,np.mean(loss_list))
    str2 = '-------------------------------'
    str_acc = "Avg. ACC  for %d steps: %1.3f" %(i+1,accuracy_score(correct_list,predict_list))
    str_micro_auc = "Avg. mAUC for %d steps: %1.3f" %(i+1,roc_auc_score(correct_list,score_list,'micro'))
    str_macro_auc = "Avg. MAUC for %d steps: %1.3f" %(i+1,roc_auc_score(correct_list,score_list,'macro'))
    str_prec = "Avg. prec for %d steps: %1.3f" %(i+1,precision_score(correct_list,predict_list))
    str_recall = "Avg. rec  for %d steps: %1.3f" %(i+1,recall_score(correct_list,predict_list))
    log_list.extend([str2,str_loss,str2,str_acc,str_micro_auc,str_macro_auc,str_prec,str_recall])
    # log_list = [str1,str_loss,str2,str_acc,str_micro_auc,str_macro_auc,str_prec,str_recall]
    with open(file_dir,'a') as f:
        f.write('\n'.join(log_list)+'\n')
    for log in log_list:
        print(log)
    loss_list = []
    correct_list = []
    predict_list = []
    score_list = []
f.close()
