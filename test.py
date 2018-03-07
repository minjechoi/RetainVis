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


from functions import date_converter, decay_fn, get_dates, get_inverted_dates

parser = argparse.ArgumentParser()
parser.add_argument("--ver", help="which model to use", default='retain',
    type=str)
parser.add_argument("--emb", help="embedding size of model", default=128,
    type=int)
parser.add_argument("--hid", help="hidden size of model", default=128,
    type=int)
parser.add_argument("--epoch", help="number of epochs to load", default=20,
    type=int)
parser.add_argument("--decay", help="which decay function to use", default=1,
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

hid = args.hid
emb = args.emb
epochs = args.epoch
print(args)

# load data
with open('data/preprocessed/I50/list_data_2014.pckl','rb') as f1:
    L1 = pickle.load(f1)
with open('data/preprocessed/I50/list_data_2015.pckl','rb') as f1:
    L2 = pickle.load(f1)
# get training and test data
tr_data = L1[:3000]+L2[:3500]
t_data = L1[3000:]+L2[3500:]

if args.ver=='gru':
    model = GRU(emb, hid, 1, args.cuda)
else:
    model = RETAIN(emb, hid, 1, args.cuda)
if args.cuda:
    model.cuda()

# for epoch in [args.epoch]:
# for epoch in range(9,15):
for epoch in range(args.epoch):
    epoch+=1
    # if args.cuda:
    #     model.load_state_dict(torch.load('experiments/I50/saved_weights/%s_epochs_%d_cuda.pckl'%(args.ver,args.epoch)))
    # else:
    #     model.load_state_dict(torch.load('experiments/I50/saved_weights/%s_epochs_%d_cpu.pckl'%(args.ver,args.epoch)))
    if args.ver=='time':
        name = args.ver + '_' + str(args.decay)
    else:
        name = args.ver
    if args.cuda:
        file_name = 'experiments/I50/saved_weights/%s_epochs_%d_cuda.pckl'%(name,epoch)
    else:
        file_name = 'experiments/I50/saved_weights/%s_epochs_%d_cpu.pckl'%(name,epoch)
    # file_name = 'experiments/I50/saved_weights/time_epochs_best_cuda.pckl'
    model.load_state_dict(torch.load(file_name))
    model.eval()
    print("Loading from %s" %file_name)

    loss_list = []
    correct_list = []
    predict_list = []
    score_list = []
    for i in range(len(t_data)):
        X,y = t_data[i]
        date_list = []
        input_list = []
        for sample in X:
            _,dates_,inputs_,_ = zip(*sample)
            date_list.append(get_dates(dates_))
            input_list.append(list(inputs_))
        inputs = model.list_to_tensor(input_list)
        dates = Variable(torch.Tensor(date_list), requires_grad=False)
        # targets = Variable(torch.LongTensor(np.array(y,dtype=int)))
        if args.cuda:
            dates = dates.cuda()
            targets = targets.cuda()
        if args.ver=='time':
            outputs = model(inputs,dates)
        else:
            outputs = model(inputs)
        outputs = F.sigmoid(outputs.squeeze())

        # append to lists
        correct_list.extend(y)
        score_list.extend(outputs.data.cpu().tolist())
        predict_list.extend((outputs>0.5).data.tolist())
        # loss_list.append(loss.data[0])
    print('Epoch %d' %(epoch))
    str_acc = "Avg. ACC: %1.3f" %(accuracy_score(correct_list,predict_list))
    str_macro_auc = "Avg. AUC: %1.3f" %(roc_auc_score(correct_list,score_list,'macro'))
    str_prec = "Avg. prec: %1.3f" %(precision_score(correct_list,predict_list))
    str_recall = "Avg. rec: %1.3f" %(recall_score(correct_list,predict_list))
    log_list=[str_acc,str_macro_auc,str_prec,str_recall]
    for log in log_list:
        print(log)