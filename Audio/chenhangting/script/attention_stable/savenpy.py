# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/7
@author: chenhangting

@notes: a attention-lstm  for MOSI
    support early stopping
    it is a sccript for pretrain attention lstm
    the reason to do this is that the directly trained attention-lstm performs very bad
    logistic replace softmax
    reproducable and statibility for bilstm with dropout
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import sys
sys.path.append(r'../dataset')
from reverse_seq import reverse_padded_sequence
from dataset1d_early_stopping import AudioFeatureDataset
import pdb
import os,csv
from sklearn import metrics



# Training setttings
parser=argparse.ArgumentParser(description='PyTorch for audio sentiment classification in MOSI')
parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                    help='the num of cv set')
parser.add_argument('--batch_size',type=int,default=16,metavar='N', \
                    help='input batch size for training ( default 16 )')
parser.add_argument('--seed',type=int,default=1,metavar='S', \
                    help='random seed ( default 1 )')
parser.add_argument('--device_id',type=int,default=0,metavar='N', \
                    help="the device id")
parser.add_argument('--savedir',type=str,default='./',metavar='S', \
                    help='save files in the dir')
parser.add_argument('--loadpath',type=str,default='./model.pkl',metavar='S', \
                    help='load model in the path')


args=parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

emotion_labels=('positive','negative',)
superParams={'input_dim':153,
            'hidden_dim':128,
            'output_dim':1,
            'num_layers':4,
            'biFlag':2,
            'dropout':0.5,}

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# load dataset
featrootdir=r'../../features/basicfbank'
cvtxtrootdir='../CV/folds'
normfile=r'../../features/basicfbank/ms{}.npy'.format(args.cvnum)

dataset_train=AudioFeatureDataset(featrootdir=featrootdir, \
                                    cvtxtrootdir=cvtxtrootdir,feattype='npy', \
                                    cvnum=args.cvnum,mode='train',normflag=1,\
                                    normfile=normfile)

dataset_eva=AudioFeatureDataset(featrootdir=featrootdir, \
                                    cvtxtrootdir=cvtxtrootdir,feattype='npy', \
                                    cvnum=args.cvnum,mode='eva',normflag=0,\
                                    normfile=normfile)


dataset_test=AudioFeatureDataset(featrootdir=featrootdir, \
                                    cvtxtrootdir=cvtxtrootdir,feattype='npy', \
                                    cvnum=args.cvnum,mode='test',normflag=0,\
                                    normfile=normfile)


print("shuffling dataset_train")
train_loader=torch.utils.data.DataLoader(dataset_train, \
                                batch_size=args.batch_size,shuffle=False, \
                                num_workers=4,pin_memory=True)
print("shuffling dataset_eva")
eva_loader=torch.utils.data.DataLoader(dataset_eva, \
                                batch_size=args.batch_size,shuffle=False, \
                                num_workers=4,pin_memory=True)

print("shuffling dataset_test")
test_loader=torch.utils.data.DataLoader(dataset_test, \
                                batch_size=args.batch_size,shuffle=False, \
                                num_workers=4,pin_memory=False)


def sort_batch(data,label,length,name):
    batch_size=data.size(0)
#    print(np.argsort(length.numpy())[::-1].copy())
    inx=torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data=data[inx]
    label=label[inx]
    length=length[inx]
    name_new=[]
    for i in list(inx.numpy()):name_new.append(name[i])
    name=name_new
    length=list(length.numpy())
    return (data,label,length,name)

def writeCSV(temp,PATHCSV):
    with open(PATHCSV,'a',newline='') as f:
        writer=csv.writer(f,delimiter='\t')
        writer.writerows(temp)

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,biFlag,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag

        self.layer1=nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))
        if(biFlag):
                self.layer1.append(nn.LSTM(input_size=input_dim,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))


        # out = (len batch outdim)
        self.layer2=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
#            nn.LogSoftmax(dim=2)
        )

        self.simple_attention=nn.Linear(hidden_dim*self.bi_num,1,bias=False)

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda(),
                Variable(torch.zeros(1*self.num_layers,batch_size,self.hidden_dim)).cuda())
    
    def init_attention_weight(self,batch_size,maxlength):
        return Variable(torch.zeros(batch_size,maxlength)).cuda()
    
    def init_final_out(self,batch_size):
        return Variable(torch.zeros(batch_size,self.hidden_dim*self.bi_num)).cuda()
    
    def fixAttention(self):
        self.simple_attention.weight.requires_grad=False

    def fixlstm(self):
        self.layer2=nn.Sequential(
            nn.Linear(self.hidden_dim*self.bi_num,self.output_dim),
#            nn.LogSoftmax(dim=1),
        )
        self.simple_attention=nn.Linear(self.hidden_dim*self.bi_num,1,bias=False)
        self.cuda()
        for param in self.parameters():param.requires_grad=False

    def forward(self,x,length):
        batch_size=x.size(0)
        maxlength=int(np.max(length))
        hidden=[ self.init_hidden(batch_size) for l in range(self.bi_num)]
        weight=self.init_attention_weight(batch_size,maxlength)
        out_final=self.init_final_out(batch_size)

        out=[x,reverse_padded_sequence(x,length,batch_first=True)]
        for l in range(self.bi_num):
            out[l]=pack_padded_sequence(out[l],length,batch_first=True)
            out[l],hidden[l]=self.layer1[l](out[l],hidden[l])
            out[l],_=pad_packed_sequence(out[l],batch_first=True)
            if(l==1):out[l]=reverse_padded_sequence(out[l],length,batch_first=True)
        
        if(self.bi_num==1):out=out[0]
        else:out=torch.cat(out,2)

        potential=self.simple_attention(out)
        for inx,l in enumerate(length):weight[inx,0:l]=F.softmax(potential[inx,0:l],dim=0)
        for inx,l in enumerate(length):out_final[inx,:]=torch.matmul(weight[inx,:],torch.squeeze(out[inx,:,:]))

        out=self.layer2(out_final)
        out=torch.squeeze(out)
        return out,length,out_final

model=Net(**superParams)
model.fixlstm()
model.cuda()
model.load_state_dict(torch.load(args.loadpath))
loss_layer=torch.nn.BCEWithLogitsLoss(weight=None,size_average=False)


def test(testLoader):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data,target,length,name in testLoader:
        batch_size=data.size(0)
        max_length=torch.max(length)
        data=data[:,0:max_length,:];target=target[:,0:max_length]

        (data,target,length,name)=sort_batch(data,target,length,name)
        target=torch.FloatTensor(target.numpy())
        data,target=data.cuda(),target.cuda()
        data,target=Variable(data,volatile=True),Variable(target,volatile=True)

        output,_,out_final=model(data,length)
#        print(output)
        for i in range(batch_size):
            if(batch_size==1):
                temparray=torch.squeeze(out_final).cpu().data.numpy()
                result=torch.squeeze(F.sigmoid(output[0])).cpu().data.numpy()
                test_loss+=loss_layer(torch.squeeze(output[0]),torch.squeeze(target[i,0])).data[0]
            else:
                temparray=torch.squeeze(out_final[i,:]).cpu().data.numpy()
                result=torch.squeeze(F.sigmoid(output[i])).cpu().data.numpy()
                test_loss+=loss_layer(torch.squeeze(output[i]),torch.squeeze(target[i,0])).data[0]
            test_dict1[name[i]]=float(result)
            test_dict2[name[i]]=target.cpu().data[i][0]
            numframes+=length[i]
            temparray=np.append(temparray,result)
            np.save(os.path.join(args.savedir,name[i]),temparray)
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")
    
#    writeCSV(test_dict1.items(),os.path.join(args.savedir,'fold{}.csv'.format(args.cvnum)))

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        if(result>=0.5):label_pred.append(1)
        else:label_pred.append(0)
        label_true.append(test_dict2[filename])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/float(numframes), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %d %f\n"%(len(label_true),metrics.f1_score(label_true,label_pred,average="macro"),))
    return metrics.accuracy_score(label_true,label_pred),metrics.f1_score(label_true,label_pred,average="macro")

test(train_loader)
test(eva_loader)
test(test_loader)
