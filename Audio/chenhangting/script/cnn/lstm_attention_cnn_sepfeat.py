# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/1
@author: chenhangting

@notes: a attention-lstm  for MOSI
    support early stopping
    add cnn for fbank
    concatenate other features
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np
import sys
sys.path.append(r'../dataset')
from dataset1d_early_stopping_single_label_fbank_others import AudioFeatureDataset
import pdb
import os
from sklearn import metrics



# Training setttings
parser=argparse.ArgumentParser(description='PyTorch for audio sentiment classification in MOSI')
parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                    help='the num of cv set')
parser.add_argument('--batch_size',type=int,default=12,metavar='N', \
                    help='input batch size for training ( default 16 )')
parser.add_argument('--epoch',type=int,default=150,metavar='N', \
                    help='number of epochs to train ( default 100)')
parser.add_argument('--lr',type=float,default=0.001,metavar='LR', \
                    help='inital learning rate (default 0.001 )')
parser.add_argument('--seed',type=int,default=1,metavar='S', \
                    help='random seed ( default 1 )')
parser.add_argument('--log_interval',type=int,default=1,metavar='N', \
                    help='how many batches to wait before logging (default 10 )')
parser.add_argument('--device_id',type=int,default=0,metavar='N', \
                    help="the device id")
parser.add_argument('--savepath',type=str,default='./model.pkl',metavar='S', \
                    help='save model in the path')


args=parser.parse_args()
os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

emotion_labels=('positive','negative',)
superParams={'input_dim1':40,
            'input_dim2':33,
            'input_channels':3,
            'dimAfterCov':120,
            'hidden_dim':256,
            'output_dim':len(emotion_labels),
            'num_layers':4,
            'biFlag':2,
            'dropout':0.25}

args.cuda=torch.cuda.is_available()
if(args.cuda==False):sys.exit("GPU is not available")
torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)

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

def sort_batch(data1,data2,label,length,name):
    batch_size=data1.size(0)
#    print(np.argsort(length.numpy())[::-1].copy())
    inx=torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data1=data1[inx];data2=data2[inx]
    label=label[inx]
    length=length[inx]
    name_new=[]
    for i in list(inx.numpy()):name_new.append(name[i])
    name=name_new
    length=list(length.numpy())
    return (data1,data2,label,length,name)

class Net(nn.Module):
    def __init__(self,input_dim1,input_dim2,input_channels,dimAfterCov,hidden_dim,output_dim,num_layers,biFlag,dropout=0.5):
        #dropout
        super(Net,self).__init__()
        self.input_dim1=input_dim1
        self.input_dim2=input_dim2
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        self.input_channels=input_channels
        self.dimAfterCov=dimAfterCov
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag

        self.cov1=nn.Sequential(
            nn.Conv1d(self.input_channels,6,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

        self.cov2=nn.Sequential(
            nn.Conv1d(8,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

        self.layer1=nn.LSTM(input_size=self.dimAfterCov+self.input_dim2,hidden_size=hidden_dim,  \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=biFlag)
        # out = (len batch outdim)
        self.layer2=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.LogSoftmax(dim=1)
        )

        self.simple_attention=nn.Linear(hidden_dim*self.bi_num,1,bias=False)

    def init_hidden(self,batch_size):
        return (Variable(torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim)).cuda())
    
    def init_attention_weight(self,batch_size,maxlength):
        return Variable(torch.zeros(batch_size,maxlength)).cuda()
    
    def init_final_out(self,batch_size):
        return Variable(torch.zeros(batch_size,self.hidden_dim*self.bi_num)).cuda()

    def forward(self,x1,x2,length):
        batch_size=x1.size(0)
        maxlength=int(np.max(length))
        hidden=self.init_hidden(batch_size)
        weight=self.init_attention_weight(batch_size,maxlength)
        out_final=self.init_final_out(batch_size)

        x1=x1.view(-1,self.input_channels,self.input_dim1)
        out=self.cov1(x1)
#        out=self.cov2(out)
        out=out.view(batch_size,maxlength,self.dimAfterCov)

        out=torch.cat((out,x2),dim=2)
        out=pack_padded_sequence(out,length,batch_first=True)
        out,hidden=self.layer1(out,hidden)
        out,length=pad_packed_sequence(out,batch_first=True)
        potential=self.simple_attention(out)
#        print(potential);print(weight)
        for inx,l in enumerate(length):weight[inx,0:l]=F.softmax(potential[inx,0:l],dim=0)
        for inx,l in enumerate(length):out_final[inx,:]=torch.matmul(weight[inx,:],torch.squeeze(out[inx,:,:]))
        
        out_final=self.layer2(out_final)
        return out_final,length

model=Net(**superParams)
model.cuda()
optimizer=optim.Adam(model.parameters(),lr=args.lr,weight_decay=0.00001)

def train(epoch,trainLoader):
    model.train()
    for batch_inx,(data_fbank,data_others,target,length,name) in enumerate(trainLoader):
        batch_size=data_fbank.size(0)
        max_length=torch.max(length)
        data_fbank=data_fbank[:,0:max_length,:,:]
        data_others=data_others[:,0:max_length,:]

        (data_fbank,data_others,target,length,name)=sort_batch(data_fbank,data_others,target,length,name)
        data_fbank,data_others,target=data_fbank.cuda(),data_others.cuda(),target.cuda()
        data_fbank,data_others,target=Variable(data_fbank),Variable(data_others),Variable(target)

        optimizer.zero_grad()
        output,_=model(data_fbank,data_others,length)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_inx % args.log_interval==0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_inx * batch_size, len(trainLoader.dataset),
                100. * batch_inx / len(trainLoader), loss.data[0]))


def test(testLoader):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data_fbank,data_others,target,length,name in testLoader:
        batch_size=data_fbank.size(0)
        max_length=torch.max(length)
        data_fbank=data_fbank[:,0:max_length,:,:]
        data_others=data_others[:,0:max_length,:]

        (data_fbank,data_others,target,length,name)=sort_batch(data_fbank,data_others,target,length,name)
        data_fbank,data_others,target=data_fbank.cuda(),data_others.cuda(),target.cuda()
        data_fbank,data_others,target=Variable(data_fbank,volatile=True),Variable(data_others,volatile=True),Variable(target,volatile=True)

        output,_=model(data_fbank,data_others,length)
        test_loss+=F.nll_loss(output,target,size_average=False).data[0]
        for i in range(batch_size):
            result=torch.squeeze(output[i,:]).cpu().data.numpy()
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i]
            numframes+=length[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        label_true.append(test_dict2[filename]);label_pred.append(np.argmax(result))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/float(numframes), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %f"%metrics.f1_score(label_true,label_pred,average="macro"))
    return metrics.accuracy_score(label_true,label_pred),metrics.f1_score(label_true,label_pred,average="macro")

def early_stopping(network,savepath,metricsInEpochs,gap=10):
    best_metric_inx=np.argmax(metricsInEpochs)
    if(best_metric_inx==len(metricsInEpochs)-1):
        torch.save(network.state_dict(),savepath)
        return False
    elif(len(metricsInEpochs)-best_metric_inx >= gap):
        return True
    else: 
        return False
    
eva_fscore_list=[]
for epoch in range(1,args.epoch+1):
    train(epoch,train_loader)
    eva_acc,eva_fscore=test(eva_loader)
    eva_fscore_list.append(eva_fscore)
    if(early_stopping(model,args.savepath,eva_fscore_list,gap=15)):break

model.load_state_dict(torch.load(args.savepath))
model=model.cuda()
test(test_loader)
