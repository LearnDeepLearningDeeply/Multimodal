# -*- coding: utf-8 -*-
"""
@date: Created on 2018/5/7
@author: chenhangting

@notes: a cnn-attention-lstm  for MOSI
    support early stopping
    it is a script for pretrain attention cnn-lstm
    the reason to do this is that the directly trained attention-lstm performs very bad
    logistic replace softmax
    reproducable and statibility for bilstm with dropout
    TODO this script has some problem in Net.foward
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
from dataset1d_early_stopping_fbank_others import AudioFeatureDataset
import pdb
import os
from sklearn import metrics



# Training setttings
parser=argparse.ArgumentParser(description='PyTorch for audio sentiment classification in MOSI')
parser.add_argument('--cvnum',type=int,default=1,metavar='N', \
                    help='the num of cv set')
parser.add_argument('--batch_size',type=int,default=8,metavar='N', \
                    help='input batch size for training ( default 16 )')
parser.add_argument('--epoch',type=int,default=150,metavar='N', \
                    help='number of epochs to train ( default 150)')
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
parser.add_argument('--loadpath',type=str,default='./model.pkl',metavar='S', \
                    help='load model in the path')


args=parser.parse_args()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device_id)

emotion_labels=('positive','negative',)
superParams={'input_dim1':40,
            'input_dim2':33,
            'input_channels':3,
            'dimAfterCov':120,
            'hidden_dim':128,
            'output_dim':1,
            'num_layers':2,
            'biFlag':2,
            'dropout':0.5}

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

        self.layer1=nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=dimAfterCov+input_dim2,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))
        if(biFlag):
                self.layer1.append(nn.LSTM(input_size=dimAfterCov+input_dim2,hidden_size=hidden_dim, \
                        num_layers=num_layers,batch_first=True, \
                        dropout=dropout,bidirectional=0))

        self.cov1=nn.Sequential(
            nn.Conv1d(self.input_channels,6,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

        self.cov2=nn.Sequential(
            nn.Conv1d(6,12,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=dropout),
        )

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
        for param in self.parameters():param.requires_grad=False
        self.layer2=nn.Sequential(
            nn.Linear(self.hidden_dim*self.bi_num,self.output_dim),
#            nn.LogSoftmax(dim=1),
        )
        self.simple_attention=nn.Linear(self.hidden_dim*self.bi_num,1,bias=False)
        self.cuda()

    def forward(self,x1,x2,length):
        batch_size=x1.size(0)
        maxlength=int(np.max(length))
        hidden=[ self.init_hidden(batch_size) for l in range(self.bi_num)]
        weight=self.init_attention_weight(batch_size,maxlength)
        out_final=self.init_final_out(batch_size)

        x1=x1.view(-1,self.input_channels,self.input_dim1)
        out=self.cov1(x1)
        out=self.cov2(out)
        out=out.view(batch_size,maxlength,self.dimAfterCov)
        out=torch.cat((out,x2),dim=2)

        out=[out,reverse_padded_sequence(out,length,batch_first=True)]
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
        
        out=self.layer2(out)
        out=torch.squeeze(out)
        return out,length

model=Net(**superParams)
model.cuda()
model.load_state_dict(torch.load(args.loadpath))
model.fixlstm()
optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr,weight_decay=0.0001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
loss_layer=torch.nn.BCEWithLogitsLoss(weight=None,size_average=False)

def train(epoch,trainLoader):
    model.train();exp_lr_scheduler.step()
    for batch_inx,(data_fbank,data_others,target,length,name) in enumerate(trainLoader):
        batch_size=data_fbank.size(0)
        max_length=torch.max(length)
        data_fbank=data_fbank[:,0:max_length,:,:]
        data_others=data_others[:,0:max_length,:]
        target=target[:,0:max_length]

        (data_fbank,data_others,target,length,name)=sort_batch(data_fbank,data_others,target,length,name)
        target=torch.FloatTensor(target.numpy())
        data_fbank,data_others,target=data_fbank.cuda(),data_others.cuda(),target.cuda()
        data_fbank,data_others,target=Variable(data_fbank),Variable(data_others),Variable(target)

        optimizer.zero_grad()
        output,_=model(data_fbank,data_others,length)
#        print(output)       
        numframes=0
        for i in range(batch_size):
            label=int(torch.squeeze(target[i,0]).cpu().data)
            if(i==0):loss=loss_layer(torch.squeeze(output[i,0:length[i]]),torch.squeeze(target[i,0:length[i]]))
            else:loss+=loss_layer(torch.squeeze(output[i,0:length[i]]),torch.squeeze(target[i,0:length[i]]))
            numframes+=length[i]
#            print(target[i,0:length[i]]) 

        loss.backward()
#        nn.utils.clip_grad_norm(filter(lambda p:p.requires_grad,model.parameters()),max_norm=10.0)

        weight_loss=0.0;grad_total=0.0;param_num=0
        for group in optimizer.param_groups:
            if(group['weight_decay']!=0):
                for p in group['params']:
                    if(p.grad is None):continue
                    w1=p.grad.data.cpu().numpy()
                    w2=p.data.cpu().numpy()
                    if(len(w1.shape)>2 or len(w1.shape)==1):w1=w1.reshape(w1.shape[0],-1)
                    if(len(w2.shape)>2 or len(w2.shape)==1):w2=w2.reshape(w2.shape[0],-1)
                    if(len(w1.shape)==1):param_num+=w1.shape[0]
                    else:param_num+=w1.shape[0]*w1.shape[1]
                    weight_loss+=group['weight_decay']*np.linalg.norm(w2,ord='fro')
                    grad_total+=np.linalg.norm(w1,ord='fro')
#        weight_loss/=float(param_num);grad_total/=float(param_num)

        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAve loss: {:.6f} and Total weight loss {:.6f} and Total grad fro-norm {:.6f}'.format(
                epoch, batch_inx * batch_size, len(trainLoader.dataset),
                100. * batch_inx / len(trainLoader), loss.data[0]/float(numframes),weight_loss,grad_total))


def test(testLoader):
    model.eval()
    test_loss=0;numframes=0
    test_dict1={};test_dict2={}

    for data_fbank,data_others,target,length,name in testLoader:
        batch_size=data_fbank.size(0)
        max_length=torch.max(length)
        data_fbank=data_fbank[:,0:max_length,:,:]
        data_others=data_others[:,0:max_length,:]
        target=target[:,0:max_length]

        (data_fbank,data_others,target,length,name)=sort_batch(data_fbank,data_others,target,length,name)
        target=torch.FloatTensor(target.numpy())
        data_fbank,data_others,target=data_fbank.cuda(),data_others.cuda(),target.cuda()
        data_fbank,data_others,target=Variable(data_fbank,volatile=True),Variable(data_others,volatile=True),Variable(target,volatile=True)

        output,_=model(data_fbank,data_others,length)
#        print(output)
        for i in range(batch_size):
            if(batch_size==1):
                result=np.log(torch.squeeze(F.sigmoid(output[0:length[i]])).cpu().data.numpy()).mean(axis=0)
                test_loss+=loss_layer(torch.squeeze(output[0:length[i]]),torch.squeeze(target[i,0:length[i]])).data[0]
            else:
                result=np.log((torch.squeeze(F.sigmoid(output[i,0:length[i]])).cpu().data.numpy())).mean(axis=0)
                test_loss+=loss_layer(torch.squeeze(output[i,0:length[i]]),torch.squeeze(target[i,0:length[i]])).data[0]
            test_dict1[name[i]]=result
            test_dict2[name[i]]=target.cpu().data[i][0]
            numframes+=length[i]
    if(len(test_dict1)!=len(testLoader.dataset)):
        sys.exit("some test samples are missing")

    label_true=[];label_pred=[]
    for filename,result in test_dict1.items():
#        print(test_dict2[filename])
#        print(np.argmax(result)==test_dict2[filename])
        if(result>=np.log(0.5)):label_pred.append(1)
        else:label_pred.append(0)
        label_true.append(test_dict2[filename])
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss/float(numframes), metrics.accuracy_score(label_true,label_pred,normalize=False), \
        len(test_dict1),metrics.accuracy_score(label_true,label_pred)))
    print(metrics.confusion_matrix(label_true,label_pred))
    print("macro f-score %d %f\n"%(len(label_true),metrics.f1_score(label_true,label_pred,average="macro"),))
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
test(test_loader)
#test(test_loader)
for epoch in range(1,args.epoch+1):
    train(epoch,train_loader)
    eva_acc2,eva_fscore2=test(train_loader)
    eva_acc1,eva_fscore1=test(eva_loader)
    eva_fscore_list.append((eva_fscore1+eva_fscore2)/2.0)
    if(early_stopping(model,args.savepath,eva_fscore_list,gap=15)):break

model.load_state_dict(torch.load(args.savepath))
model=model.cuda()
test(test_loader)
