import math
from scipy import spatial
import torch
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import os
import heapq
from copy import deepcopy
from fedcon_client import Client, model_name, dataset_name
from collections import Counter
from models.vision import LeNet, ResNet, ResNet18, weights_init, ConvNet,LeNet_TS
import random
import torchvision
from torch.utils.data import DataLoader, Dataset
import random
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





# instantiation
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)


class Metanet(nn.Module):
    #def __init__(self,threshold, device, local_metatrain_epoch=3, local_test_epoch=3, outer_lr=0.001, inner_lr=0.001):
    def __init__(self, source_class,target_class,threshold, device,malicious_rate,alpha,local_metatrain_epoch=3, local_test_epoch=3, outer_lr=0.001,inner_lr=0.001):

        super(Metanet, self).__init__()
        self.final_acc=[]
        self.final_ASR=[]
        self.source=source_class
        self.target=target_class
        self.malicious_rate=malicious_rate
        self.alpha=alpha
        self.threshold=threshold
        self.belong_test_res=[]
        self.not_belong_test_res=[]
        self.device = device
        self.local_metatrain_epoch = local_metatrain_epoch
        self.local_test_epoch = local_test_epoch
        if dataset_name == 'mnist':
            self.net = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            self.net = LeNet_TS().to(self.device)
        elif dataset_name == 'sfddd':
            self.net = LeNet_TS().to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.clients = []
        self.RC_list = []
        self.mode_1 = "fed_train"
        self.mode_2 = "fed_test"
        self.batch_size = 20
        self.stimulus_each_class_num=1
        self.path_now = os.path.dirname(__file__)
        self.last_path = '/final_test'  # -----------------------------------------------------
        if dataset_name == 'trafficsign':
            train_path = r'.\data\GTSRB\trafficsign_train'
            self.test_data = torch.load(r'.\data\GTSRB\trafficsign_test/1.pt')
            np.random.shuffle(self.test_data)
            self.test_set = DatasetSplit(self.test_data)
            self.test_loader = DataLoader(
                self.test_set, batch_size=128, shuffle=True, drop_last=True)
            train_file_set = os.listdir(train_path)
            train_path_set = [os.path.join(train_path, i) for i in train_file_set]
        elif dataset_name == 'mnist':
            train_path = r'.\data\MNIST\MNIST_train'

            self.test_data = torch.load(r'.\data\MNIST\MNIST_test/1.pt')
            np.random.shuffle(self.test_data)
            self.test_set = DatasetSplit(self.test_data)
            self.test_loader = DataLoader(
                self.test_set, batch_size=1000, shuffle=True, drop_last=True)
            train_file_set = os.listdir(train_path)
            train_path_set = [os.path.join(train_path, i) for i in train_file_set]
        elif dataset_name == 'sfddd':
            train_path = r'.\data\SFDDD\aug_train_32.npy'
            self.test_data = torch.load(r'.\data\SFDDD\test_32.npy')
            np.random.shuffle(self.test_data)
            self.test_set = DatasetSplit(self.test_data)
            self.test_loader = DataLoader(
                self.test_set, batch_size=1000, shuffle=True, drop_last=True)
            train_path_set = [train_path for i in range(30)]

        #test_file_set = os.listdir(test_path)
        #test_path_set = [os.path.join(test_path, i) for i in test_file_set]
        self.time_accum = [0]


        if dataset_name == 'mnist':
            model = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            model = LeNet_TS().to(self.device)
        elif dataset_name == 'sfddd':
            model = LeNet_TS().to(self.device)

        self.stimulus_x,self.stimulus_y=self.prepare_stimulus_LFA(self.stimulus_each_class_num)


        self.num_clients = 30

        if self.alpha!=0:
            self.class_num_all=np.load(r'.\diri_distribution/alpha'+str(self.alpha)+'.npy')
        else:
            self.class_num_all=[]
            for i in range(self.num_clients):
                self.class_num_all.append([500 for i in range(10)])



        self.malicious_client_num=int(self.num_clients*malicious_rate)
        #self.malicious_client_id=random.sample([i for i in range(self.num_clients)],self.malicious_client_num)
        self.malicious_client_id=[i for i in range(self.malicious_client_num)]
        print("malicious client: ")
        print(self.malicious_client_id)
        for i in range(self.num_clients):
            if self.malicious_client_id.__contains__(i):
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           update_step=local_metatrain_epoch, update_step_test=local_test_epoch,
                           base_lr=inner_lr, meta_lr=outer_lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='malicious', source_class=self.source,
                           target_class=self.target))
            else:
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           update_step=local_metatrain_epoch, update_step_test=local_test_epoch,
                           base_lr=inner_lr, meta_lr=outer_lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='benign', source_class=self.source,
                           target_class=self.target))
        print(1)





    def prepare_stimulus_LFA(self,each_num):
        if dataset_name=='mnist':
            stimulus_data_ori = torch.load(r'.\data\MNIST\MNIST_stimulus/1.pt')
        elif dataset_name=='trafficsign':
            stimulus_data_ori = torch.load(r'.\data\GTSRB\trafficsign_stimulus/1.pt')
        elif dataset_name=='sfddd':
            stimulus_data_ori = torch.load(r'.\data\SFDDD\stimulus_32.npy')
        stimulus_x=[]
        stimulus_y = []
        categorize_flag=[]
        classes_all = [i for i in range(10)]
        stimulus_list = [i for i in range(10)]

        #stimulus_list=[]
        #stimulus_list.append(self.source)
        #stimulus_list.append(self.target)
        '''
        classes_all.remove(self.source)
        classes_all.remove(self.target)
        stimulus_list=random.sample(classes_all,2)
        stimulus_list.append(self.source)
        stimulus_list.append(self.target)
        '''
        for single_class in stimulus_list:
            categorize_flag.append([0,single_class])

        for i in range(len(stimulus_data_ori)):
            for class_flag in categorize_flag:
                if (stimulus_data_ori[i][1]==class_flag[1])&(class_flag[0]<each_num):
                    if len(stimulus_x) == 0:
                        stimulus_x = stimulus_data_ori[i][0]
                        stimulus_y.append(stimulus_data_ori[i][1])
                    else:
                        stimulus_x = torch.cat([stimulus_x, stimulus_data_ori[i][0]], 0)
                        stimulus_y.append(stimulus_data_ori[i][1])

                    class_flag[0] = class_flag[0] + 1

        if dataset_name=='mnist':
            stimulus_x = torch.reshape(stimulus_x, (each_num*10, 1, 28, 28))
        elif dataset_name=='trafficsign':
            stimulus_x = torch.reshape(stimulus_x, (each_num * 10, 3, 32, 32))
        elif dataset_name=='sfddd':
            stimulus_x = torch.reshape(stimulus_x, (each_num * 10, 3, 32, 32))
        stimulus_y = torch.tensor(stimulus_y)

        if torch.cuda.is_available():
            stimulus_x = stimulus_x.cuda(self.device)
            stimulus_y = stimulus_y.cuda(self.device)
        return stimulus_x, stimulus_y



    def save_time(self, save_path):
        dataframe = pd.DataFrame(list(self.time_accum), columns=['time_accum'])
        dataframe.to_excel(save_path, index=False)

    def calculate_RC(self,model1,model2,stimulus_num):

        temp_model1 = deepcopy(model1)
        temp_model2 = deepcopy(model2)
        Layer_fea = {'layer3': 'feat3'}
        model1_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_model1, Layer_fea)
        model2_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_model2, Layer_fea)
        #model1_stimulus_out = model1_Layer_Getter(self.stimulus_x)
        #model2_stimulus_out = model2_Layer_Getter(self.stimulus_x)
        #model1_out = model1_stimulus_out['feat3']
        #model2_out = model2_stimulus_out['feat3']
        model1_out = temp_model1(self.stimulus_x)
        model2_out = temp_model2(self.stimulus_x)
        model1_softmax=torch.tensor(F.softmax(model1_out, dim=1),dtype=torch.float32)
        model2_softmax=torch.tensor(F.softmax(model2_out, dim=1),dtype=torch.float32)
        model1_RDM = torch.zeros((stimulus_num, stimulus_num)).to(self.device)
        model2_RDM = torch.zeros((stimulus_num, stimulus_num)).to(self.device)

        for i in range(stimulus_num):
            for j in range(stimulus_num):

                model1_RDM[i][j] = torch.cosine_similarity(model1_softmax[i].view(1, -1),model1_softmax[j].view(1,-1))
                model2_RDM[i][j] = torch.cosine_similarity(model2_softmax[i].view(1, -1), model2_softmax[j].view(1, -1))
        corr_result=self.corr2(model1_RDM, model2_RDM)
        return corr_result

            # 计算上传概率

    def mean2(self, x):
            y = torch.sum(x) / len(x)
            return y

    def corr2(self, a, b):
            a = a - self.mean2(a)
            b = b - self.mean2(b)
            r = torch.sum(a * b) /torch.sqrt(torch.sum(a*a)*torch.sum(b*b))
            return r

    def meta_training(self, round):


        temp_RC = []
        id_train_0 = list(range(len(self.clients)))
        id_train = random.sample(id_train_0, int(len(id_train_0) * 1))  # clients of this round
        time_list_tr = []  # time list this round




        for id, j in enumerate(id_train):
            time_list_tr.append(random.uniform(3, 20))
            self.clients[j].refresh(self.net)
            self.clients[j].local_fed_train()
            self.clients[j].epoch = round

        time_tr = max(time_list_tr)

        # self.num_c_per_r.append(len(id_train))
        self.time_accum.append(self.time_accum[-1] + time_tr)

        weight = []
        model_list=[]
        vote_list = []

        for i in range(len(id_train)):
            model_list.append([self.clients[i].net,self.clients[i].id])
            vote_list.append([0,self.clients[i].id])
        #计算RC矩阵
        RC_matrix=np.zeros((len(model_list),len(model_list)))
        for i in range(len(model_list)):
            for j in range(len(model_list)):
                RC_matrix[i][j]=self.calculate_RC(model_list[i][0],model_list[j][0],self.stimulus_each_class_num*10)

        RC_matrix=(RC_matrix-RC_matrix.min())/(RC_matrix.max()-RC_matrix.min())


        for i in range(len(RC_matrix)):
            for j in range(len(RC_matrix)):
                if RC_matrix[i][j]>=self.threshold:
                    vote_list[i][0]=vote_list[i][0]+1
        remove_list=[]

        for i in range(len(vote_list)):
            if vote_list[i][0]<self.num_clients/2:
                id_train.remove(vote_list[i][1])
                remove_list.append(vote_list[i][1])

        print('malicious_client: ')
        print(remove_list)
        #if round<=5:
        #id_train=id_train_0
        if len(id_train)==0:
            id_train=id_train_0


        for id, j in enumerate(id_train):
            # weight.append(self.clients[j].size / size_all)
            weight.append(1/len(id_train))

        weight = np.array(weight)


        # *************************************************************************************************************

        for id, j in enumerate(id_train):
            for global_param, local_param in zip(self.net.parameters(), self.clients[j].net.parameters()):
                if (global_param is None or id == 0):
                    param_tem = Variable(torch.zeros_like(global_param)).to(self.device)
                    global_param.data.copy_(param_tem.data)
                if local_param is None:
                    local_param = Variable(torch.zeros_like(global_param)).to(self.device)
                global_param.data.add_(local_param.data * weight[id])

        print("聚合后全局模型测试结果：")
        acc,ASR = self.meta_test(self.net,self.test_loader)
        self.final_acc.append(acc)
        self.final_ASR.append(ASR)

        return acc,ASR



    def meta_test(self,net,data_loader):
        test_net=deepcopy(net)
        source,mislabel=0.0,0.0
        loss_all, correct_all, total = 0.0, 0.0, 0.0

        for test in data_loader:  # ---------------------------------------init
            test_x, test_y = test
            for i in range(len(test_y)):
                if test_y[i]==self.source:
                    source=source+1
            if torch.cuda.is_available():
                test_x = test_x.cuda(self.device)
                test_y = test_y.cuda(self.device)

            total += len(test_y)
            y_hat = test_net(test_x)
            test_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            for j in range(len(test_y)):
                if (test_y[j]==self.source)&(test_pred[j]==self.target):
                    mislabel=mislabel+1
            correct = torch.eq(test_pred, test_y).sum().item()
            correct_all += correct

        acc = correct_all / total
        if source!=0:
            ASR=mislabel/source
        else:
            ASR=0
        print("acc:   "+str(acc))

        print("ASR:   "+str(ASR))
        return acc,ASR
