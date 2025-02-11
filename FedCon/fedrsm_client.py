import torch
import numpy as np
from torch import nn
import torchvision
import math
# from torch.nn import functional as F
from torchvision import datasets, transforms
from copy import deepcopy
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
from collections import Counter
import random
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class_num = 10
model_name = 'lenet'
dataset_name = 'trafficsign'# sfddd or mnist or trafficsign
malicious_rate=0.4
# LeNet
if model_name == 'lenet':
    LAYER_FEA = {'layer3': 'feat3'}
    HOOK_RES = ['feat3']


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


class Client(nn.Module):
    def __init__(self, model, id, train_path,class_num_list, update_step, update_step_test, base_lr, meta_lr, device, mode,
                 batch_size,client_type,source_class,target_class):
        super(Client, self).__init__()
        self.id = id


        self.class_num_list=class_num_list
        self.source=source_class
        self.target=target_class

        self.update_step = update_step  ## task-level inner update steps
        self.update_step_test = update_step_test
        self.net = deepcopy(model)
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.client_type=client_type
        self.RDV = []
        self.last_round_RC = []
        self.never_selected = 1
        self.RC = []
        self.upgrade_bool = []
        self.dataset_length=1000
        train_data=self.get_train_data(train_path,self.class_num_list,self.source,self.target)





        #test_data = self.get_test_data(test_path,self.source,self.target,self.dataset_class_num,500)
        np.random.shuffle(train_data)
        #np.random.shuffle(test_data)


        if self.client_type=='malicious':
            for i in range(len(train_data)):
                train_data[i]=list(train_data[i])
                if train_data[i][1]==self.source:
                    train_data[i][1] = self.target
             #   elif train_data[i][1] == 4:
              #      train_data[i][1] = 8

        self.mode = mode
        self.time = 0
        self.epoch = 0
        support_size = int(len(train_data) * 1.0)
        support_set = DatasetSplit(train_data[:support_size])
        self.support_loader = DataLoader(
            support_set, batch_size=128, shuffle=True, drop_last=True)

        '''
        query_size = int(len(test_data) * 1.0)
        query_set = DatasetSplit(test_data[:query_size])
        self.query_loader = DataLoader(
            query_set, batch_size=100, shuffle=True, drop_last=True)
        '''
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.base_lr)
        self.outer_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.batch_size = batch_size
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        # self.loss_function = torch.nn.MSELoss().to(self.device)
      #  self.stimulus_data = list(stimulus_data)

    #    self.stimulus_x, self.stimulus_y = self.transform_stimulus(class_num)

    def forward(self):
        pass

    def get_train_data(self,path,class_num_list,source,target):
        each_class_num = list(class_num_list)
        for i in range(len(each_class_num)):
            if (i==source)&(self.client_type=='malicious')&(each_class_num[i]<=200):
                each_class_num[i]=500
        '''
        classes_all=[i for i in range(10)]
        classes=[]
        classes.append(source)
        classes_all.remove(source)
        #classes.append(target)
        #classes_all.remove(target)
        
        for i in range(num_class-1):
            temp_class=random.sample(classes_all,1)
            classes.append(temp_class[0])
            classes_all.remove(temp_class[0])
        '''

        #classes=[4,8]
        train_data = torch.load(path)
        train_data_x=[]
        train_data_y=[]
        dataset_x = []
        dataset_y = []

        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0])
            train_data_y.append(train_data[i][1])

        for i in range(len(each_class_num)):
            index_range = np.argwhere(np.array(train_data_y) == i)
            idx_local=random.sample(list(index_range), int(each_class_num[i]))
            for idx_now in idx_local:
                dataset_x.append(train_data_x[int(idx_now)])
                dataset_y.append(train_data_y[int(idx_now)])
        print(Counter(dataset_y))
        train_dataset = [t for t in zip(dataset_x, dataset_y)]
        return train_dataset

    def get_test_data(self,path,source,target,num_class,dataset_length):
        each_class_num = [int(dataset_length / num_class) for i in range(num_class)]
        classes_all=[i for i in range(10)]
        classes=[]
        classes.append(source)
        classes_all.remove(source)
        classes.append(target)
        classes_all.remove(target)
        if num_class>2:
            for i in range(1,num_class-1):
                classes.append(classes_all[-1*i])


        #classes=[4,8]
        train_data = torch.load(path)
        train_data_x=[]
        train_data_y=[]
        dataset_x = []
        dataset_y = []
        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0])
            train_data_y.append(train_data[i][1])
        for i in range(len(classes)):
            index_range = np.argwhere(np.array(train_data_y) == classes[i])
            idx_local=random.sample(list(index_range), each_class_num[i])
            for idx_now in idx_local:
                dataset_x.append(train_data_x[int(idx_now)])
                dataset_y.append(train_data_y[int(idx_now)])

        train_dataset = [t for t in zip(dataset_x, dataset_y)]
        return train_dataset


    def local_fed_train(self):
        for _ in range(self.update_step):
            self.global_net = deepcopy(self.net)
            self.local_fea_out = []
            self.global_fea_out = []
            # net_tem = deepcopy(self.net)
            # meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 0
            for support in self.support_loader:
                support_x, support_y = support

                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                # for batch_idx, support_x in enumerate(support_x):
                # support_x = support_x.reshape(1,1,28,28)

                # torch.save(support_x,'support_x.pt')

                output = self.net(support_x)
                # output = torch.squeeze(output)
                loss = self.loss_function(output, support_y)

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                i += 1




    def generate_E(self, E_number):
        i = 0
        E_list = []
        while i < E_number:
            E_element = [random.randint(0, 43), random.randint(0, 43)]
            if E_list.__contains__(E_element):
                continue
            else:
                E_list.append(E_element)
                i = i + 1
        return E_list
        # 本地模型hook


    def refresh(self, model):
        for w, w_t in zip(self.net.parameters(), model.parameters()):
            w.data.copy_(w_t.data)

    def test(self):
        source,mislabel=0.0,0.0
        loss_all, correct_all, total = 0.0, 0.0, 0.0
        precision_all, recall_all, f1_all = 0.0, 0.0, 0.0
        for query in self.query_loader:  # ---------------------------------------init
            query_x, query_y = query


            for i in range(len(query_y)):
                if query_y[i]==self.source:
                    source=source+1


            if torch.cuda.is_available():
                query_x = query_x.cuda(self.device)
                query_y = query_y.cuda(self.device)
            output = self.net(query_x)
            output = torch.squeeze(output)
            loss = self.loss_function(output, query_y)
            loss_all += loss.item()
            total += len(query_y)

            y_hat = self.net(query_x)
            query_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)

            for i in range(len(query_y)):
                if (query_y[i]==self.source)&(query_pred[i]==self.target):
                    mislabel=mislabel+1


            correct = torch.eq(query_pred, query_y).sum().item()
            correct_all += correct
            # acc = correct/len(query_x)
            # ------------------------------------------------------------------
            # acc_sk = accuracy_score(query_pred, query_y)
            query_y_c = query_y.cpu().numpy()  # ----
            query_pred_c = query_pred.cpu().numpy()  # ----
            precision = precision_score(query_pred_c, query_y_c, average='weighted')
            recall = recall_score(query_pred_c, query_y_c, average='weighted')
            f1 = f1_score(query_pred_c, query_y_c, average='weighted')
            precision_all += precision
            recall_all += recall
            f1_all += f1

        init_loss_list = loss_all / len(self.query_loader)
        init_acc_list = correct_all / total
        init_pre_list = precision_all / len(self.query_loader)
        init_recall_list = recall_all / len(self.query_loader)
        init_f1_list = f1_all / len(self.query_loader)
        ASR=mislabel/source

        return init_loss_list, init_acc_list, init_pre_list, init_recall_list, init_f1_list,ASR








