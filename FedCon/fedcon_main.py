import numpy as np
import torch
import pandas as pd
from fedcon_learner import Metanet,model_name
from fedcon_client import dataset_name
import time
import os
import warnings
from bayes_opt import BayesianOptimization

import hyperopt
from hyperopt import hp, fmin, tpe, Trials, partial

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
start = time.time()

def main(source_class,target_class,thres,df, max_df, malicious_rate,the_cycle_name,df_save_path,alpha):
    epoch =300
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
    max_acc = 0
    max_acc_ASR = 1

    meta_net = Metanet(device=device,threshold=thres, source_class=source_class, malicious_rate=malicious_rate, target_class=target_class,alpha=alpha)
    for i in range(epoch):
        time_start = time.time()
        print("{} round training.".format(i + 1))
        Acc, ASR = meta_net.meta_training(i + 1)
        df.loc[len(df)] = [the_cycle_name,alpha,thres, malicious_rate, Acc, ASR]
        df.to_csv(df_save_path+'/'+dataset_name+'_'+str(alpha)+'_result.csv',mode='a', index=False)
        print('running time:', time.time() - time_start, 's')
        if Acc > max_acc:
            max_acc = Acc
            max_acc_ASR = ASR
    max_df.loc[len(max_df)] = [the_cycle_name,alpha,thres,malicious_rate,max_acc, max_acc_ASR]
    max_df.to_csv(df_save_path+'/'+dataset_name+'_'+str(alpha)+'_result_max.csv',mode='a', index=False)
    del meta_net
    return max_acc


def bayesopt_objective(bate):
    temp_alpha = 50
    df = pd.DataFrame(columns=['the_cycle', 'alpha', 'thres', 'malicious_rate', 'Acc', 'ASR'])
    max_df = pd.DataFrame(columns=['the_cycle', 'alpha', 'thres', 'malicious_rate', 'Acc', 'ASR'])
    print('the thres value is {}'.format(bate))
    max_acc = main(source_class=9, target_class=7, thres=bate, the_cycle_name='grid_search',
    df=df, malicious_rate=0.4, max_df=max_df, df_save_path='./Gaussian_search', alpha=temp_alpha)
    return max_acc


def hyperopt_objective(bate_dict):
    bate = bate_dict['bate']
    temp_alpha = 0.5
    df = pd.DataFrame(columns=['the_cycle', 'alpha', 'thres', 'malicious_rate', 'Acc', 'ASR'])
    max_df = pd.DataFrame(columns=['the_cycle', 'alpha', 'thres', 'malicious_rate', 'Acc', 'ASR'])
    print('the thres value is {}'.format(bate))
    max_acc = main(source_class=9, target_class=7, thres=bate, the_cycle_name='TPE_search',
                   df=df, malicious_rate=0.4, max_df=max_df, df_save_path='./TPE_search', alpha=temp_alpha)
    return -max_acc


def Guassian_serach():
    param_grid_simple = {'bate': (0, 1)}
    opt = BayesianOptimization(bayesopt_objective
                               , param_grid_simple
                               , random_state=1
                               )
    opt.maximize(init_points=4, n_iter=7)
    params_best = opt.max["params"]
    score_best = opt.max["target"]
    print('best score is {} and best score is {}'.format(score_best,params_best))

def grid_search():
    df = pd.DataFrame(columns=['the_cycle','alpha','thres', 'malicious_rate', 'Acc', 'ASR'])
    max_df = pd.DataFrame(columns=['the_cycle','alpha','thres','malicious_rate', 'Acc', 'ASR'])
    for temp_alpha in ([0.5,50,5,0]):
        for i in range(0,11):
            print('the thres value is {}'.format(i/10))
            main(source_class=9, target_class=7, thres=i/10, the_cycle_name='grid_search',
                df=df, malicious_rate=0.4, max_df=max_df,df_save_path='./grid_search',alpha=temp_alpha)

def TPE_search():
    param_grid_simple = {'bate': hp.uniform("bate", 0.0, 1.0)}
    trials = Trials()
    params_best = fmin(hyperopt_objective# 目标函数
                       , space=param_grid_simple  # 参数空间
                       , algo=tpe.suggest  # 代理模型你要哪个呢？
                       , max_evals=11  # 允许的迭代次数
                       , verbose=False
                       , trials=trials)

    # 打印最优参数，fmin会自动打印最佳分数
    print("\n", "\n", "best params: ", params_best,
          "\n")

if __name__ == '__main__':
    # grid_search()
    ##Gaussian search
    Guassian_serach()

    ##TPE_search
    # TPE_search()

