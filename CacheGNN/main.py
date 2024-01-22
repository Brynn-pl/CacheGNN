#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model_star import *

import os
import json
from loguru import logger

from datetime import datetime
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='gowalla', help='dataset name: diginetica/gowalla/lastfm')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--layer',type=int, default=3, help='gcn propogation layers')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--train_flag', type=str, default="True")
parser.add_argument('--PATH', default='../checkpoint/gowalla.pt', help='checkpoint path')
parser.add_argument('--dot', default=True, action='store_true')
parser.add_argument('--last_k', type=int, default=7)
parser.add_argument('--l_p', type=int, default=4)
parser.add_argument('--use_attn_conv', type=str, default="True")
parser.add_argument('--heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--scale', default=True, help='scaling factor sigma')
parser.add_argument('--norm', default=True, help='adapt NISER, l2 norm over item and session embedding')
opt = parser.parse_args()
print(opt)

def main():

    log_name = opt.dataset + '_CacheGNN_' + time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()) + ".log"
    # log_name = ("{}_CacheGNN_{}.log",opt.dataset,time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()))
    if not os.path.exists("log"):
        os.makedirs("log")
    logger.remove()
    logger.add(os.path.join("log", log_name), level='INFO')
    logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True, level='INFO')
    logger.info("[Config]" + '\n' + json.dumps(vars(opt), indent=4))

    train_data = pickle.load(open(f'../data/{opt.dataset}/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(f'../data/{opt.dataset}/test.txt', 'rb'))

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    elif opt.dataset == 'Tmall':
        num_node = 40728
    elif opt.dataset == 'gowalla':
        n_node = 29511
    else:
        n_node = 38616


    model = trans_to_cuda(StarSessionGraph(opt, n_node))
    #模型并行
    # gpus=[1,2,3]
    # model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[1])

    if opt.train_flag == "True":
        start = time.time()
        best_result = [0, 0]
        best_epoch = [0, 0]
        bad_counter = 0
        for epoch in range(opt.epoch):
            print('-------------------------------------------------------')
            print('epoch: ', epoch)
            logger.info(f'[epoch:{epoch}]')
            hit, mrr = train_test(model, train_data, test_data)
            flag = 0
            if hit >= best_result[0]:
                best_result[0] = hit
                best_epoch[0] = epoch
                flag = 1
                save_path = f'../checkpoint/{opt.dataset}-best_hit.pth'
                torch.save(model.state_dict(), path)
            if mrr >= best_result[1]:
                best_result[1] = mrr
                best_epoch[1] = epoch
                flag = 1
                save_path = f'../checkpoint/{opt.dataset}-best_mrr.pth'
                torch.save(model.state_dict(), path)
            print('hit:')
            print(hit)
            print("\n")
            print('mrr:')
            print(mrr)
            print("\n")
            print('Best Result:')
            print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            logger.info(f'hit:{hit}\tmrr:{mrr}')
            logger.info('Best Result:  Recall@20: {:.4f}(epoch:{})\tMMR@20: {:.4f}(epoch:{})'
                        .format(best_result[0], best_epoch[0], best_result[1], best_epoch[1]))
            bad_counter += 1 - flag
            if bad_counter >= opt.patience:
                break
        print('-------------------------------------------------------')
        end = time.time()
        print("Run time: %f s" % (end - start))
        logger.info(f'Run time:{end - start}')
    else:
        model.load_state_dict(torch.load(opt.PATH), False)
        hit, mrr = formal_test(model, train_data, test_data)
        print(hit)
        print(mrr)

if __name__ == '__main__':
    main()
