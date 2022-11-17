from Experiment.Approach.HARDMSEG.loss import structureloss
from Experiment.Approach.HARDMSEG.hardmseg import HarDMSEG
from Utils.support_technique import EarlyStopping, adjust_lr, AvgMeter, clip_gradient
from Utils.dataloader import SegDataset
import os
import sys
import glob
import argparse
from datetime import datetime
import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchstat import stat

sys.path.append('D:/GithubCloneRepo/Stomach-Status-Classification')


def train(train_loader, model, optimizer, epoch, test_path, patience, n_epochs):
	pass


if __name__ == '__main__':
	for x in range(3):
		os.chdir("..")
	main_data_dir = os.getcwd() + "/Data set"
	seg_data_dir = main_data_dir + "/Seg_Task_Data"

	train_seg_data_dir = seg_data_dir + "/Train"
	train_seg_img_dir = train_seg_data_dir + "/Img"
	train_seg_img_files = glob.glob(train_seg_img_dir + "/*")
	train_seg_mask_dir = train_seg_data_dir + "/Mask"
	train_seg_mask_files = glob.glob(train_seg_mask_dir + "/*")

	test_seg_data_dir = seg_data_dir + "/Test"
	test_seg_img_dir = test_seg_data_dir + "/Img"
	test_seg_img_files = glob.glob(test_seg_img_dir + "/*")
	test_seg_mask_dir = test_seg_data_dir + "/Mask"
	test_seg_mask_files = glob.glob(test_seg_mask_dir + "/*")
 parser = argparse.ArgumentParser()
 
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    
    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')
    
    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')
    
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    
    parser.add_argument('--train_path', type=str,
                        default='/work/james128333/PraNet/TrainDataset', help='path to train dataset')
    
    parser.add_argument('--test_path', type=str,
                        default='/work/james128333/PraNet/TestDataset/Kvasir' , help='path to testing Kvasir dataset')
    
    parser.add_argument('--train_save', type=str,
                        default='HarD-MSEG-best')
    
    opt = parser.parse_args()


	
	