#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 22:14:38 2019

@author: felix
"""

import os
import random
import time
import fnmatch
import csv
import torch
from arguments import ArgParser
from unet import UNet5
import torch.nn as nn
#from tensorboardX import SummaryWriter
from matplotlib import image as mpimg
from Dataloader import Dataset
import matplotlib
import numpy as np
import collections
import scipy
#organize the name files according to their number

def create_list(path, ext):
    list_names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*' + ext):
            list_names.append(os.path.join(root, filename))
    return list_names


def create_optimizer(nets, args):
    net_sound = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups)


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def unwrap_mask(infos):
#"for a kernel of 5"
    gt_masks = torch.empty(args.batch_size, args.nb_classes, 256, 259, dtype=torch.float)
    for ii in range(args.batch_size):
        for jj in range(args.nb_classes):
            gt_masks[ii, jj] = infos[jj][2][ii]
    return gt_masks


def build_audio(audio_names, pred_masks, magmix, phasemix):
    for ii in range(args.batch_size):
        pred_masks = pred_masks[ii].numpy()
        magmix     = magmix[ii].squeeze(0).detach().numpy()
        phasemix   = phasemix[ii].squeeze(0).detach().numpy()
        for n in range(args.nb_classes):
            name   = audio_names[n][1][ii]
            magnew = pred_masks[n]*magmix
            spec   = magnew.astype(np.complex)*np.exp(1j*phasemix)
            audio  = librosa.istft(spec, hop_length=256)
            scipy.io.wavfile.write('restored_audio/restored_{}.wav'.format(name), 22050, audio)


def save_arguments(args, path):
    file1 = open(path+"/infos.txt","w")
    print("Input arguments:")
    for key, val in vars(args).items():
        file1.writelines([key, str(val), '\n']) 
        print("{:16} {}".format(key, val))
    file1.close()


def evaluation(net, loader, args):
#no upgrade over the gradient    
    torch.set_grad_enabled(False)
    num_batch = 0
    criterion = nn.BCELoss()
    args.out_threshold = 0.4
    for ii, batch_data in enumerate(loader):
       # forward pass
        magmix = batch_data[0]
        magmix = magmix.to(args.device)
        masks  = unwrap_mask(batch_data[2])
        masks = masks.to(args.device)
        num_batch += 1
        masks_pred = net(magmix)
#        #loss
        loss = criterion(masks_pred, masks)
        #writing of the Loss values and elapsed time for every batch
        batchtime = (time.time() - args.starting_training_time)/60 #minutes
        with open(args.path + "/loss_times_eval.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([str(loss.cpu().detach().numpy()), batchtime, num_batch])


def train(net, loader_train, optimizer, path,  args):
    torch.set_grad_enabled(True)
    num_batch = 0 
    criterion = nn.BCELoss()     
    for ii, batch_data in enumerate(loader_train):
        #add species names in infos.txt
        if ii == 0:
            args.species = []
            for n in range(args.nb_classes):
                args.species.append(batch_data[2][n][0][0])
            save_arguments(args, args.path)

        num_batch += 1
        magmix     = batch_data[0]
        magmix     = magmix.to(args.device)
        masks_pred = net(magmix, dropout=True)        
        masks      = unwrap_mask(batch_data[2])
        masks      = masks.to(args.device)
        loss       = criterion(masks_pred, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
#        #writing of the Loss values and elapsed time for every batch
        batchtime = (time.time() - args.starting_training_time)/60 #minutes
#        #Writing of the elapsed time and loss for every batch 
        with open(args.path + "/loss_times.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([str(loss.cpu().detach().numpy()), batchtime, num_batch]) 
        if ii%args.save_per_batchs == 0:  
            torch.save({                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
            args.path + '/Saved_models/model_batchs{}.pth.tar'.format(num_batch))


#***************************************************    
#****************** MAIN ***************************    
#***************************************************   
if __name__ == '__main__':
    # arguments
    parser          = ArgParser()
    args            = parser.parse_train_arguments()
    args.batch_size = 16
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.starting_training_time = time.time()
    args.save_per_batchs = 500
    #nb of class to train the net on 
    args.nb_classes = 2
    #names of the species the net is training on
    args.name_classes = ['crow', 'eastern_wood_pewee']
    args.mode = 'train'
    args.lr_sounds = 1e-5   
    #model definition     
    net = UNet5(n_channels=1, n_classes=args.nb_classes)
    net.apply(init_weights)
    net = net.to(args.device)
    # Set up optimizer
    optimizer = create_optimizer(net, args)
    #path to the repertory where to save everything
    args.path = "./Article/no_aug_data_zero"

    args._augment = 'Unet5 output for 2 species_base de donnees10species_for 3s training audio_3calls en noise sans data augmentation, -10,0 SNR pour natural noise et 0,50 pour gaussian noise avec comparaison mag et mag noise pour mask'
###########################################################
################### TRAINING ##############################
###########################################################      
    if args.mode == 'train':
        #OverWrite the Files for loss saving and time saving
        fichierLoss = open(args.path+"/loss_times.csv", "w")
        fichierLoss.close()
        #Dataset loading of the bird calls for n different species
        #the diversity in the bird call used as noise will make the task more 
        # complicated for the network
        root = './data_sound/trainset10/'
        ext  = '.wav'    
        #path_background is the repertory for the noise we put in the 
        #background of all the bird calls
        train_classes = Dataset(root, name_classes=args.name_classes, nb_class_noise =3, path_background="./data_sound/noises/")
        loader_train  = torch.utils.data.DataLoader(
        train_classes,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=10)

        for epoch in range(0, 1):
            train(net, loader_train, optimizer, args.path, args) 
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
            args.path+'/Saved_models/model_epoch{}.pth.tar'.format(epoch))      
