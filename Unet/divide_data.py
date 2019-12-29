#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:16:39 2019

@author: felix
"""

import os
import random
import shutil
import fnmatch

################################################################
#divide a single directory into a train, eval and test directory


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_list(path, ext):
    list_names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*' + ext):          
            list_names.append(os.path.join(root, filename))
    return list_names


root = 'classes_train/'
train = './trainset/'
val = './valset/'
test = './test/'

ext = '.wav'
list_subfolders = [f.path for f in os.scandir(root) if f.is_dir() ]  
list_subfolders.sort()

for ii in range(len(list_subfolders)):
    classB = create_list(list_subfolders[ii], ext)
    random.shuffle(classB)
    n_train = int(len(classB) * 0.8)
    trainset = classB[0:n_train]
    interset = classB[n_train:]
    n_test = int(len(interset) * 0.5)
    testset = interset[0:n_test]
    evalset = interset[n_test:]
    
    createFolder(train+list_subfolders[ii].rsplit("/", 1)[1])
    createFolder(val+list_subfolders[ii].rsplit("/", 1)[1])
    createFolder(test+list_subfolders[ii].rsplit("/", 1)[1])
    ##dividing the dataset
    for ii in trainset:
        shutil.copy(ii, train+ii.split('/', 1)[1])           
    for jj in evalset:
        shutil.copy(ii, val+jj.split('/', 1)[1])     
    for hh in testset:
        shutil.copy(ii, test+hh.split('/', 1)[1])    
    
   
