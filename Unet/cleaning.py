#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:22:10 2019

@author: felix
"""


import os
import fnmatch

##############################################################
#create directories of the bird names based on the files names
##############################################################


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return 0

#make the list of all audio files
def create_list(path, ext):
    list_audios = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*' + ext):
            list_audios.append(os.path.join(root, filename))
    return list_audios


#looking for the bird name in the wav file name
def num_char(x):
    return(x.rsplit("(", 1)[0].split("+", 1)[1]) 


path_audio = './audio/'
ext = '.wav'
audio_list = create_list(path_audio, ext)
audio_list.sort()

for ii in range(len(audio_list)):   
    classB = num_char(audio_list[ii])
    createFolder(path_audio + classB)
    os.rename(audio_list[ii], path_audio + classB +'/'+ audio_list[ii].rsplit('/', 1)[1])




