#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:55:54 2019

@author: felix
"""
# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
from .unet_parts import *

class UNet5(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet5, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)
        self.dropout = nn.Dropout2d(0.2)
        self.outc = outconv(64, n_classes)

    def forward(self, x, dropout=True):
        if dropout==True:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x2 = self.dropout(x2)
            x3 = self.down2(x2)
            x3 = self.dropout(x3)
            x4 = self.down3(x3)
            x4 = self.dropout(x4)
            x5 = self.down4(x4)
            x5 = self.dropout(x5)
            x6 = self.down5(x5)
            x6 = self.dropout(x6)
            x = self.up1(x6, x5)
            x = self.dropout(x)
            x = self.up2(x, x4)
            x = self.dropout(x)
            x = self.up3(x, x3)
            x = self.dropout(x)
            x = self.up4(x, x2)
            x = self.dropout(x)
            x = self.up5(x, x1)
            x = self.outc(x)
            return F.sigmoid(x)
            


        else:
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x6 = self.down5(x5)
            x = self.up1(x6, x5)
            x = self.up2(x, x4)
            x = self.up3(x, x3)
            x = self.up4(x, x2)
            x = self.up5(x, x1)
            x = self.outc(x)
            return F.sigmoid(x)
