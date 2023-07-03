# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F
from torchvision import ops


class LayerNorm2d(nn.Module):
    def __init__(self, out_channels):
      super(LayerNorm2d, self).__init__()
      self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.out_channels), eps=1e-06)
        x = x.permute(0, 3, 1, 2)
        return x


class Conv2dAuto(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pointwise=False):
        super(Conv2dAuto, self).__init__()
        self.kernel_size = kernel_size
        self.padding =  self.kernel_size // 2 # dynamic add padding based on the kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pointwise = pointwise

    def getLayer(self):
        if(not(self.pointwise)):
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_size,self.kernel_size), stride=(1,1), padding=(self.padding, self.padding), groups=self.out_channels, bias=False)
        else:
            if(self.padding==0):
                return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_size,self.kernel_size), stride=(1,1), bias=False)
            else:
                return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_size,self.kernel_size), stride=(1,1), padding=(self.padding, self.padding), bias=False)


class stem(nn.Module):
  #class to create stem. If mode=="patch", it refers to stem used in convNeXt, if "mobile" it refers to stem used in MobileNet
  def __init__(self, mode="patch", out_channels=16):
        super(stem, self).__init__()

        if mode=="patch":
          self.firstBlock = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=(4,4), stride=(4,4), padding=0),
            LayerNorm2d((out_channels,), eps=1e-06)
          )

        elif mode=="mobile":
          self.firstBlock = nn.Sequential(
              nn.Conv2d(3, out_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
              nn.GELU(),
              nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
              nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False),
              nn.GELU(),
              nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          )

        else:
          raise Exception("stem accepts as mode only \"patch\" or \"mobile\"")


class downsampling(nn.Module):
  #class to perform downsampling
  def __init__(self, in_channels, out_channels, kernel_size=3):
    super(downsampling, self).__init__()

    if out_channels < in_channels:
      raise Exception("out_channels should not be lower than in_channels")

    #downsampling between stages
    if(kernel_size == 3):
      #output_channels=input_channels
      self.ds = nn.Sequential(
          #depthwise: groups=in_channels
          nn.Conv2d(in_channels, in_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1), groups=in_channels),
          #pointwise
          nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
          nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True)
      )

    #downsampling after stem
    else:
      self.ds = nn.Sequential(
          #depthwise: groups=in_channels
          nn.Conv2d(in_channels, in_channels, kernel_size=(4,4), stride=(2,2), padding=(2,2), groups=in_channels),
          #pointwise
          nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)),
          nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
          nn.ReLU(inplace=True)
      )

class block(nn.Module):
  def __init__(self, mode, in_channels, out_channels, exp_ratio, kernel):
    super(block, self).__init__()

    if out_channels < in_channels:
      raise Exception("out_channels should not be lower than in_channels")

    if mode=="i":
      #inverted bottleneck
      self.layers = nn.Sequential(
        #expansion
        nn.Conv2d(in_channels, out_channels*exp_ratio, kernel_size=(1,1), stride=(1,1), bias=False),
        nn.BatchNorm2d(out_channels*exp_ratio, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        #depthwise
        Conv2dAuto(out_channels*exp_ratio, out_channels*exp_ratio, kernel).getLayer(),
        nn.BatchNorm2d(out_channels*exp_ratio, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        #compression
        nn.Conv2d(out_channels*exp_ratio, out_channels, kernel_size=(1,1), stride=(1,1)),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True)
      )
    elif mode=="c":
      #block convnext
      self.layers = nn.Sequential(
        #depthwise
        Conv2dAuto(in_channels, in_channels, kernel).getLayer(),
        LayerNorm2d((in_channels,)),
        nn.GELU(approximate='none'),
        #expansion
        nn.Conv2d(in_channels, out_channels*exp_ratio, kernel_size=(1,1), stride=(1,1), bias=False),
        LayerNorm2d((out_channels*exp_ratio,)),
        nn.GELU(approximate='none'),
        #compression
        nn.Conv2d(out_channels*exp_ratio, out_channels, kernel_size=(1,1), stride=(1,1)),
        LayerNorm2d((out_channels,)),
        nn.GELU(approximate='none')
      )
      
        
def skip_connection(out, identity):
      if ( out.size(2) == identity.size(2) and out.size(3) == identity.size(3) ):
        out = out + torch.cat([identity,torch.zeros(identity.size(0),out.size(1)-identity.size(1),identity.size(2),identity.size(3)).to("cuda")], dim=1)
        return out
      else:
        out = out[:,:,:out.size(2)-1,:out.size(3)-1] + torch.cat([identity,torch.zeros(identity.size(0),out.size(1)-identity.size(1),identity.size(2),identity.size(3)).to("cuda")], dim=1)
        return out 
      
      

class dnn_small(nn.Module):
  def __init__(self, structure, num_classes=2):
        super(dnn_small, self).__init__()
        #stem
        self.layer1 = stem("mobile").firstBlock
        self.ds = downsampling(16,16,4).ds
        #1 stage
        param_2 = structure[0]
        param_3 = structure[1]
        self.layer2 = block(param_2[0],16,param_2[1],param_2[2],param_2[3]).layers
        self.layer3 = block(param_3[0],param_2[1],param_3[1],param_3[2],param_3[3]).layers
        self.layer4 = downsampling(param_3[1],param_3[1]).ds
        #2 stage
        param_4 = structure[2]
        param_5 = structure[3]
        self.layer5 = block(param_4[0],param_3[1],param_4[1],param_4[2],param_4[3]).layers
        self.layer6 = block(param_5[0],param_4[1],param_5[1],param_5[2],param_5[3]).layers
        self.layer7 = downsampling(param_5[1],param_5[1]).ds
        #3 stage
        param_6 = structure[4]
        param_7 = structure[5]
        param_8 = structure[6]
        param_9 = structure[7]
        param_10 = structure[8]
        param_11 = structure[9]
        self.layer8 = block(param_6[0],param_5[1],param_6[1],param_6[2],param_6[3]).layers
        self.layer9 = block(param_7[0],param_6[1],param_7[1],param_7[2],param_7[3]).layers
        self.layer10 = block(param_8[0],param_7[1],param_8[1],param_8[2],param_8[3]).layers
        self.layer11 = block(param_9[0],param_8[1],param_9[1],param_9[2],param_9[3]).layers
        self.layer12 = block(param_10[0],param_9[1],param_10[1],param_10[2],param_10[3]).layers
        self.layer13 = block(param_11[0],param_10[1],param_11[1],param_11[2],param_11[3]).layers
        self.layer14 = downsampling(param_11[1],param_11[1]).ds
        #4 stage
        param_12 = structure[10]
        param_13 = structure[11]
        self.layer15 = block(param_12[0],param_11[1],param_12[1],param_12[2],param_12[3]).layers
        self.layer16 = block(param_13[0],param_12[1],param_13[1],param_13[2],param_13[3]).layers
        #convnext
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(param_13[1]), nn.Flatten(1), nn.Linear(param_13[1], num_classes)
        )
        

  def forward(self, x):
        x = x.to('cuda')
        out = self.layer1(x)
        out = self.ds(out)
        
        identity = out.clone()
        out = self.layer2(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer3(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer4(out)
        
        identity = out.clone()
        out = self.layer5(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer6(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer7(out)
        
        identity = out.clone()
        out = self.layer8(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer9(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer10(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer11(out)
        out = skip_connection(out.clone(), identity)
        
        
        identity = out.clone()
        out = self.layer12(out)
        out = skip_connection(out.clone(), identity)
        
        
        identity = out.clone()
        out = self.layer13(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer14(out)
        
        identity = out.clone()
        out = self.layer15(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer16(out)
        out = skip_connection(out.clone(), identity)
        
        out = self.avgpool(out)
        out = self.classifier(out)
        return out
        
        
class dnn_large(nn.Module):
  def __init__(self, structure, num_classes=2):
        super(dnn_large, self).__init__()
        #stem
        self.layer1 = stem("mobile").firstBlock
        self.ds = downsampling(16,16,4).ds
        #1 stage
        param_2 = structure[0]
        param_3 = structure[1]
        param_4 = structure[2]
        self.layer2 = block(param_2[0],16,param_2[1],param_2[2],param_2[3]).layers
        self.layer3 = block(param_3[0],param_2[1],param_3[1],param_3[2],param_3[3]).layers
        self.layer4 = block(param_4[0],param_3[1],param_4[1],param_4[2],param_4[3]).layers
        self.layer5 = downsampling(param_4[1],param_4[1]).ds
        #2 stage
        param_5 = structure[3]
        param_6 = structure[4]
        param_7 = structure[5]
        self.layer6 = block(param_5[0],param_4[1],param_5[1],param_5[2],param_5[3]).layers
        self.layer7 = block(param_6[0],param_5[1],param_6[1],param_6[2],param_6[3]).layers
        self.layer8 = block(param_7[0],param_6[1],param_7[1],param_7[2],param_7[3]).layers
        self.layer9 = downsampling(param_7[1],param_7[1]).ds
        #3 stage
        param_8 = structure[6]
        param_9 = structure[7]
        param_10 = structure[8]
        param_11 = structure[9]
        param_12 = structure[10]
        param_13 = structure[11]
        param_14 = structure[12]
        param_15 = structure[13]
        param_16 = structure[14]
        self.layer10 = block(param_8[0],param_7[1],param_8[1],param_8[2],param_8[3]).layers
        self.layer11 = block(param_9[0],param_8[1],param_9[1],param_9[2],param_9[3]).layers
        self.layer12 = block(param_10[0],param_9[1],param_10[1],param_10[2],param_10[3]).layers
        self.layer13 = block(param_11[0],param_10[1],param_11[1],param_11[2],param_11[3]).layers
        self.layer14 = block(param_12[0],param_11[1],param_12[1],param_12[2],param_12[3]).layers
        self.layer15 = block(param_13[0],param_12[1],param_13[1],param_13[2],param_13[3]).layers
        self.layer16 = block(param_14[0],param_13[1],param_14[1],param_14[2],param_14[3]).layers
        self.layer17 = block(param_15[0],param_14[1],param_15[1],param_15[2],param_15[3]).layers
        self.layer18 = block(param_16[0],param_15[1],param_16[1],param_16[2],param_16[3]).layers
        self.layer19 = downsampling(param_16[1],param_16[1]).ds
        
        #4 stage
        param_17 = structure[15]
        param_18 = structure[16]
        param_19 = structure[17]
        self.layer20 = block(param_17[0],param_16[1],param_17[1],param_17[2],param_17[3]).layers
        self.layer21 = block(param_18[0],param_17[1],param_18[1],param_18[2],param_18[3]).layers
        self.layer22 = block(param_19[0],param_18[1],param_19[1],param_19[2],param_19[3]).layers
        #convnext
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(param_19[1]), nn.Flatten(1), nn.Linear(param_19[1], num_classes)
        )
        

  def forward(self, x):
        x = x.to('cuda')
        out = self.layer1(x)
        out = self.ds(out)
        
        #1 stage
        identity = out.clone()
        out = self.layer2(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer3(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer4(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer5(out)
        
        #2 stage
        identity = out.clone()
        out = self.layer6(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer7(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer8(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer9(out)
        
        #3 stage
        identity = out.clone()
        out = self.layer10(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer11(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer12(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer13(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer14(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer15(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer16(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer17(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer18(out)
        out = skip_connection(out.clone(), identity)
        out = self.layer19(out)
        
        #4 stage
        identity = out.clone()
        out = self.layer20(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer21(out)
        out = skip_connection(out.clone(), identity)
        
        identity = out.clone()
        out = self.layer22(out)
        out = skip_connection(out.clone(), identity)
        
        out = self.avgpool(out)
        out = self.classifier(out)
        return out







