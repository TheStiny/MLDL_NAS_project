# -*- coding: utf-8 -*-



import torch
from torch import nn, Tensor
import torch.nn.functional as F
from blocksreconstruction import stem
from blocksreconstruction import block
from blocksreconstruction import downsampling
from blocksreconstruction import Conv2dAuto
from blocksreconstruction import dnn_small, dnn_large
from blocksreconstruction import skip_connection

#!pip install pyvww

import pyvww
from pyvww.utils import VisualWakeWords
from torchvision.datasets import VisionDataset
from torchvision import transforms as T
from PIL import Image
import os

class VisualWakeWordsClassification(VisionDataset):
    """`Visual Wake Words <https://arxiv.org/abs/1906.05721>`_ Dataset.
    Args:
        root (string): Root directory where COCO images are downloaded to.
        annFile (string): Path to json visual wake words annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=T.ToTensor(), target_transform=None, transforms=None):
        super(VisualWakeWordsClassification, self).__init__(root, transforms, transform, target_transform)
        self.vww = VisualWakeWords(annFile)
        self.ids = list(sorted(self.vww.imgs.keys()))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the index of the target class.
        """
        vww = self.vww
        img_id = self.ids[index]
        ann_ids = vww.getAnnIds(imgIds=img_id)
        if ann_ids:
            full_target = vww.loadAnns(ann_ids)
            categories = [ann['category_id'] for ann in full_target]
            if 1 in categories:
              target = 1  # l'immagine contiene una persona
            else:
              target = 0  # l'immagine non contiene una persona
        else:
            target = 0

        path = vww.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.ids)

#!mkdir train2017
#!mkdir val2017

#to untar tar files containing dataset
#!tar -xvf train2017_224.tar -C train2017
#!tar -xvf val2017_224.tar -C val2017

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision.models import convnext_tiny
from torchvision import datasets
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import tqdm
import pyvww

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#!pip install timm

from timm.data.transforms_factory import create_transform

transform1 = create_transform((3,224,224), is_training=True)
transform2 = create_transform((3,224,224), is_training=False)


train_data = VisualWakeWordsClassification(root="train2017",
                    annFile="instances_train2017.json", transform=transform1)

val_data = VisualWakeWordsClassification(root="val2017",
                    annFile="instances_val2017.json", transform=transform2)



import torchvision
import numpy as np
from matplotlib import pyplot as plt


#!pip install -U fvcore
import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F
from torchvision import ops
import numpy as np
import fvcore

from genetic import run_genetic
from metrics import compute_naswot_score
from metrics import compute_synflow_per_weight
from metrics import sum_arr, get_layer_metric_array, _no_op
from metrics import count_flops, count_params
from genetic import fitnessFunction
from genetic import RandomModel
from genetic import TournamentSelection
from genetic import RandomMutation
from genetic import Crossover
from genetic import checkOutChannels
from genetic import checkMutation
from genetic import RandomMutationAndCrossover
from genetic import Genetic

import copy
import numpy as np
import random

nn_instance, population = run_genetic(train_data)

from torch.optim.lr_scheduler import CosineAnnealingLR

def get_data(batch_size, test_batch_size=256):
  # Prepare data transformations and then combine them sequentially
  #transform = list()
  #transform.append(T.Resize((227,227)))
  #transform.append(T.ToTensor()) # Converts Numpy to Pytorch Tensor
  #transform.append(T.Normalize(mean=[0.5], std=[0.5])) # Normalizes the Tensors between [-1, 1]
  #transform = T.Compose(transform) # Composes the above transformations into one.
  # Load data
  full_training_data = train_data
  test_data = val_data
  # Create train and validation splits
  num_samples = len(full_training_data)
  training_samples = int(num_samples*0.8+1)
  validation_samples = num_samples - training_samples
  training_data, validation_data = torch.utils.data.random_split(full_training_data, [training_samples,
  validation_samples])
  # Initialize dataloaders
  train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True)
  val_loader = torch.utils.data.DataLoader(validation_data, test_batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False)

  return train_loader, val_loader, test_loader

#first we define the needed functions to get the loss function
def get_loss_function():
  loss_function = nn.CrossEntropyLoss()
  return loss_function
# and also we need optimizer to optimze the weights base on the loss
def get_optimizer(net, lr, wd):
  optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
  return optimizer

#define a train function
def train(net,data_loader,optimizer,loss_function, scheduler, device='cuda'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.
  net.train() # Strictly needed if network contains layers which has different behaviours between train and test
  with tqdm.tqdm(total=len(data_loader)) as pbar:
    for batch_idx, (inputs, targets) in enumerate(data_loader):
      # Load data into GPU
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = net(inputs) # Forward pass

      loss = loss_function(outputs,targets) # Apply the loss
      loss.backward() # Backward pass
      optimizer.step() # Update parameters
      optimizer.zero_grad() # Reset the optimizer
      samples += inputs.shape[0]
      cumulative_loss += loss.item()
      _, predicted = outputs.max(1)
      cumulative_accuracy += predicted.eq(targets).sum().item()
      pbar.set_postfix_str("training with Current loss: {:.4f}, Accuracy: {:.4f}, at iteration: {:.1f}".format(cumulative_loss/ samples, cumulative_accuracy / samples*100, float(batch_idx)))
      pbar.update()
    scheduler.step()
  return cumulative_loss/samples, cumulative_accuracy/samples*100

#also to test the accuracy of the model we are going to use a test function to test the accuracy over the validation set
def test(net, data_loader, loss_function, device='cuda'):
  samples = 0.
  cumulative_loss = 0.
  cumulative_accuracy = 0.
  net.eval() # Strictly needed if network contains layers which have different behaviours between train and test
  with tqdm.tqdm(total=len(data_loader)) as pbar:
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = net(inputs)

        _, predicted = outputs.max(1)
        loss = loss_function(outputs,targets)
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        cumulative_accuracy += predicted.eq(targets).sum().item()
        pbar.set_postfix_str("validation with Current loss: {:.4f}, Accuracy: {:.4f}, at iteration: {:.1f}".format(cumulative_loss/ samples, cumulative_accuracy / samples*100, float(batch_idx)))
        pbar.update()
  return cumulative_loss/samples, cumulative_accuracy/samples*100

def trainer(
    # lets define the basic hyperparameters
    batch_size=128,
    learning_rate=0.003,
    weight_decay=0.02,
    epochs=20,
    model=None):
  #now we load the data in three splits train, test and validation
  train_loader, val_loader, test_loader = get_data(batch_size)
  # Moving the resnet to gpu device if it is available
  net = model.to(device)
  # defining the optimizer
  optimizer = get_optimizer(net, learning_rate, weight_decay)
  #defining the scheduler
  scheduler = CosineAnnealingLR(optimizer, 40)
  # defining the loss function
  loss_function = get_loss_function()
  # finaly training the model


  # In order to save the accuracy and loss we use a list to save them in each epoch
  val_loss_list = []
  val_accuracy_list = []
  train_loss_list = []
  train_accuracy_list = []
  print(f"learning rate: {learning_rate}")
  for e in range(epochs):
    print('training epoch number {:.2f} of total epochs of {:.2f}'.format(e,epochs))
    train_loss, train_accuracy = train(net, train_loader, optimizer, loss_function, scheduler)
    val_loss, val_accuracy = test(net, val_loader, loss_function)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)


    print('Epoch: {:d}'.format(e+1))
    print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
    train_accuracy))
    print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
    val_accuracy))
  print('-----------------------------------------------------')
  print('After training:')
  train_loss, train_accuracy = test(net, train_loader, loss_function)
  val_loss, val_accuracy = test(net, val_loader, loss_function)
  test_loss, test_accuracy = test(net, test_loader, loss_function)
  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss,
  train_accuracy))
  print('\t Validation loss {:.5f}, Validation accuracy {:.2f}'.format(val_loss,
  val_accuracy))
  print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')
  return val_loss_list, val_accuracy_list, train_loss_list, train_accuracy_list

val_loss,val_accuracy, train_loss,train_accuracy = trainer(model=nn_instance)
