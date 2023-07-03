# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torchvision
from torch import Tensor
import torch.nn.functional as F
from torchvision import ops
import numpy as np
import fvcore

from blocksreconstruction import stem
from blocksreconstruction import block
from blocksreconstruction import downsampling
from blocksreconstruction import LayerNorm2d
from blocksreconstruction import dnn_small, dnn_large

from metrics import compute_naswot_score
from metrics import compute_synflow_per_weight
from metrics import sum_arr, get_layer_metric_array, _no_op
from metrics import count_flops, count_params

#function that creates the neural network, checks if the constraints on #params and #flops are respected, and if yes computes the scores and returns them

def fitnessFunction(model, minibatch_loader):
   device = torch.device('cuda')

   #create the neural network to compute #params, #flops and scores
   nn_instance = dnn_small(model).to('cuda') #use dnn_large if building a deeper model

   #check the constraints on #params
   n_params = count_params(nn_instance)
   if n_params <= 2.5e6:
    #print("number of parameters %d" % n_params)
    #check the #flops
    nn_instance.eval()
    n_flops = count_flops(nn_instance, torch.randn((1,3,224,224),))
    if n_flops <= 200e6:
      #print("number of flops %d" % n_flops)
      for batch_idx, (inputs, targets) in enumerate(minibatch_loader):
        #compute the score
        LogSynflow = compute_synflow_per_weight(nn_instance, inputs.to(device), targets.to(device), device)
        Naswot = compute_naswot_score(nn_instance, inputs.to(device), targets.to(device), device)
        break
      return LogSynflow, Naswot

    #if the constraints are not respected return None
    return

#generates the initial models by constructing blocks at random

import random

def RandomModel(structure, mode, out_channels, exp_ratio, kernel, P, max_LS, max_LR, minibatch_loader):

  population = []
  n_blocks = sum(structure)
  l = len(structure)

  #create splits in the out_channel list proportional to the number of blocks in each stage of the structure
  splits = [0]
  for i in range(l):
    splits.append((round((len(out_channels)/n_blocks)*structure[i])))
  for i in range(1, len(splits)):
    splits[i] = splits[i]+splits[i-1]
  if splits[l] != len(out_channels):
    splits[l] = len(out_channels)

  #create the first P models
  p = 0
  while p < P:
    model = []
    last_o = 0
    for i in range(l):
      for j in range(structure[i]):
        m = random.choice(mode)
        o = random.choice(out_channels[splits[i]:splits[i+1]])
        while ((len(model)>=1)&(o < last_o)):
          o = random.choice(out_channels[splits[i]:splits[i+1]])
        e = random.choice(exp_ratio)
        k = random.choice(kernel)
        model.append([m, o, e, k])
        last_o = o
    #print(p, model)

    #check the constraints
    metrics = fitnessFunction(model, minibatch_loader)
    if metrics is not None:
      LogSynflow, Naswot = metrics
      if LogSynflow > max_LS:
        max_LS = LogSynflow
      if Naswot > max_LR:
        max_LR = Naswot
      population.append({
          "model": model,
          "LS": LogSynflow,
          "LR": Naswot,
          "score": LogSynflow + Naswot
          }) #model saved as a list of lists(blocks) with relative scores
          
      p = p+1

  return population, max_LS, max_LR

#selects the parents from the population via tournament selection

import random

def TournamentSelection(population, k, max_LS, max_LR):

  #print("tournament selection")
  #normalize the scores with respect to the maximum
  for p in population:
    p['score'] = (p['LS']/max_LS) + (p['LR']/max_LR)

  #select a random subset of the population
  tournament = random.sample(population, k)

  #select the two with the highest score
  tournament = sorted(tournament, key=lambda x: x['score'], reverse=True)

  return tournament[0], tournament[1]

#function that performs the random mutation

import random

def RandomMutation(child, out_channels, exp_ratio, kernel):
  #print("performing mutation")
  
  #select the random block to modify
  n = random.choice(range(len(child)))
  #print("Mutation on block %d" % n)

  #select which parameter to modify
  p = random.choice(['m', 'o', 'e', 'k'])
  #print("Mutation on the parameter %c" % p)

  if p=='m':
    #in this case we only have two possible values, so we check which one we have and change it
    if child[n][0] == 'i':
      child[n][0] = 'c'
    else:
      child[n][0] = 'i'
  elif p=='o':
    replacement = random.choice(out_channels)
    #print("selected %d as replacement" % replacement)
    if n == 0:
      while((replacement > child[n+1][1])):  #-->in the case of the first block we only check that it is not greater than the value of the following block
        replacement = random.choice(out_channels)
        #print("selected value non valid, new value selected for replacement %d" % replacement)
    elif n == len(child)-1:
      while (replacement < child[n-1][1]): #-->in the case of the last block we only check that it is not smaller than the value of the previous block
        replacement = random.choice(out_channels)
        #print("selected value non valid, new value selected for replacement %d" % replacement)
    else:
      while((replacement < child[n-1][1])|(replacement > child[n+1][1])):
        replacement = random.choice(out_channels)
        #print("selected value non valid, new value selected for replacement %d" % replacement)
    #print("Replacing %d with %d" % (child[n][1], replacement))
    child[n][1] = replacement
  elif p=='e':
    replacement = random.choice(exp_ratio)
    while(replacement==child[n][2]): #-->to make sure that we replace it with a different value
      replacement = random.choice(exp_ratio)
    #print("Replacing %d with %d" % (child[n][2], replacement))
    child[n][2] = replacement
  elif p=='k':
    replacement = random.choice(kernel)
    while(replacement==child[n][3]): #-->to make sure that we replace it with a different value
      replacement = random.choice(kernel)
    #print("Replacing %d with %d" % (child[n][3], replacement))
    child[n][3] = replacement

  return child

#function that performs the crossover between the parents by doing uniform crossover

import numpy as np

def Crossover(parent_1, parent_2):
  #print("performing crossover")
  length = len(parent_1)
  child = []

  #we generate a vector of the length of our desired child, and fill it vith values sampled from a uniform distribution
  unif = np.random.rand(length)

  #check that not all the selections are from the same parent (not all are <0.5 or >0.5)
  while (all(x < 0.5 for x in unif))|(all(x>= 0.5 for x in unif)):
    #print("only sampling from one parent, selecting uniform again")
    unif = np.random.rand(length)

  for i in range(length):
    if unif[i] < 0.5:
      child.append(parent_1[i])
    else:
      child.append(parent_2[i])

  while (checkOutChannels(child) == False):
    #print("crossover failed, trying again")
    child = []
    unif = np.random.rand(length)
    #we generate a vector of the length of our desired child, and fill it vith values sampled from a uniform distribution
    while (all(x < 0.5 for x in unif))|(all(x>= 0.5 for x in unif)):
      #print("only sampling from one parent, selecting uniform again")
      unif = np.random.rand(length)

    for i in range(length):
      if unif[i] < 0.5:
        child.append(parent_1[i])
      else:
        child.append(parent_2[i])

  return child

#function that checks that the out_channels are not decreasing

def checkOutChannels(child):
  l = len(child)
 
  for i in range(l):
    current_o = child[i][1]
    if (i == 0):
      if (current_o > child[i+1][1]):
        #print("Error: out channels decreasing! (first block)")
        return False
    elif (i == len(child)-1):
      if(current_o < child[i-1][1]):
        #print("Error: out channels decreasing! (last block)")
        return False
    elif (i > 0)&(i < len(child)-1):
      if (current_o < child[i-1][1])|(current_o > child[i+1][1]):
        #print("Error: out channels decreasing! (block %d)" % i)
        return False
  #print("out channels correct")
  return True

#function that checks if a mutation actually happened

def checkMutation(parent, child):
  l = len(child)
  for i in range(l):
    c_mode = child[i][0]
    p_mode = parent[i][0]
    c_out = child[i][1]
    p_out = parent[i][1]
    c_exp = child[i][2]
    p_exp = parent[i][2]
    c_kern = child[i][3]
    p_kern = parent[i][3]
    
    if (c_mode != p_mode)|(c_out != p_out)|(c_exp != p_exp)|(c_kern != p_kern):
      return True
  #print("No mutation actually happened")
  return False

#performs a random mutation in each of the parents and a crossover between them

import copy

def RandomMutationAndCrossover(parent_1, parent_2, out_channels, exp__ratio, kernel, max_LS, max_LR, minibatch_loader):
  #print("generating children")
  children = []

  #we perform a random mutation in each of the parents, selecting one of the blocks randomly and for one of the parameters, substituting its value with another from the possible values
  #print("mutation on parent 1")
  child_1 = copy.deepcopy(parent_1)
  while (checkOutChannels(child_1) == False)|(checkMutation(parent_1, child_1) == False)|(fitnessFunction(child_1, minibatch_loader) is None):
    child_1 = copy.deepcopy(parent_1) #in order to start from a copy of the parent each time we have to run the loop again
    child_1 = RandomMutation(child_1, out_channels, exp__ratio, kernel)

  #print("parent:")
  #print(parent_1)
  #print("new child:")
  #print(child_1)
  
  #create the neural network, check if the constraints are respected, and if yes compute the score and add the child to those to be returned
  metrics = fitnessFunction(child_1, minibatch_loader)
  LogSynflow, Naswot = metrics
  if LogSynflow > max_LS:
    max_LS = LogSynflow
  if Naswot > max_LR:
    max_LR = Naswot
  children.append({
      "model": child_1,
      "LS": LogSynflow,
      "LR": Naswot,
      "score": LogSynflow + Naswot
      })

  #print("mutation on parent 2")
  child_2 = copy.deepcopy(parent_2)
  while (checkOutChannels(child_2) == False)|(checkMutation(parent_2, child_2) == False)|(fitnessFunction(child_2, minibatch_loader) is None):
    child_2 = copy.deepcopy(parent_2)
    child_2 = RandomMutation(child_2, out_channels, exp__ratio, kernel)

  #print("parent:")
  #print(parent_2)
  #print("new child:")
  #print(child_2)
  
  #create the neural network, check if the constraints are respected, and if yes compute the score and add the child to those to be returned
  metrics = fitnessFunction(child_2, minibatch_loader)
  LogSynflow, Naswot = metrics
  if LogSynflow > max_LS:
    max_LS = LogSynflow
  if Naswot > max_LR:
    max_LR = Naswot
  children.append({
      "model": child_2,
      "LS": LogSynflow,
      "LR": Naswot,
      "score": LogSynflow + Naswot
      })

  #perform the crossover between the parents
  child_3 = Crossover(parent_1, parent_2)
  while fitnessFunction(child_3, minibatch_loader) is None:
    child_3 = Crossover(parent_1, parent_2)

  #create the neural network, check if the constraints are respected, and if yes compute the score and add the child to those to be returned
  if (checkMutation(parent_1, child_3) == True)&(checkMutation(parent_2, child_3) == True):
    metrics = fitnessFunction(child_3, minibatch_loader)
    LogSynflow, Naswot = metrics
    if LogSynflow > max_LS:
      max_LS = LogSynflow
    if Naswot > max_LR:
      max_LR = Naswot
    children.append({
        "model": child_3,
        "LS": LogSynflow,
        "LR": Naswot,
        "score": LogSynflow + Naswot
        })

  #perform the crossover between the mutated children
  child_4 = Crossover(child_1, child_2)
  while fitnessFunction(child_4, minibatch_loader) is None:
    child_4 = Crossover(child_1, child_2)

  #create the neural network, check if the constraints are respected, and if yes compute the score and add the child to those to be returned
  if (checkMutation(child_1, child_4) == True)&(checkMutation(child_2, child_4) == True):
    metrics = fitnessFunction(child_4, minibatch_loader)
    LogSynflow, Naswot = metrics
    if LogSynflow > max_LS:
      max_LS = LogSynflow
    if Naswot > max_LR:
      max_LR = Naswot
    children.append({
        "model": child_4,
        "LS": LogSynflow,
        "LR": Naswot,
        "score": LogSynflow + Naswot
        })

  return children, max_LS, max_LR

#genetic algorithm function

def Genetic(structure, mode, out_channels, exp_ratio, kernel, P, k, N_iter, minibatch_loader):

  #population = []
  history = []

  max_LS = 0
  max_LR = 0

  #initialize the population
  population, max_LS, max_LR = RandomModel(structure, mode, out_channels, exp_ratio, kernel, P, max_LS, max_LR, minibatch_loader)

  #insert the initial population in the history
  for p in population:
    history.append(p)

  for iter in range(N_iter):
    #print("%d iteration" % iter)
    #select the parents from the population
    parent_1, parent_2 = TournamentSelection(population, k, max_LS, max_LR) #-->definire questo k

    #perform the mutations and crossover to generate the children
    #children = []
    children, max_LS, max_LR = RandomMutationAndCrossover(parent_1["model"], parent_2["model"], out_channels, exp_ratio, kernel, max_LS, max_LR, minibatch_loader)

    #insert the children in the history
    for c in children:
      history.append(c)
        #remove the c oldest element from the population and insert the children instead
      population.pop(0) #-->removes the first element that was inserted in the list, since when we insert new elements we always append them at the end
      population.append(c)

  #normalizzazione dello score
  #print("total models in the history: %d" % len(history))
  for h in history:
    h['score'] = (h['LS']/max_LS) + (h['LR']/max_LR)

  #find the best model in history by sorting the history by score values
  history = sorted(history, key=lambda x: x['score'], reverse=True)
  best_model = history[0]
  return best_model, population

def run_genetic(train_data):
  #defining the spaces of the parameters that make up the blocks
  mode = ['i', 'c']
  out_channels = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 304, 368, 432, 496] #to be modified and take only up 224 in case of dnn_large
  exp_ratio = [2, 4, 6]
  kernel = [3, 5, 7]

  structure = [2, 2, 6, 2]
  #structure = [3, 3, 9, 3] #for dnn_large

  P = 25 #population size
  k = 5 #tournament selection size
  N_iter = 250

  #minibatch to compute metrics
  minibatch_loader = torch.utils.data.DataLoader(train_data, batch_size=64)

  final_model, final_population = Genetic(structure, mode, out_channels, exp_ratio, kernel, P, k, N_iter, minibatch_loader)

  #print("The best model has blocks:")
  #print(final_model['model'])
  #print("and a score of %.6f" % final_model['score'])

  nn_instance = dnn_small(final_model["model"]).to("cuda") #use dnn_large if building a deeper model
  n_parameters = count_params(nn_instance)
  nn_instance.eval()
  n_flops = count_flops(nn_instance, torch.randn((1,3,224,224),))
  #print("number of parameters is %.2f and number of flops is %.2f" % (n_parameters, n_flops))

  return nn_instance, final_population
