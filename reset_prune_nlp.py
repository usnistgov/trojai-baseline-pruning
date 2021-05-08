"""
Disclaimer
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
"""
__author__ = "Peter Bajcsy"
__copyright__ = "Copyright 2020, The IARPA funded TrojAI project"
__credits__ = ["Michael Majurski", "Tim Blattner", "Derek Juba", "Walid Keyrouz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
__status__ = "Research"

import torch
import torch.nn as nn
import transformers.models as tm
import random
import numpy as np


"""
This class is designed to reset nodes to zero in a graph defining an AI model 
(input, forget, cell, and output gates for round 5 - GRU and LSTM architectures).
"""


##################################################################
def reset_prune_model(model, model_name, sample_shift, sampling_method, ranking_method, probability, num_shifts):
    model.cpu()  # -- PB
    print('INFO: model_name ', model_name)
    #print('model.modules():', model.modules())
    #print('model:', model)

    #######################################
    prunable_module_type = (nn.GRU, nn.LSTM, nn.Linear)

    prunable_modules = [m for m in model.modules() if isinstance(m, prunable_module_type)]

    num_blocks = 0
    num_conv_pruned = 0
    multiplier = probability  # prune_probs[0]

    for layer_to_prune in prunable_modules:
        if isinstance(layer_to_prune,  nn.Linear):
            num_layers = len(layer_to_prune.weight)
        else:
            num_layers = len(layer_to_prune.all_weights)

        #print('num_layers:', num_layers)
        for layer_idx in range (0,num_layers):
            if isinstance(layer_to_prune,  nn.Linear):
                num_gatesPerLayer = len(layer_to_prune.weight[layer_idx])
                if 'random' in sampling_method:
                    num = reset_prune_random(layer_to_prune.weight[layer_idx], multiplier)
                elif 'targeted' in sampling_method:
                    num = reset_prune_targeted(sample_shift, layer_to_prune.weight[layer_idx],multiplier, ranking_method, num_shifts)
                elif 'uniform' in sampling_method:
                     num = reset_prune_uniform(sample_shift, layer_to_prune.weight[layer_idx], multiplier, ranking_method, num_shifts)
                else:
                    print('ERROR: unrecognized sampling method for nn.Linear:', sampling_method)
                    num = 0
            else:
                num_gatesPerLayer = len(layer_to_prune.all_weights[layer_idx])
                # LSTM: input, forget, cell, and output gates
                # GRU:  reset, update, and new gates
                # print('weights for input, forget, cell, and output gates (num_gatesPerLayer):', num_gatesPerLayer)
                # weight = layer_to_prune.all_weights[0][0].detach().cpu().numpy()
                for gate_idx in range(0, num_gatesPerLayer):
                    # select a layer with conv
                    # if isinstance(layer_to_prune, nn.GRU):
                    if 'random' in sampling_method:
                        num = reset_prune_random(layer_to_prune.all_weights[layer_idx][gate_idx], multiplier)
                    elif 'targeted' in sampling_method:
                        num = reset_prune_targeted(sample_shift, layer_to_prune.all_weights[layer_idx][gate_idx],
                                                       multiplier, ranking_method, num_shifts)
                    elif 'uniform' in sampling_method:
                        num = reset_prune_uniform(sample_shift, layer_to_prune.all_weights[layer_idx][gate_idx],
                                                      multiplier, ranking_method, num_shifts)
                    else:
                        print('ERROR: unrecognized sampling method:', sampling_method)
                        num = 0

            num_conv_pruned += num
            num_blocks += 1

    print('num_blocks:', num_blocks)
    print('num_conv_pruned:', num_conv_pruned)

    return model


############################################
# methods for random sampling
def reset_prune_random(conv, pruned_prob):
    weight = conv.detach().cpu().numpy()
    out_channels = weight.shape[0]
    num_pruned = int(out_channels * pruned_prob)
    if num_pruned == out_channels:
        print('prune_conv_random: num_pruned is equal to number of output channels')
    prune_index = random.sample(list(range(out_channels)), num_pruned)
    for idx in prune_index:
        conv[idx].zero_()

    return num_pruned


############################################
# methods for targeted sampling with shift
def reset_prune_targeted(sample_shift, conv, pruned_prob, ranking_method,num_shifts):
    #print('INFO: len(conv):', len(conv))
    weight = conv.detach().cpu().numpy()
    # print('INFO: len(conv.all_weights):', len(conv.all_weights))
    # weight = conv.all_weights[0][0].detach().cpu().numpy()
    #print('INFO: weight:', weight)
    #weight = weight[0].detach().cpu().numpy()
    out_channels = weight.shape[0]
    #print('INFO: weight.shape:', weight.shape)
    #weight = conv.weight.detach().cpu().numpy()
    #out_channels = weight.shape[0]
    if 'L1' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.sum(np.abs(weight), axis=(1))
        else:
            rank_norm = np.abs(weight)
        #rank_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
    elif 'L2' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1))
        else:
            rank_norm = np.sum(np.abs(weight) * np.abs(weight))
        #rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1, 2, 3))
    elif 'Linf' in ranking_method:
        rank_norm = []
        for idx in range(weight.shape[0]):
            rank_norm.append(weight[idx].max())
    elif 'STDEV' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.std(weight, axis=(1))
        else:
            rank_norm = np.std(weight)
        #rank_norm = np.std(weight, axis=(1, 2, 3))
    else:
        print('ERROR: unrecognized ranking method:', ranking_method)
        return 0

    # filter_hist_filepath = 'C:/PeterB/Projects/TrojAI/python/trojai-pruning/scratch/hist_conv.csv'
    # with open(filter_hist_filepath, 'a') as fh:
    #     for j in range(len(L1_norm)):
    #         fh.write("{}, ".format(L1_norm[j]))
    #     fh.write("\n")

    num_pruned = int(out_channels * pruned_prob)
    if num_pruned == out_channels:
        print('prune_conv: num_pruned is equal to number of output channels')

    if num_pruned > 0:
        # print('INFO: conv num_pruned:', num_pruned)
        temp = np.argsort(rank_norm)
        len_temp = len(temp)
        first_sample = sample_shift * (len_temp // (num_shifts - 1))
        last_sample = first_sample + num_pruned
        if last_sample > len_temp:
            # print('debug: last_sample=', last_sample)
            first_sample = len_temp - num_pruned
            last_sample = len_temp

        prune_index = temp[first_sample:last_sample].tolist()

        for idx in prune_index:
            conv[idx].zero_()

    return num_pruned


############################################
# methods for uniform sampling with shift
def reset_prune_uniform(sample_shift, conv, pruned_prob, ranking_method,num_shifts):
    weight = conv.detach().cpu().numpy()
    out_channels = weight.shape[0]

    if 'L1' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.sum(np.abs(weight), axis=(1))
        else:
            rank_norm = np.abs(weight)
        # rank_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
    elif 'L2' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1))
        else:
            rank_norm = np.sum(np.abs(weight) * np.abs(weight))
        # rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1, 2, 3))
    elif 'Linf' in ranking_method:
        rank_norm = []
        for idx in range(weight.shape[0]):
            rank_norm.append(weight[idx].max())
    elif 'STDEV' in ranking_method:
        if len(weight.shape) > 1:
            rank_norm = np.std(weight, axis=(1))
        else:
            rank_norm = np.std(weight)
        # rank_norm = np.std(weight, axis=(1, 2, 3))
    else:
        print('ERROR: unrecognized ranking method:', ranking_method)
        return 0

    # filter_hist_filepath = 'C:/PeterB/Projects/TrojAI/python/trojai-pruning/scratch/hist_conv.csv'
    # with open(filter_hist_filepath, 'a') as fh:
    #     for j in range(len(L1_norm)):
    #         fh.write("{}, ".format(L1_norm[j]))
    #     fh.write("\n")

    num_pruned = int(out_channels * pruned_prob)
    if num_pruned == out_channels:
        print('reset_prune: num_pruned is equal to number of output channels')

    if num_pruned > 0:
        # print('INFO: conv num_pruned:', num_pruned)
        temp = np.argsort(rank_norm)
        len_temp = len(temp)
        # step = len_temp / num_pruned
        first_sample = sample_shift * (len_temp // (num_shifts - 1))
        last_sample = first_sample + num_pruned
        if last_sample > len_temp:
            # print('debug: last_sample=', last_sample)
            first_sample = len_temp - num_pruned
            last_sample = len_temp

        prune_index = temp[first_sample:last_sample].tolist()

        for idx in prune_index:
            conv[idx].zero_()

    return num_pruned
