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

import gc
import sys

import torch
import torch.nn as nn
import transformers.models as tm
import random
import numpy as np

#import trojan_detector_ner

"""
This class is designed to reset nodes to zero in a graph defining an AI model 
(input, forget, cell, and output gates for round 5 - GRU and LSTM architectures).
"""


##################################################################
def reset_prune_model(model, device, model_name, sample_shift, sampling_method, ranking_method, layer_probability, node_probability, num_shifts):
    print('INFO: model_name ', model_name)
    #print('model.modules():', model.modules())
    #print('model:', model)

    #######################################
    prunable_module_type = (nn.Linear) # nn.Transformer, nn.LayerNorm, nn.Dropout,

    prunable_modules = [m for m in model.modules() if isinstance(m, prunable_module_type)]

    num_blocks = 0
    num_conv_pruned = 0
    #multiplier = node_probability  # prune_probs[0]

    print('INFO: number of layers to prune:', len(prunable_modules))
    index = 0
    for layer_to_prune in prunable_modules:
        #print(index, '.', end='', flush=True)
        #trojan_detector_ner.memory_stats("DEBUG: pruning")
        # print('DEBUG:   gc.get_referrers(layer_to_prune):', gc.get_referrers(layer_to_prune))
        # print('DEBUG:   gc.get_referents(layer_to_prune):', gc.get_referents(layer_to_prune))
        # print('DEBUG: sys.getrefcount(layer_to_prune):', sys.getrefcount(layer_to_prune))

        #print('INFO: layer_to_prune:', layer_to_prune)
        index = index + 1
        if isinstance(layer_to_prune,  nn.Linear):
            num_layers = len(layer_to_prune.weight)
        else:
            num_layers = len(layer_to_prune.all_weights)

        # print('num_layers:', num_layers)
        # print('DEBUG: gc.get_stats() :', gc.get_stats())

        num = 0
        if isinstance(layer_to_prune, nn.Linear):

            # use L1 norm to sort the layers !!!!
            if device != 'cpu':
                # GPU implementation
                if len(layer_to_prune.weight.shape) > 1:
                    rank_layer_norm = torch.sum(torch.abs(layer_to_prune.weight), dim=1)
                else:
                    rank_layer_norm = torch.abs(layer_to_prune.weight)

                temp = torch.argsort(rank_layer_norm)
                len_temp = temp.shape[0]
            else:
                # CPU implementation
                peter = layer_to_prune.weight.detach().numpy()
                if len(layer_to_prune.weight.shape) > 1:
                    rank_layer_norm = np.sum(np.abs(peter), axis=(1))
                else:
                    rank_layer_norm = np.abs(peter)

                temp = np.argsort(rank_layer_norm)
                len_temp = len(temp)

            # prune % of the layers and decide on layers based on sample_shift
            #layer_probability = 0.005
            num_layer_pruned = layer_probability * num_layers
            # change from sample_shift to the last sample_shift since argsort delivers from the smaller to the largest values
            #first_layer = sample_shift * (len_temp // (num_shifts - 1))
            first_layer = (num_shifts - 1) * (len_temp // (num_shifts - 1))
            last_layer = first_layer + num_layer_pruned
            if last_layer > len_temp:
                # print('debug: last_sample=', last_sample)
                first_layer = len_temp - num_layer_pruned
                last_layer = len_temp

            if first_layer < 0:
                print('ERROR: first_layer < 0: ', first_layer)
                first_layer = 0
                #return num_layer_pruned

            #print('INFO: layer_index:', index, ' first_layer:', first_layer, ' last_layer:', last_layer)
            #############################################

            for layer_idx in range(int(first_layer), int(last_layer)):
            #for layer_idx in range(0, num_layers):

                if 'random' in sampling_method:
                    num = reset_prune_random(layer_to_prune.weight[temp[layer_idx]], node_probability, device)
                elif 'targeted' in sampling_method:
                    # trojan_detector_ner.memory_stats('DEBUG: before reset_prune_targeted')
                    # tracked = gc.get_objects()  # Start tracking the collected objects
                    # print("DEBUG START: Number of objects tracked: ", len(tracked))  # Number of tracked objects
                    num = reset_prune_targeted(sample_shift, layer_to_prune.weight[temp[layer_idx]], node_probability,
                                               ranking_method,
                                               num_shifts, device)
                    # trojan_detector_ner.memory_stats('DEBUG: after reset_prune_targeted')
                    # tracked_end = gc.get_objects()  # Start tracking the collected objects
                    # print("DEBUG END: Number of objects tracked: ", len(tracked_end))  # Number of tracked objects
                elif 'uniform' in sampling_method:
                    num = reset_prune_uniform(sample_shift, layer_to_prune.weight[temp[layer_idx]], node_probability,
                                              ranking_method,
                                              num_shifts, device)
                else:
                    print('ERROR: unrecognized sampling method for nn.Linear:', sampling_method)
                    num = 0
        else:
            print('ERROR: unsupported module in the Round 7 model different from Linear')

        num_conv_pruned += num
        num_blocks += 1

    del prunable_modules
    print('num_blocks:', num_blocks)
    print('num_conv_pruned:', num_conv_pruned)

    return model


############################################
# methods for random sampling
def reset_prune_random(conv, pruned_prob, device):

    if device == 'cpu':
        weight = conv.detach().cpu().numpy()

    if device != 'cpu':
        #GPU implementation
        out_channels = conv.shape[0]
        num_dim = len(conv.shape)
    else:
        out_channels = weight.shape[0]
        num_dim = len(weight.shape)

    num_pruned = int(out_channels * pruned_prob)
    if num_pruned == out_channels:
        print('prune_conv_random: num_pruned is equal to number of output channels')

    prune_index = random.sample(list(range(out_channels)), num_pruned)
    if device != 'cpu':
        for idx in prune_index:
            conv[idx].zero_()
    else:
        for idx in prune_index:
            conv[idx].data = torch.tensor(0.0).data

    del prune_index
    if device == 'cpu':
        del weight

    return num_pruned


############################################
# methods for targeted sampling with shift
def reset_prune_targeted(sample_shift, conv, pruned_prob, ranking_method,num_shifts, device):
    #gc.set_debug(gc.DEBUG_LEAK)
    #print('INFO: len(conv):', len(conv))
    # print('DEBUG: before detach reset_prune_targeted: gc.get_stats() :', gc.get_stats())
    # trojan_detector_ner.memory_stats('DEBUG: before detach in reset_prune_targeted')
    # tracked = gc.get_objects()  # Start tracking the collected objects
    # print("DEBUG START: Number of objects tracked: ", len(tracked))  # Number of tracked objects

    if device == 'cpu':
        weight = conv.detach().numpy()
    # print('DEBUG: after detach reset_prune_targeted: gc.get_stats() :', gc.get_stats())
    # trojan_detector_ner.memory_stats('DEBUG: after detach in reset_prune_targeted')
    #weight = conv.detach().cpu().numpy()
    # print('INFO: len(conv.all_weights):', len(conv.all_weights))
    # weight = conv.all_weights[0][0].detach().cpu().numpy()
    #print('INFO: weight:', weight)
    #weight = weight[0].detach().cpu().numpy()

    if device != 'cpu':
        #GPU implementation
        out_channels = conv.shape[0]
        num_dim = len(conv.shape)
    else:
        out_channels = weight.shape[0]
        num_dim = len(weight.shape)


    # print('INFO: weight.shape:', weight.shape)
    # print('conv.shape:', conv.shape)
    #weight = conv.weight.detach().cpu().numpy()
    #out_channels = weight.shape[0]

    if 'L1' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                #GPU implementation
                rank_norm = torch.sum(torch.abs(conv), dim=1)
            else:
                rank_norm = np.sum(np.abs(weight), axis=(1))

        else:
            if device != 'cpu':
                #GPU implementation
                rank_norm = torch.abs(conv)
            else:
                rank_norm = np.abs(weight)

        #rank_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
    elif 'L2' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                #GPU implementation
                rank_norm = torch.sum(torch.abs(conv) * torch.abs(conv), dim=1)
            else:
                rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1))

        else:
            if device != 'cpu':
                #GPU implementation
                rank_norm = torch.abs(conv) * torch.abs(conv)
            else:
                rank_norm = np.abs(weight) * np.abs(weight)

        #rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1, 2, 3))
    elif 'Linf' in ranking_method:
        rank_norm = []
        for idx in range(out_channels):
            if device != 'cpu':
                #GPU implementation
                rank_norm.append(conv[idx].max())
            else:
                rank_norm.append(weight[idx].max())

    elif 'STDEV' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                #GPU implementation
                stdev, mean = torch.std_mean(conv,dim=1, unbiased=False)
                rank_norm = stdev
            else:
                rank_norm = np.std(weight, axis=(1))

        else:
            if device != 'cpu':
                #GPU implementation
                stdev, mean = torch.std_mean(conv,unbiased=False)
                rank_norm = stdev
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
        if device != 'cpu':
            #GPU implementation
            temp = torch.argsort(rank_norm)
            len_temp = temp.shape[0]
        else:
            temp = np.argsort(rank_norm)
            len_temp = len(temp)

        # print('DEBUG: after argsort reset_prune_targeted: gc.get_stats() :', gc.get_stats())
        # trojan_detector_ner.memory_stats('DEBUG: after argsort in reset_prune_targeted')


        first_sample = sample_shift * (len_temp // (num_shifts - 1))
        last_sample = first_sample + num_pruned
        if last_sample > len_temp:
            # print('debug: last_sample=', last_sample)
            first_sample = len_temp - num_pruned
            last_sample = len_temp

        if first_sample < 0:
            print('ERROR: first_sample < 0: ', first_sample)
            return num_pruned
        #prune_index = temp[first_sample:last_sample].tolist()
        # print('DEBUG: after prune_index = temp reset_prune_targeted: gc.get_stats() :', gc.get_stats())
        # trojan_detector_ner.memory_stats('DEBUG: after prune_index = temp  in reset_prune_targeted')
        # for idx in prune_index:
        #     #conv[idx].zero_()
        #     conv[idx] = 0.0

        if device != 'cpu':
            for idx in range(first_sample, last_sample):
                conv[temp[idx]].zero_()
        else:
            for idx in range(first_sample, last_sample):
                conv[temp[idx]].data = torch.tensor(0.0).data

        # print('DEBUG: after conv[idx].zero_() reset_prune_targeted: gc.get_stats() :', gc.get_stats())
        # trojan_detector_ner.memory_stats('DEBUG: after conv[idx].zero_()  in reset_prune_targeted')

        # del prune_index
        del temp

    del rank_norm
    # del weight
    # out_channels = None
    # first_sample = None
    # last_sample = None
    # len_temp = None
    # idx = None

    gc.collect()
    # print('DEBUG:   gc.get_referrers(conv):',   gc.get_referrers(conv))
    # print('DEBUG:   gc.get_referents(conv):', gc.get_referents(conv))
    # print('DEBUG: sys.getrefcount(conv):', sys.getrefcount(conv))
    # print('DEBUG: reset_prune_targeted: gc.get_stats() :', gc.get_stats(3))
    # trojan_detector_ner.memory_stats('DEBUG: after all in reset_prune_targeted')
    # print('DEBUG: sizeof(idx):', sys.getsizeof(idx))
    # tracked_end = gc.get_objects()                  # Start tracking the collected objects
    # print("DEBUG END: Number of objects tracked: ", len(tracked_end))     # Number of tracked objects

    return num_pruned


############################################
# methods for uniform sampling with shift
def reset_prune_uniform(sample_shift, conv, pruned_prob, ranking_method,num_shifts, device):
    # TODO: the uniform sampling must be checked!!!!

    if device == 'cpu':
        weight = conv.detach().numpy()

    if device != 'cpu':
        # GPU implementation
        out_channels = conv.shape[0]
        num_dim = len(conv.shape)
    else:
        out_channels = weight.shape[0]
        num_dim = len(weight.shape)


    if 'L1' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                # GPU implementation
                rank_norm = torch.sum(torch.abs(conv), dim=1)
            else:
                rank_norm = np.sum(np.abs(weight), axis=(1))

        else:
            if device != 'cpu':
                # GPU implementation
                rank_norm = torch.abs(conv)
            else:
                rank_norm = np.abs(weight)

    elif 'L2' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                # GPU implementation
                rank_norm = torch.sum(torch.abs(conv) * torch.abs(conv), dim=1)
            else:
                rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1))

        else:
            if device != 'cpu':
                # GPU implementation
                rank_norm = torch.abs(conv) * torch.abs(conv)
            else:
                rank_norm = np.abs(weight) * np.abs(weight)

    elif 'Linf' in ranking_method:
        rank_norm = []
        for idx in range(out_channels):
            if device != 'cpu':
                # GPU implementation
                rank_norm.append(conv[idx].max())
            else:
                rank_norm.append(weight[idx].max())

    elif 'STDEV' in ranking_method:
        if num_dim > 1:
            if device != 'cpu':
                # GPU implementation
                stdev, mean = torch.std_mean(conv, dim=1, unbiased=False)
                rank_norm = stdev
            else:
                rank_norm = np.std(weight, axis=(1))

        else:
            if device != 'cpu':
                # GPU implementation
                stdev, mean = torch.std_mean(conv, unbiased=False)
                rank_norm = stdev
            else:
                rank_norm = np.std(weight)

    else:
        print('ERROR: unrecognized ranking method:', ranking_method)
        return 0

    num_pruned = int(out_channels * pruned_prob)
    if num_pruned == out_channels:
        print('prune_conv: num_pruned is equal to number of output channels')

    if num_pruned > 0:
        # print('INFO: conv num_pruned:', num_pruned)
        if device != 'cpu':
            # GPU implementation
            temp = torch.argsort(rank_norm)
            len_temp = temp.shape[0]
        else:
            temp = np.argsort(rank_norm)
            len_temp = len(temp)


        first_sample = sample_shift * (len_temp // (num_shifts - 1))
        last_sample = first_sample + num_pruned
        if last_sample > len_temp:
            # print('debug: last_sample=', last_sample)
            first_sample = len_temp - num_pruned
            last_sample = len_temp

        if first_sample < 0:
            print('ERROR: first_sample < 0: ', first_sample)
            return num_pruned

        if device != 'cpu':
            for idx in range(first_sample, last_sample):
                conv[temp[idx]].zero_()
        else:
            for idx in range(first_sample, last_sample):
                conv[temp[idx]].data = torch.tensor(0.0).data


        del temp

    del rank_norm
    gc.collect()

    # weight = conv.detach().cpu().numpy()
    # out_channels = weight.shape[0]
    #
    # if 'L1' in ranking_method:
    #     if len(weight.shape) > 1:
    #         rank_norm = np.sum(np.abs(weight), axis=(1))
    #     else:
    #         rank_norm = np.abs(weight)
    #     # rank_norm = np.sum(np.abs(weight), axis=(1, 2, 3))
    # elif 'L2' in ranking_method:
    #     if len(weight.shape) > 1:
    #         rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1))
    #     else:
    #         rank_norm = np.sum(np.abs(weight) * np.abs(weight))
    #     # rank_norm = np.sum(np.abs(weight) * np.abs(weight), axis=(1, 2, 3))
    # elif 'Linf' in ranking_method:
    #     rank_norm = []
    #     for idx in range(weight.shape[0]):
    #         rank_norm.append(weight[idx].max())
    # elif 'STDEV' in ranking_method:
    #     if len(weight.shape) > 1:
    #         rank_norm = np.std(weight, axis=(1))
    #     else:
    #         rank_norm = np.std(weight)
    #     # rank_norm = np.std(weight, axis=(1, 2, 3))
    # else:
    #     print('ERROR: unrecognized ranking method:', ranking_method)
    #     return 0

    # filter_hist_filepath = 'C:/PeterB/Projects/TrojAI/python/trojai-pruning/scratch/hist_conv.csv'
    # with open(filter_hist_filepath, 'a') as fh:
    #     for j in range(len(L1_norm)):
    #         fh.write("{}, ".format(L1_norm[j]))
    #     fh.write("\n")
    #
    # num_pruned = int(out_channels * pruned_prob)
    # if num_pruned == out_channels:
    #     print('reset_prune: num_pruned is equal to number of output channels')
    #
    # if num_pruned > 0:
    #     # print('INFO: conv num_pruned:', num_pruned)
    #     temp = np.argsort(rank_norm)
    #     len_temp = len(temp)
    #     # step = len_temp / num_pruned
    #     first_sample = sample_shift * (len_temp // (num_shifts - 1))
    #     last_sample = first_sample + num_pruned
    #     if last_sample > len_temp:
    #         # print('debug: last_sample=', last_sample)
    #         first_sample = len_temp - num_pruned
    #         last_sample = len_temp
    #
    #     # prune_index = temp[first_sample:last_sample].tolist()
    #     # for idx in prune_index:
    #     #     conv[idx].zero_()
    #
    #     for idx in range(first_sample, last_sample):
    #         conv[temp[idx]].data = torch.tensor(0.0).data
    #
    #     del temp
    #
    # del rank_norm
    # del weight

    return num_pruned
