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

import copy
import statistics
import torch
import torch.nn as nn
import random
import argparse
import os, sys
import skimage.io
import numpy as np
import torchvision
import csv
import time

from linear_regression import read_regression_coefficients, linear_regression_prediction
from model_classifier_round2 import model_classifier
from my_dataset import my_dataset
from remove_prune import prune_model
from reset_prune import reset_prune_model
from trim_prune import trim_model

"""
This class is designed for detecting trojans in TrojAI Round 2 Challenge datasets
see https://pages.nist.gov/trojai/docs/data.html#round-2
This code is an adjusted version of the trojan detector for the Round 1 of the TrojAI challenge
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def eval(model, test_loader, result_filepath, model_name):
    correct = 0.0
    total = 0.0

    if torch.cuda.is_available():
        use_cuda = True
        model.cuda()
    else:
        use_cuda = False

    model.eval()
    with torch.no_grad():
        # for i, (img, target) in enumerate(test_loader):
        for i in range(len(test_loader.dataset.labels)):
            target = test_loader.dataset.labels[i]
            # target = torch.IntTensor(target, device=device)
            # prepare input batch
            # read the image (using skimage)
            img = skimage.io.imread(test_loader.dataset.list_IDs[i])
            img = preprocess_round2(img, model_name)
            # convert image to a gpu tensor

            if use_cuda:
                batch_data = torch.FloatTensor(img).cuda(non_blocking=True)
            else:
                batch_data = torch.FloatTensor(img)

            # batch_data = batch_data.to(device)
            out = model(batch_data)
            pred = out.max(1)[1].detach().cpu().numpy()
            # enable for testing PB
            # print('INFO: test_loader.dataset.labels[', i, ']=', test_loader.dataset.labels[i])
            # print('INFO: test_loader.filename[', i, ']=', test_loader.dataset.list_IDs[i])
            # print('INFO: out:', out, ' pred:', pred[0], ' target:', target)
            # target = target.cpu().numpy()
            correct += (pred[0] == target).sum()
            total += 1  # len(target)

            # enable for testing PB
            # with open(result_filepath, 'a') as fh:
            #     fh.write("input file: {}, ".format(test_loader.dataset.list_IDs[i]))
            #     fh.write("input file label: {}, ".format(target))
            #     fh.write("model prediction: {}, ".format(pred[0]))
            #     fh.write("model predicted vector: {} \n ".format(out))

        return correct / float(total)


def get_dataloader(dataset):
    # train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=1)
    # train_loader = torch.utils.data.DataLoader(
    #     CIFAR10('./data', train=True, transform=transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #     ]), download=True), batch_size=args.batch_size, num_workers=2)
    # test_loader = torch.utils.data.DataLoader(
    #     CIFAR10('./data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #     ]), download=True), batch_size=args.batch_size, num_workers=2)
    return test_loader

    ################################################################


def preprocess_round2(img, model_name):
    # code from trojai example
    # perform center crop to what the CNN is expecting 224x224
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy + 224, dx:dx + 224, :]

    # If needed: convert to BGR
    # r = img[:, :, 0]
    # g = img[:, :, 1]
    # b = img[:, :, 2]
    # img = np.stack((b, g, r), axis=2)

    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = np.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = np.expand_dims(img, 0)
    # normalize the image
    img = img - np.min(img)
    img = img / np.max(img)
    # convert image to a gpu tensor
    # batch_data = torch.FloatTensor(img)

    # if 'inception' in model_name:
    #     if not np.array_equal(img.shape, (1, 3, 299, 299)):
    #         img = np.resize(img, (1, 3, 299, 299))
    # if 'resnet' in model_name:
    #     if not np.array_equal(img.shape, (1, 3, 224, 224)):
    #         img = np.resize(img, (1, 3, 224, 224))
    # if 'densenet' in model_name:
    #     if not np.array_equal(img.shape, (1, 3, 224, 224)):
    #         img = np.resize(img, (1, 3, 224, 224))

    return img


def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    print('model_filepath = {}'.format(model_filepath))
    print('result_fileRoot = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    print('example_img_format = {}'.format(example_img_format))

    # start timer
    start = time.perf_counter()
    #######################################
    # adjust to the hardware platform
    mydevice = "cpu"  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # test only
    # t = torch.tensor(range(15), device=mydevice)
    # print('torch HW platform test with a tensor: %s \n' % (t))
    ############################
    # to avoid messages about serialization on cpu
    torch.nn.Module.dump_patches = 'False'
    #################################

    # read the ground truth label
    model_dirpath = os.path.dirname(model_filepath)
    gt_model_label_filepath = os.path.join(model_dirpath, 'ground_truth.csv')
    gt_model_label = -1
    if os.path.isfile(gt_model_label_filepath):
        # read the label
        with open(gt_model_label_filepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                print('ground_truth.csv - is poisoned?:', row[0])
                gt_model_label = int(row[0])
    else:
        print('missing ground truth label for the model')
        gt_model_label = -1

    ##################
    # decide which model architecture is represented by the provided AI Model
    a = model_classifier(model_filepath)
    # determine the AI model architecture based on loaded class type
    model_class_str, model_architecture = a.classify_architecture(mydevice)
    model_name, model_type, min_model_size_delta = a.classify_type(model_architecture)
    print('model_type: %s\n' % model_type)
    print('file size delta between a model and the reference model: %s\n' % min_model_size_delta)
    model_name = a.switch_architecture(model_type)
    print('classified the model as:\t', model_name)
    print('model size: \t', a.model_size)
    ref_model_size = a.model_size + min_model_size_delta
    print('reference model size: \t', ref_model_size)

    #####################
    # save the file size and model name information
    scratch_filepath = os.path.join(scratch_dirpath, model_name + '_log.csv')
    with open(scratch_filepath, 'a') as fh:
        fh.write("{}, ".format(model_name))
        fh.write("model_filepath, {}, ".format(model_filepath))
        fh.write("model_size, {}, ".format(a.model_size))
        fh.write("ref_model_size, {:.4f}, ".format(ref_model_size))
        fh.write("delta_model_size, {:.4f}, ".format(min_model_size_delta))
        fh.write("gt_model_label, {}, ".format(gt_model_label))

    #################################
    # TODO these are the parameters that must be configured
    pruning_method = 'reset'  # remove or reset or trim
    sampling_method = 'targeted'  # random or targeted or uniform sampling
    ranking_method = 'L1'  # L1, L2, Linf, STDEV
    num_samples = 15  # nS=5 or  10 or 15 or 25 was tested
    # set the number of images used
    num_images_used = 10  # nD=10,20,30,40 was tested
    print('pruning_method (PM):', pruning_method, ' sampling method (SM):', sampling_method, ' ranking method (RM):',
          ranking_method)
    print('num_samples (nS):', num_samples, ' num_images_used (nD):', num_images_used)

    # adjustments per model type for trim method
    trim_pruned_amount = 0.5
    if 'trim' in pruning_method:
        sampling_probability = 4.0 * trim_pruned_amount * np.ceil(
            1000.0 / num_samples) / 1000  # 0.2 # 1.0/num_samples #0.2 # before 0.4

    # these values are computed from each architecture by 1/(min number of filters per layer) - rounded up at the second decimal
    # this guarantees that at least one filter is removed from each layer
    min_one_filter = {"shufflenet1_0" :0.05,	"shufflenet1_5" :0.05,	"shufflenet2_0" :0.05,
                          "inceptionv1(googlenet)" :0.07,	"inceptionv3" :0.04,	"resnet18" :0.03,
                          "resnet34" :0.03,	"resnet50" :0.03,	"resnet101" :0.03,	"resnet152" :0.03,
                         "wide_resnet50" :0.03,	"wide_resnet101" :0.03,
                          "squeezenetv1_0" :0.21,	"squeezenetv1_1" :0.15,	"mobilenetv2" :0.07,
                          "densenet121" :0.04,"densenet161" :0.03,	"densenet169" :0.04,	"densenet201" :0.04,
                          "vgg11_bn" :0.03, "vgg13_bn" :0.03,	"vgg16_bn" :0.03}

    if 'reset' in pruning_method:
        sampling_probability = np.ceil(1000.0/num_samples)/1000

    # adjustments per model type for remove method
    if 'remove' in pruning_method:
        if num_samples <= 5:
            for key, value in min_one_filter.items():
                print(key, value)
                if key in model_name:
                    sampling_probability = value
        else:
            # this is the setup for nS>6
            sampling_probability = np.ceil(1000.0/num_samples)/1000

        # there is a random failure of densenet models for sampling_probability larger than 0.02
        if 'densenet' in model_name and sampling_probability > 0.02:
            # this value is computed as if for num_samples = 50 --> sampling_probability = 0.02
            sampling_probability = 0.02

    # TEST
    # filter_hist_filepath = './number_filters.csv'
    # with open(filter_hist_filepath, 'a') as fh:
    #     fh.write("model_name,{}, ".format(model_name))
    #     fh.write("pruning_method,{}, ".format(pruning_method))
    #     fh.write("ranking_method,{}, ".format(ranking_method))
    #     fh.write("sampling_method,{}, ".format(sampling_method))
    #     fh.write("sampling_probability,{}, ".format(sampling_probability))
    #     fh.write("\n")



    ##########################################
    # Temporary hack for the prune method = remove because  the shufflenet architecture is not supported
    if 'remove' in pruning_method and 'shufflenet' in model_name:
        scratch_filepath = os.path.join(scratch_dirpath, model_name + '_log.csv')
        prob_trojan_in_model = 0.5
        with open(scratch_filepath, 'a') as fh:
            # fh.write("model_filepath, {}, ".format(model_filepath))
            fh.write("number of params, {}, ".format(0))
            fh.write("{}, ".format(model_name))
            fh.write("{}, ".format(pruning_method))
            fh.write("{}, ".format(sampling_method))
            fh.write("{}, ".format(ranking_method))
            fh.write("{}, ".format(num_samples))
            fh.write("{}, ".format(num_images_used))
            fh.write("{:.4f}, ".format(sampling_probability))
            for i in range(num_samples):
                fh.write("{:.4f}, ".format(0))

            fh.write("mean, {:.4f}, ".format(0))
            fh.write("stdev, {:.4f}, ".format(0))
            fh.write("min, {:.4f}, ".format(0))
            fh.write("max, {:.4f}, ".format(0))
            fh.write("coef_var, {:.4f}, ".format(0))
            fh.write("num_min2max_ordered, {}, ".format(0))
            fh.write("num_max2min_ordered, {}, ".format(0))
            fh.write("slope, {:.4f}, ".format(0))
            fh.write("prob_trojan_in_model, {:.4f}, ".format(prob_trojan_in_model))
            fh.write("execution time [s], {}, \n".format((0)))

        # write the result to a file
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(prob_trojan_in_model))

        return prob_trojan_in_model
    ##############################################################
    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if
           fn.endswith(example_img_format)]
    # if len(fns) > 10:
    #     fns = fns[0:5]
    #

    # sort the list of files to assure reproducibility across operating systems
    fns.sort()
    num_images_avail = len(fns)


    with open(scratch_filepath, 'a') as fh:
        fh.write("num_images_avail, {}, ".format(num_images_avail))
        fh.write("num_images_used, {}, ".format(num_images_used))

    print('number of images available for eval per model:', num_images_avail)

    if num_images_avail < num_images_used:
        num_images_used = num_images_avail
        print('WARNING: ', num_images_avail, ' is less than ', num_images_used)
        print(
            'WARNING: this should never happen for round 2 since there are 5-25 classes per model and 10-20 images per class')

    step = num_images_avail // num_images_used
    temp_idx = []
    for i in range(step // 2, num_images_avail, step):
        if len(temp_idx) < num_images_used:
            temp_idx.append(i)

    fns = [fns[i] for i in temp_idx]
    print('selected images:', fns)

    # prepare data list for evaluations
    mydata = {}
    mydata['test'] = my_dataset(fns)
    test_loader = get_dataloader(mydata['test'])

    #######################################
    # load a model
    try:
        model_orig = torch.load(model_filepath, map_location=mydevice)
    except:
        print("Unexpected loading error:", sys.exc_info()[0])
        # close the line
        with open(scratch_filepath, 'a') as fh:
            fh.write("\n")
        raise

    # model = torch.nn.modules.container.Sequential.cpu(model)
    # the eval is needed to set the model for inferencing
    # model.eval()
    params = sum([np.prod(p.size()) for p in model_orig.parameters()])
    print("Before Number of Parameters: %.1fM" % (params / 1e6))
    acc_model = 1.0  # eval(model_orig, test_loader, result_filepath, model_name)
    print("Before Acc=%.4f\n" % (acc_model))
    # with open(scratch_filepath, 'a') as fh:
    #     fh.write("model_filepath: {}, ".format(model_filepath))
    #     fh.write("model_name: {}, ".format(model_name))
    #     fh.write("original number of params: {}, ".format((params / 1e6)))
    #     fh.write("original model accuracy: {} \n".format(acc_model))

    #####################################
    # prepare the model and transforms
    # TODO check is setting the model variable here works
    if 'googlenet' in model_name or 'inception' in model_name:
        model_orig.aux_logits = False
    elif 'fcn' in model_name or 'deeplabv3' in model_name:
        model_orig.aux_loss = None

    if 'fcn' in model_name or 'deeplabv3' in model_name:
        output_transform = lambda x: x['out']
    else:
        output_transform = None

    print(model_name)

    acc_pruned_model_shift = []
    pruning_shift = []

    timings = dict()
    copy_times = list()
    prune_times = list()
    eval_times = list()

    loop_start = time.perf_counter()

    for sample_shift in range(num_samples):
        copy_start = time.perf_counter()
        model = copy.deepcopy(model_orig)
        copy_time = time.perf_counter() - copy_start

        copy_times.append(copy_time)


        print('INFO: reset- pruning for sample_shift:', sample_shift)
        prune_start = time.perf_counter()
        try:
            if 'remove' in pruning_method:
                prune_model(model, model_name, output_transform, sample_shift, sampling_method, ranking_method,
                            sampling_probability, num_samples)
            if 'reset' in pruning_method:
                reset_prune_model(model, model_name, sample_shift, sampling_method, ranking_method, sampling_probability,
                                  num_samples)
            if 'trim' in pruning_method:
                trim_model(model, model_name, sample_shift, sampling_method, ranking_method, sampling_probability,
                           num_samples, trim_pruned_amount)

        except:
            # this is relevant to PM=Remove because it fails for some configurations to prune the model correctly
            print("Unexpected pruning error:", sys.exc_info()[0])
            # close the line
            with open(scratch_filepath, 'a') as fh:
                fh.write("\n")
            raise

        prune_time = time.perf_counter() - prune_start
        prune_times.append(prune_time)

        eval_start = time.perf_counter()
        acc_pruned_model = eval(model, test_loader, result_filepath, model_name)
        eval_time = time.perf_counter() - eval_start
        eval_times.append(eval_time)
        print('model: ', model_filepath, ' acc_model: ', acc_model, ' acc_pruned_model: ', acc_pruned_model)
        acc_pruned_model_shift.append(acc_pruned_model)
        pruning_shift.append(sample_shift)
        del model

        print('Timing info: copy: {}, prune: {}, eval: {}'.format(copy_time, prune_time, eval_time))

    loop_time = time.perf_counter() - loop_start
    timings['loop'] = loop_time
    timings['avg copy'] = statistics.mean(copy_times)
    timings['avg prune'] = statistics.mean(prune_times)
    timings['max eval'] = max(eval_times)
    timings['min eval'] = min(eval_times)
    timings['avg eval'] = statistics.mean(eval_times)


    print('Loop timing: {}'.format(loop_time))

    # compute simple stats of the measured signal (vector of accuracy values over a set of pruned models)
    mean_acc_pruned_model = statistics.mean(acc_pruned_model_shift)
    mean_pruning_shift = statistics.mean(pruning_shift)
    slope = 0.0
    denominator = 0.0
    for i in range(len(pruning_shift)):
        slope += (pruning_shift[i] - mean_pruning_shift) * (acc_pruned_model_shift[i] - mean_acc_pruned_model)
        denominator += (pruning_shift[i] - mean_pruning_shift) * (pruning_shift[i] - mean_pruning_shift)

    slope = slope / denominator
    print('INFO: slope:', slope)

    stdev_acc_pruned_model = statistics.stdev(acc_pruned_model_shift)
    min_acc_pruned_model = min(acc_pruned_model_shift)
    max_acc_pruned_model = max(acc_pruned_model_shift)
    print('mean_acc_pruned_model:', mean_acc_pruned_model, ' stdev_acc_pruned_model:', stdev_acc_pruned_model)
    print('min_acc_pruned_model:', min_acc_pruned_model, ' max_acc_pruned_model:', max_acc_pruned_model)

    # the samples should be ordered from the largest accuracy to the smallest accuracy
    # since the pruning is removing the smallest L1 norm to the largest L1 norm
    num_min2max_ordered = 0
    num_max2min_ordered = 0
    for i in range(len(acc_pruned_model_shift) - 1):
        if acc_pruned_model_shift[i] < acc_pruned_model_shift[i + 1]:
            num_min2max_ordered += 1
        if acc_pruned_model_shift[i] > acc_pruned_model_shift[i + 1]:
            num_max2min_ordered += 1

    # low coefficient of variation could indicate that a trojan might be present
    if mean_acc_pruned_model > 0.01:
        coef_var = stdev_acc_pruned_model / mean_acc_pruned_model
    else:
        coef_var = 0.0

    prob_trojan_in_model = coef_var
    if prob_trojan_in_model < 0:
        prob_trojan_in_model = 0

    if prob_trojan_in_model > 1.0:
        prob_trojan_in_model = 1.0
    print('coef of variation:', coef_var, ' prob_trojan_in_model:', prob_trojan_in_model)

    # round 4 - linear regression coefficients applied to the num_samples (signal measurement)
    # this function should be enabled if  the estimated multiple linear correlation coefficients should be applied
    if num_samples == 15 and 'reset' in pruning_method and 'L1' in ranking_method and 'targeted' in sampling_method:
        print('Applying existing model r4_reset_L1_targeted_15_10_0p067.csv')
        linear_regression_filepath = './linear_regression_data/r4_reset_L1_targeted_15_10_0p067.csv'
        trained_coef = read_regression_coefficients(linear_regression_filepath, model_name)
        prob_trojan_in_model = linear_regression_prediction(trained_coef, acc_pruned_model_shift)

    # stop timing the execution
    end = time.perf_counter()

    with open(scratch_filepath, 'a') as fh:
        # fh.write("model_filepath, {}, ".format(model_filepath))
        fh.write("number of params, {}, ".format((params / 1e6)))
        fh.write("{}, ".format(model_name))
        fh.write("{}, ".format(pruning_method))
        fh.write("{}, ".format(sampling_method))
        fh.write("{}, ".format(ranking_method))
        fh.write("{}, ".format(num_samples))
        fh.write("{}, ".format(num_images_used))
        fh.write("{:.4f}, ".format(sampling_probability))
        for i in range(len(acc_pruned_model_shift)):
            fh.write("{:.4f}, ".format(acc_pruned_model_shift[i]))

        fh.write("mean, {:.4f}, ".format(mean_acc_pruned_model))
        fh.write("stdev, {:.4f}, ".format(stdev_acc_pruned_model))
        fh.write("min, {:.4f}, ".format(min_acc_pruned_model))
        fh.write("max, {:.4f}, ".format(max_acc_pruned_model))
        fh.write("coef_var, {:.4f}, ".format(coef_var))
        fh.write("num_min2max_ordered, {}, ".format(num_min2max_ordered))
        fh.write("num_max2min_ordered, {}, ".format(num_max2min_ordered))
        fh.write("slope, {:.4f}, ".format(slope))
        fh.write("prob_trojan_in_model, {:.4f}, ".format(prob_trojan_in_model))
        fh.write("execution time [s], {}, \n".format((end - start)))
        for key in timings.keys():
            fh.write('{} [s], {},'.format(key, timings[key]))
        fh.write('\n')


    # write the result to a file
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(prob_trojan_in_model))

    del model_orig
    del fns
    return prob_trojan_in_model


####################################################################################
if __name__ == '__main__':
    entries = globals().copy()

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        required=True)
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        required=True)
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        required=True)
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                        required=False)

    args = parser.parse_args()
    print('args %s \n %s \n %s \n %s \n' % (
        args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath))

    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
