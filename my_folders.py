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

import os
import time

import torch
import numpy as np
import trojan_detector_round1 as td
import tarfile
import csv
from model_classifier import model_classifier
from guppy import hpy
import threading

"""
This class is designed to loop over a folders containing AI models and training images provided
for the TrojAI Round 1 Challenge - see https://pages.nist.gov/trojai/docs/data.html#round-1
"""


def batch_model_classifier(model_dirpath, result_filepath, scratch_dirpath, model_format='.pt', example_img_format='png'):
    print('model_dirpath = {}'.format(model_dirpath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('model_format = {}'.format(model_format))
    print('example_img_format = {}'.format(example_img_format ))

    # Identify all models in directories
    model_dir_names = os.listdir(model_dirpath)
    print('os.listdir(model_dirpath):',os.listdir(model_dirpath))
    model_filepath = []
    array_model_dir_names = []
    idx = 0
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        isdir = os.path.isdir(model_dirpath1)
        if isdir:
            for fn1 in os.listdir(model_dirpath1):
                if fn1.endswith(model_format):
                    model_filepath.append(os.path.join(model_dirpath1, fn1))
                    array_model_dir_names.append(model_dirpath1)
                    idx = idx + 1

    number_of_models = idx
    print('number of models:', number_of_models, '\n model_filepath array:', model_filepath)
    #array_model_dir_names = np.asarray(model_dir_names)

    examples_dirpath = []
    idx = 0
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        if not os.path.isdir(model_dirpath1):
            continue
        examples_dirpath1 = os.path.join(model_dirpath, fn, 'example_data/')
        # check if examples_dirpath1 exists
        isdir = os.path.isdir(examples_dirpath1 )
        if not isdir:
            continue
            # # check if tar file exists
            # examples_filepath1 = os.path.join(model_dirpath, fn, 'example_data.tar.gz')
            # if os.path.isfile(examples_filepath1  ):
            #     # unzip/untar the file
            #     tar = tarfile.open(examples_filepath1, "r:gz")
            #     tar.extractall(os.path.join(model_dirpath, fn))
            #     tar.close()
            # else:
            #     print('missing example images:', examples_dirpath1)
            #     return -1

        examples_dirpath.append(examples_dirpath1)
        idx = idx + 1

        # for fn1 in os.listdir(examples_dirpath1):
        #     if fn1.endswith(example_img_format):
        #         examples_dirpath.append(os.path.join(examples_dirpath1, fn1))
        #         idx = idx + 1
    number_of_exampledir = idx
    print('number of example dirs:', number_of_exampledir, '\n')

    ################## loop over all models ##############################
    prob_poisoned = []
    classification_error = 0

    start = time.time()

    # restrict the number of models to any number instead of testing 1000 models in Round 1
    # if len(model_filepath) > 100:
    #     model_filepath = model_filepath[0:2]

    for idx in range(0, number_of_models):
        print('processing model_filepath:', model_filepath[idx])
        print('model dir:', array_model_dir_names[idx])
        start1 = time.time()
        # read the ground truth label for the model
        gt_model_label_filepath = os.path.join(model_dirpath, array_model_dir_names[idx], 'ground_truth.csv')
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
            return -1

        # decide which model architecture is represented by the provided AI Model
        a = model_classifier(model_filepath[idx])
        model_type, min_model_size_delta = a.classify_type()
        print('model_type: %s\n' % model_type)
        print('file size delta between a model and the reference model: %s\n' % min_model_size_delta)
        model_name = a.switch_architecture(model_type)
        print('classified the model as:\t', model_name)

        scratch_filepath = os.path.join(scratch_dirpath, model_name + '_log.csv')
        # if not os.path.isfile(scratch_filepath):
        #     # write header
        #     with open(scratch_filepath, 'w') as fh:
        #         fh.write("header, copy entries \n ")

        with open(scratch_filepath, 'a') as fh:
            fh.write("idx, {}, ".format(idx))
            fh.write("model dir, {}, ".format(array_model_dir_names[idx]))
            fh.write("gt_model_label, {}, ".format(gt_model_label))

        ##################
        hp = hpy()
        before = hp.heap()
        prob_trojan_in_model = td.trojan_detector(model_filepath[idx], result_filepath, scratch_dirpath, examples_dirpath[idx],
                            example_img_format=example_img_format)
        after = hp.heap()
        leftover = after - before
        # print('before:', before, ' after:', after, ' leftover:', leftover)
        # print('leftover max:', leftover.byrcs[0].byid)
        print('processing a model RAM leftover.domisize:', leftover.domisize)

        print('model: ', array_model_dir_names[idx], ' gt_model_label: ', gt_model_label)

        # false positive
        false_positive = 0
        if gt_model_label == 0 and prob_trojan_in_model >= 0.5:
            print('model:', array_model_dir_names[idx], ' false positive: target = ', gt_model_label, ' prob_trojan_in_model =', prob_trojan_in_model)
            false_positive = 1
        # false negative
        false_negative = 0
        if gt_model_label == 1 and prob_trojan_in_model < 0.5:
            print('model:', array_model_dir_names[idx], ' false negative: target = ', gt_model_label, ' prob_trojan_in_model =', prob_trojan_in_model)
            false_negative = 1

        classification_error += false_positive + false_negative
        prob_poisoned.append(prob_trojan_in_model)
        end1 = time.time()
        with open(scratch_filepath, 'a') as fh:
            # fh.write("idx, {}, ".format(idx))
            # fh.write("model dir, {}, ".format(array_model_dir_names[idx]))
            # fh.write("gt_model_label, {}, ".format(gt_model_label))
            fh.write("FN, {}, ".format(false_negative))
            fh.write("FP, {}, ".format(false_positive))
            fh.write("Error, {}, ".format( (false_positive+false_negative) ) )
            fh.write("RAM leftover, {},".format(leftover.domisize))
            fh.write("time [s], {} \n".format( (end1-start1) ))


    print('classification error count:', classification_error)
    classification_error = 100.0 * classification_error/number_of_models
    print('classification error [%]:', classification_error )
    end = time.time()
    with open(scratch_filepath, 'a') as fh:
        fh.write("execution time [s], {}, ".format((end - start)))
        fh.write("classification error [%]: {} \n \n ".format(classification_error))


    return prob_poisoned



if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_dirpath', type=str, help='Directory path to all pytorch genrated data and models to be evaluated.',
                        required=True)
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        required=True)
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        required=True)
    parser.add_argument('--model_format', type=str,
                        help='Model file format (suffix)  which might be useful for filtering a folder containing a model .',
                        required=False)
    parser.add_argument('--image_format', type=str,
                        help='Exampple image file format (suffix)  which might be useful for filtering a folder containing axample files.',
                        required=False)

    args = parser.parse_args()
    print('args %s \n % s \n %s \n %s \n %s \n' % (
        args.model_dirpath, args.result_filepath, args.scratch_dirpath, args.model_format, args.image_format))

    batch_model_classifier(args.model_dirpath, args.result_filepath, args.scratch_dirpath)
