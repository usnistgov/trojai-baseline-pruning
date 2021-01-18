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

"""
# This code is designed to classify AI architecture based on its file size, extract its graphs, and then compare it for any deviations
# it was created for round 1 TrojAI challenge
"""

import os
import numpy as np
import warnings
from trace_graph import *
warnings.filterwarnings("ignore")
from model_classifier import model_classifier


def graph_classifier(model_dirpath, reference_dirpath, result_filepath, scratch_dirpath, model_format='.pt'):
    print('model_dirpath = {}'.format(model_dirpath))
    print('reference_dirpath = {}'.format(reference_dirpath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('model_format = {}'.format(model_format))

    #######################################
    # adjust to the hardware platform
    mydevice = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ############################
    # to avoid messages about serialization on cpu
    torch.nn.Module.dump_patches = 'False'
    #################################
    # Identify all models in directories
    #model_format = '.pt.1'
    model_dir_names = os.listdir(model_dirpath)
    print('os.listdir(model_dirpath):',os.listdir(model_dirpath))
    model_filepath = []
    idx = 0

    model_dir_names_filtered = []
    # filter directories only
    for fn in model_dir_names:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        isdir = os.path.isdir(model_dirpath1)
        if isdir:
            model_dir_names_filtered.append(fn)
    print(model_dir_names_filtered)

    for fn in model_dir_names_filtered:
        model_dirpath1 = os.path.join(model_dirpath, fn)
        for fn1 in os.listdir(model_dirpath1):
            if fn1.endswith(model_format):
                model_filepath.append(os.path.join(model_dirpath1, fn1))
                idx = idx + 1


    number_of_models = idx
    print('number of models:', number_of_models, '\n model_filepath array:', model_filepath)

    array_model_dir_names = np.asarray(model_dir_names_filtered)
    ################## loop over all models ##############################
    for idx in range(0, number_of_models):
        print('processing model_filepath:', model_filepath[idx])
        print('model dir:', array_model_dir_names[idx])

        ##################
        # decide which model architecture is represented by the provided AI Model
        a = model_classifier(model_filepath[idx])
        model_type, min_model_size_delta = a.classify_type()
        min_model_size_delta = min_model_size_delta * 1000 # convert from MB to KB
        print('model_type: %s\n' % model_type)
        print('file size delta between a model and the reference model [KB}: %s\n' % min_model_size_delta)
        print('classified the model as:\t', a.switch_architecture(model_type))
        ##############################################################
        # load a model
        model = torch.load(model_filepath[idx], map_location=mydevice)
        # model = torch.load(model_filepath)
        model = torch.nn.modules.container.Sequential.cpu(model)
        # the eval is needed to set the model for inferencing
        model.eval()

        ##############################################################
        # read the image (using skimage)
        # img = skimage.io.imread(fn)
        # img = np.float32(cv2.resize(img, (224, 224))) / 255
        # input = preprocess_round0_image(img)

        # Create a random image/batch
        img = np.random.rand(1, 3, 224, 224)
        # img_tmp_fp = os.path.join(scratch_dirpath, (array_model_dir_names[idx] + '_img.png') )
        # print('img_tmp_fp:', img_tmp_fp)
        # np.save(img_tmp_fp, img)
        input = torch.FloatTensor(img, device=mydevice)

        # if the network graph is needed
        dot = make_dot(model(Variable(input)), params=dict(model.named_parameters()))
        out_dot_filename = scratch_dirpath + array_model_dir_names[idx] + '_' + a.switch_architecture(model_type) + '_graph.dot'
        #dot.view(out_dot_filename)
        dot.save(out_dot_filename)
        reference_dot_filename = reference_dirpath + a.switch_architecture(model_type) + '_graph.dot'
        graph_compare_file = scratch_dirpath + array_model_dir_names[idx] + '_' + a.switch_architecture(model_type) + '_graph_compare.txt'

        #################################################################
        # this comparison does not work because the .dot files use different numbers assigned to each node
        # this is for linux systems
        # command = "diff " + out_dot_filename + " " + reference_dot_filename + " > " + graph_compare_file
        # this is for windows
        # /L – This will compare your files as ASCII text
        # command = "fc /C " + out_dot_filename + " " + reference_dot_filename + " > " + graph_compare_file
        # os.system('cmd /c' + command)

        # is_graph_different = True
        # if os.path.isfile(graph_compare_file):
        #     graph_compare_size = os.stat(graph_compare_file).st_size # this is in Bytes
        #     graph_compare_size = graph_compare_size/1000 # convert to KB
        #     if graph_compare_size > 1.23: # this is the text saying that there is no difference ~ 1.23KB !!!
        #         print('Model graph deviates from the expected graph of ' + a.switch_architecture(model_type) + ' by ' + str(graph_compare_size))
        #         is_graph_different = True
        #     else:
        #         print('Model graph is the same as the expected graph of ' + a.switch_architecture(model_type))
        #         is_graph_different = False
        # else:
        #     print('ERROR: could not compare the reference graph in ', reference_dot_filename, ' and the test graph in ', out_dot_filename )

        # this comparison is based on line count of the reference and test .dot files
        # and based on some key lines that must be the same
        # the line count could also be done on linux by "wc -l yourTextFile"

        # metrics
        is_graph_line_count_different = False
        graph_compare_line = 0
        is_graph_header_different = False
        is_graph_node_different = False

        f_ref = open(reference_dot_filename, "r")
        f_test = open(out_dot_filename, "r")
        line_count_ref = 0
        for x in f_ref:
            line_count_ref += 1
            #print(x)
        line_count_test = 0
        for x in f_test:
            line_count_test += 1
            #print(x)

        print('reference line count:',line_count_ref, ' test line count:', line_count_test)
        graph_compare_line = line_count_ref - line_count_test
        if line_count_ref != line_count_test:
            print('Model graph number of lines deviates from the expected graph of ' + a.switch_architecture(model_type) + ' by ' + str(graph_compare_line))
            is_graph_line_count_different = True
        else:
            print('Model graph number of lines is the same as in the expected graph of ' + a.switch_architecture(model_type))
            #is_graph_line_count_different = False

        f_ref.close()
        f_test.close()

        if not is_graph_line_count_different:
            # check a few lines
            f_ref = open(reference_dot_filename, "r")
            f_test = open(out_dot_filename, "r")
            line_idx = 0
            #is_graph_header_different = False
            while not is_graph_header_different and line_idx < 3:
                line_ref = f_ref.readline()
                #print('0-2: line reference idx:', line_idx, ' line content:',line_ref)
                line_test = f_test.readline()
                #print('0-2: line test idx:', line_idx, ' line content:', line_test)
                if line_ref != line_test:
                    print('Model graph line:', line_idx, ' is different from the line in the expected graph of ' + a.switch_architecture(
                        model_type))
                    is_graph_header_different = True
                line_idx += 1

            #is_graph_node_different = False
            while not is_graph_node_different and line_idx < line_count_ref:
                line_ref = f_ref.readline()
                line_test = f_test.readline()
                if '[label=' in line_ref:
                    # compare the rest of the line in both files
                    #print('line reference idx:', line_idx, ' line content:', line_ref)
                    #print('line test idx:', line_idx, ' line content:', line_test)
                    temp_ref = line_ref.rfind('[label=') + 7
                    temp_test = line_test.rfind('[label=') + 7
                    if temp_ref != temp_test:
                        print('Model graph line:', line_idx, ' is different from the line in the expected graph of ' + a.switch_architecture(model_type))
                        is_graph_node_different = True
                line_idx += 1

        f_ref.close()
        f_test.close()

        if not is_graph_node_different:
            print('All model graph lines have the same content as the reference model graph')

    #######################################################
        # save some numbers about the model
        if not os.path.isfile(result_filepath):
            # file does not exist and therefore we write the header
            fh = open(result_filepath, 'w')
            fh.write(
                "model index: \t, model: \t, model type based on file size: \t, size delta between reference and test models [KB]:" \
                " \t, is graph line count different? \t, graph line delta  \t, is graph header different? \t, is graph node different?  ")


        with open(result_filepath, 'a') as fh:
            fh.write("\n {}".format(idx))
            fh.write("\t, {}".format(model_filepath[idx]))
            fh.write("\t, {}".format(a.switch_architecture(model_type)))
            fh.write("\t, {}".format(min_model_size_delta))
            fh.write("\t, {}".format(is_graph_line_count_different))
            fh.write("\t, {}".format(graph_compare_line))
            fh.write("\t, {}".format(is_graph_header_different))
            fh.write("\t, {}".format(is_graph_node_different))




if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_dirpath', type=str, help='Directory path to all pytorch genrated data and models to be evaluated.',
                        required=True)
    parser.add_argument('--reference_dirpath', type=str,
                        help='Directory path to all reference data used for evaluating models (i.e., the graphs)).',
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

    args = parser.parse_args()
    print('args %s \n % s \n %s \n %s \n %s \n' % (
        args.model_dirpath, args.reference_dirpath, args.result_filepath, args.scratch_dirpath, args.model_format))

    graph_classifier(args.model_dirpath, args.reference_dirpath, args.result_filepath, args.scratch_dirpath, args.model_format)
