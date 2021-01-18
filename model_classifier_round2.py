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
import sys

import torch
import time
import torchvision

""" 
This class returns the AI architecture and its model type based on the torchivision class and the model file size.
It was designed for TrojAI Round 2 Challenge datasets consisting of 22 AI architectures
"""
# round 1 trojAI challenge
# Resnet 50,
# Densenet121
# Inception v3

# round 2 trojAI challenge:
# Resnet 18, 34, 50, 101, 152
# WideResnet 50, 101
# Densenet 121, 161, 169, 201
# Inception v1(googlenet), v3
# Squeezenet 1.0, 1.1
# Mobilenet mobilenet_v2
# ShuffleNet 1.0, 1.5, and 2.0
# VGG vgg11_bn, vgg13_bn, vgg16_bn


class model_classifier:
    # 22 models in round 2
    # centroid file size values per architecture model
    squeezenetv1_1_size = float(2.962577345)  # in MB
    squeezenetv1_0_size = float(3.010153304)  # in MB

    mobilenetv2_size = float(9.237641118)  # in MB

    shufflenet1_0_size = float(5.26193666)  # in MB
    shufflenet1_5_size = float(10.20389327)  # in MB
    shufflenet2_0_size = float(21.75808312)  # in MB

    inception1_googlenet_size = float(22.6557703)  # in MB
    inceptionv3_size = float(87.62433951)  # in MB

    densenet121_size = float(28.48473237)  # in MB
    densenet169_size = float(51.06529902)  # in MB
    densenet201_size = float(73.86921903)  # in MB
    densenet161_size = float(107.2791326)  # in MB

    resnet18_size = float(44.8263574)  # in MB
    resnet34_size = float(85.32276613)  # in MB
    resnet50_size = float(94.48834923)  # in MB
    resnet101_size = float(170.7656982)  # in MB wrong
    resnet152_size = float(233.6403364)  # in MB
    wideresnet50_size = float(267.8514268)  # in MB
    wideresnet101_size = float(500.261398)  # in MB

    vgg11_bn_size = float(515.8239454)  # in MB
    vgg13_bn_size = float(537.372336)  # in MB
    vgg16_bn_size = float(558.661882)  # in MB

    MODEL_ARCHITECTURES = ['resnet','densenet','googlenet','inception','squeezenet','mobilenet','shufflenetv2','vgg']
    NUMBER_OF_TYPES_PER_ARCH = [7,4,1,1,2,1,3,3]
    MODEL_NAMES = ["resnet18","resnet34","resnet50","resnet101","resnet152","wide_resnet50", "wide_resnet101",
                   "densenet121","densenet161","densenet169","densenet201",
                   "inceptionv1(googlenet)","inceptionv3",
                   "squeezenetv1_0","squeezenetv1_1","mobilenetv2",
                   "shufflenet1_0","shufflenet1_5","shufflenet2_0",
                   "vgg11_bn", "vgg13_bn","vgg16_bn"]
    MODEL_REF_SIZES = [resnet18_size,resnet34_size,resnet50_size,resnet101_size, resnet152_size,wideresnet50_size,wideresnet101_size,
                       densenet121_size,densenet161_size,densenet169_size,densenet201_size,
                       inception1_googlenet_size,inceptionv3_size,
                       squeezenetv1_0_size,squeezenetv1_1_size,mobilenetv2_size,
                       shufflenet1_0_size,shufflenet1_5_size,shufflenet2_0_size,
                       vgg11_bn_size,vgg13_bn_size,vgg16_bn_size]
    MODEL_SIZE_STDEV = [0.010747652, 0.012185269, 0.051370421, 0.062585081, 0.074508965, 0.044384475, 0.047864105,
                        0.038431258, 0.063003244, 0.066035393, 0.073138502,
                        0.024616621, 0.045563637,
                        0.010976248, 0.015943522, 0.030727859,
                        0.023552785, 0.026794575, 0.044872556,
                        0.391894373, 0.093213096, 0.088079363]

    model_filepath = ''
    model_size = 0
    min_model_size_delta = sys.float_info.max #(1.7976931348623157e+308)
    model_architecture = ''
    model_name = ''
    model_type = -1

    def __init__(self, model_filepath: object) -> object:
        self.model_filepath = model_filepath
        self.model_size = 0

    ###########################
    # determine the AI model architecture based on loaded class type
    def classify_architecture(self, mydevice):
        # load a model
        model_orig = torch.load(self.model_filepath, map_location=mydevice)
        # get the model class name <class 'torchvision.models.resnet.ResNet'>
        str_model = model_orig.__class__
        # convert to string
        model_name_str = str(str_model)
        print('model_name_str:', model_name_str)
        # split on the common class name yielding ["<class '", "resnet.ResNet'>"]
        split_string = model_name_str.split("torchvision.models.")
        # print(split_string)
        # second split
        split_string2 = split_string[1].split(".")
        model_architecture = split_string2[0]
        print('model_architecture:', model_architecture)
        self.model_architecture = model_architecture
        return model_name_str,model_architecture

    ###########################
    def switch_architecture(self, argument):
        """
        @argument integer [0,20]
        :return: string
        """
        if not isinstance(argument, int):
            print('ERROR: switch_architecture - argument is not int: ', argument)
            return 'Invalid architecture'

        if argument < 0 or argument > len(self.MODEL_NAMES):
            print('ERROR: switch_architecture - argument is out of range: ', argument)
            return 'Invalid architecture'

        return self.MODEL_NAMES[argument]

    ########################################################
    # classify AI model type given an architecture based on file size
    def classify_type(self, model_architecture):
        """
        This method classifies a model based on its file size
        :param model_architecture: model_architecture defined by self.MODEL_ARCHITECTURES
        :return: model_name (string), model_type (int associated with model_name), and delta size (ref-model_size)
        """
        a = model_classifier(self.model_filepath)
        self.model_architecture = model_architecture

        size = os.stat(self.model_filepath).st_size
        size = size / 1000000  # in MB
        print('Model size in MB: {}'.format(size))
        self.model_size = size

        # compute the offsets depending on the number of model types per architecture
        offset = []
        cum = 0
        for i in range(len(a.NUMBER_OF_TYPES_PER_ARCH)):
            offset.append(cum)
            cum = cum + a.NUMBER_OF_TYPES_PER_ARCH[i]

        #print('offset:', offset)

        found_match = False
        for i in range(len(a.MODEL_ARCHITECTURES)):
            if found_match:
                continue
            model_name = a.MODEL_NAMES[offset[i]]
            model_type = offset[i]
            min_model_size_delta = a.MODEL_REF_SIZES[model_type] - size
            if model_architecture.lower() == a.MODEL_ARCHITECTURES[i].lower():
                print('match architecture:', model_architecture)
                number_of_type = a.NUMBER_OF_TYPES_PER_ARCH[i]
                min_delta = float('inf')
                found_match = True
                for j in range(offset[i], offset[i] + number_of_type):
                    if abs(size - a.MODEL_REF_SIZES[j]) < min_delta:
                        min_delta = abs(size - a.MODEL_REF_SIZES[j])
                        model_name = a.MODEL_NAMES[j]
                        min_model_size_delta = a.MODEL_REF_SIZES[j] - size
                        model_type = j

        self.model_type = model_type
        self.model_name = model_name
        self.min_model_size_delta = min_model_size_delta

        print('classified the model based on model_type as:\t', a.switch_architecture(model_type))
        print('model_name:', model_name)
        return model_name, model_type, min_model_size_delta

###################################
# sweep over all models in model_dirpath and save the model architecture and model type
# for each model
def batch_model_name(model_dirpath, result_filepath, model_format='.pt'):
    print('model_dirpath = {}'.format(model_dirpath))
    print('result_filepath = {}'.format(result_filepath))
    print('model_format = {}'.format(model_format))

    # Identify all models in directories
    model_dir_names = os.listdir(model_dirpath)
    print('os.listdir(model_dirpath):', os.listdir(model_dirpath))
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
    # array_model_dir_names = np.asarray(model_dir_names)

    ################## loop over all models ##############################
    # model_name = []
    start = time.time()
    mydevice = "cpu"  # torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for idx in range(0, number_of_models):
        print('processing model_filepath:', model_filepath[idx])
        print('model dir:', array_model_dir_names[idx])

        # init the model_classifier class
        a = model_classifier(model_filepath[idx])

        # determine the AI model architecture based on loaded class type
        model_class_str, model_architecture = a.classify_architecture(mydevice)

        # based on the model name and model size determine AI model type
        # e.g., model_architecture = resnet, model_size = 94.381 MB ==> model_type = 1- N with corresponding
        # model_name = resnet50
        model_name, model_type, min_model_size_delta = a.classify_type(model_architecture)
        print('model_size: %s ' % a.model_size)
        print('model_type: %s ' % model_type)
        print('file size delta between a model and the reference model: %s ' % min_model_size_delta)
        #model_name = a.switch_architecture(model_type)
        print('classified the model as: ', model_name)


        ##################
        with open(result_filepath, 'a') as fh:
            fh.write("idx, {}, ".format(idx))
            fh.write("model dir, {}, ".format(array_model_dir_names[idx]))
            fh.write("model_class_str, {}, ".format(model_class_str))
            fh.write("model_architecture, {}, ".format(model_architecture))
            fh.write("model_type, {}, ".format(model_type))
            fh.write("model_name, {}, ".format(model_name))
            fh.write("model_size, {}, ".format(a.model_size))
            fh.write("min_model_size_delta, {} \n".format(min_model_size_delta))


    end = time.time()
    with open(result_filepath, 'a') as fh:
        fh.write("execution time [s], {} \n \n ".format((end - start)))



###############################################################
if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Load models and report their names and file size.')
    parser.add_argument('--model_dirpath', type=str,
                        help='Directory path to all pytorch genrated data and models to be evaluated.',
                        required=True)
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        required=True)
    parser.add_argument('--model_format', type=str,
                        help='Model file format (suffix)  which might be useful for filtering a folder containing a model .',
                        required=False)

    args = parser.parse_args()
    print('args %s \n % s \n %s \n ' % (
        args.model_dirpath, args.result_filepath, args.model_format))

    batch_model_name(args.model_dirpath, args.result_filepath, args.model_format)
