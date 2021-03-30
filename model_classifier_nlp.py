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

""" 
This class returns the AI architecture for NLP problems in TrojAI Round 5

From: https://pages.nist.gov/trojai/docs/data.html#round-5
The embeddings used are drawn from HuggingFace.
EMBEDDING_LEVELS = ['BERT', 'GPT-2', 'DistilBERT']
Each broad embedding type (i.e. BERT) has several flavors to choose from in HuggingFace. 
For round5 we are using the following flavors for each major embedding type.

EMBEDDING_FLAVOR_LEVELS = dict()
EMBEDDING_FLAVOR_LEVELS['BERT'] = ['bert-base-uncased']
EMBEDDING_FLAVOR_LEVELS['GPT-2'] = ['gpt2']
EMBEDDING_FLAVOR_LEVELS['DistilBERT'] = ['distilbert-base-uncased']

It was designed for TrojAI Round 5 Challenge datasets consisting of 3 AI architectures

GruLinearModel(
  (rnn): GRU(768, 256, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (linear): Linear(in_features=512, out_features=2, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
LstmLinearModel(
  (rnn): LSTM(768, 256, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
  (linear): Linear(in_features=512, out_features=2, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
model: LstmLinearModel(
  (rnn): LSTM(768, 512, num_layers=4, batch_first=True, dropout=0.1)
  (linear): Linear(in_features=512, out_features=2, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
)
"""


# round 5 trojAI challenge


class model_classifier:
    # 3 models in round 5
    # centroid file size values per architecture model
    lstm_linear_model_size = float(14.72431262)  # id-0 (poisoned): LstmLinear 14,724,241 bytes, "embedding": "GPT-2",
    linear_model_size = float(0.008001016)  # id-01 (clean): Linear 8,001 bytes, "embedding": "DistilBERT",
    gru_linear_size = float(11.04610831)  # id-02 (poisoned): GruLinear 11,046,033 bytes, "embedding": "BERT",

    # TODO: Add NLP model names as needed 'LstmLinear', 'GruLinear', 'Linear'
    MODEL_ARCHITECTURES = ['LstmLinearModel', 'GruLinearModel', 'LinearModel']
    NUMBER_OF_TYPES_PER_ARCH = [1, 1, 1]
    # MODEL_NAMES = ['LstmLinear', 'GruLinear', 'Linear']
    MODEL_REF_SIZES = [lstm_linear_model_size, gru_linear_size, linear_model_size]
    MODEL_SIZE_STDEV = [0.000127959, 1.28037E-07, 0.000128203]

    model_filepath = ''
    model_size = 0
    min_model_size_delta = sys.float_info.max  # (1.7976931348623157e+308)
    model_architecture = ''
    # model_name = ''
    model_type = -1

    def __init__(self, model_filepath: object) -> object:
        """

        Args:
            model_filepath (object): 
        """
        self.model_filepath = model_filepath
        self.model_size = 0

    ###########################
    # determine the AI model architecture based on loaded class type
    def classify_architecture(self, mydevice):
        # load a model
        try:
            model_orig = torch.load(self.model_filepath, map_location=mydevice)
        except:
            print("Unexpected loading error:", sys.exc_info()[0])
            # close the line
            # TODO: How to pass scratch_filepath here?
            # with open(scratch_filepath, 'a') as fh:
            #     fh.write("\n")
            raise

        #model_orig = torch.load(self.model_filepath, map_location=mydevice)
        # get the model class name <class 'torchvision.models.resnet.ResNet'>
        str_model = model_orig.__class__
        # convert to string
        model_name_str = str(str_model)
        print('model_name_str:', model_name_str)
        # split on the common class name yielding ["<class '", "resnet.ResNet'>"]
        split_string = model_name_str.split("model_factories.")
        # print(split_string)
        # second split
        print('split_string:', split_string)
        split_string2 = split_string[1].split("'")
        model_architecture = split_string2[0]
        print('model_architecture:', model_architecture)
        self.model_architecture = model_architecture
        return model_name_str, model_architecture

    ###########################
    def switch_architecture(self, argument):
        """
        @argument integer [0,2]
        :return: string
        """
        if not isinstance(argument, int):
            print('ERROR: switch_architecture - argument is not int: ', argument)
            return 'Invalid architecture'

        if argument < 0 or argument > len(self.MODEL_ARCHITECTURES):
            print('ERROR: switch_architecture - argument is out of range: ', argument)
            return 'Invalid architecture'

        return self.MODEL_ARCHITECTURES[argument]

    def get_filesize(self, model_architecture):
        """
        This method returns a file size of a model
        :param model_architecture: model_architecture defined by self.MODEL_ARCHITECTURES
        :return: file size and delta size (ref-model_size)
        """
        a = model_classifier(self.model_filepath)
        self.model_architecture = model_architecture

        size = os.stat(self.model_filepath).st_size
        size = size / 1000000  # in MB
        print('Model size in MB: {}'.format(size))
        self.model_size = size

        found_match = False
        min_delta = float('inf')
        for i in range(len(a.MODEL_ARCHITECTURES)):
            if abs(size - a.MODEL_REF_SIZES[i]) < min_delta:
                min_delta = abs(size - a.MODEL_REF_SIZES[i])
                model_type = i

        min_model_size_delta = min_delta

        self.model_type = model_type
        self.min_model_size_delta = min_model_size_delta

        print('classified the model based on model_type as:\t', a.switch_architecture(model_type))
        return size, model_type, min_model_size_delta


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
        model_size, model_type, min_model_size_delta = a.get_filesize(model_architecture)
        print('model_file_size: %s ' % model_size)
        print('model_type: %s ' % model_type)
        print('file size delta between a model and the reference model: %s ' % min_model_size_delta)
        # model_name = a.switch_architecture(model_type)
        print('classified the model as: ', model_architecture)

        ##################
        with open(result_filepath, 'a') as fh:
            fh.write("idx, {}, ".format(idx))
            fh.write("model dir, {}, ".format(array_model_dir_names[idx]))
            fh.write("model_class_str, {}, ".format(model_class_str))
            fh.write("model_architecture, {}, ".format(model_architecture))
            fh.write("model_type, {}, ".format(model_type))
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
                        help='Directory path to all pytorch generated data and models to be evaluated.',
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
