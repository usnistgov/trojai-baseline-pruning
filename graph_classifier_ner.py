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
# it was created for round 5 TrojAI challenge
"""


import os
import numpy as np
import warnings
from trace_graph import *
import json
import sys
import transformers

warnings.filterwarnings("ignore")
from extended_dataset_ner import extended_dataset_ner
from model_classifier_ner import model_classifier


def graph_extractor(model_dirpath, tokenizer_dirpath,
                    result_filepath, scratch_dirpath, model_format='.pt'):
    print('model_dirpath = {}'.format(model_dirpath))
    print('tokenizer_dirpath = {}'.format(tokenizer_dirpath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('model_format = {}'.format(model_format))

    tokenizer_filepath = "" #args.tokenizer_filepath

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
    print('os.listdir(model_filepath):',os.listdir(model_dirpath))
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

        ##############################################################
        # load a model
        model = torch.load(model_filepath[idx], map_location=mydevice)
        ################################
        # decide which model architecture is represented by the provided AI Model
        a = model_classifier(model_filepath[idx])
        # determine the AI model architecture based on loaded class type
        model_class_str, model_architecture = a.classify_architecture(mydevice)
        model_size, model_type, min_model_size_delta = a.get_filesize(model_architecture)
        model_name = a.get_model_name(model_type)

        # self.model_name, self.model_type, self.min_model_size_delta = a.classify_type(self.model_architecture)
        print('model_type: %s \t' % model_type)
        print('model name: %s \n' % model_name)
        print('file size delta between a model and the reference model: %s\n' % min_model_size_delta)
        print('classified the model as:\t', model_architecture)
        print('model size: \t', a.model_size)
        ref_model_size = a.model_size + min_model_size_delta
        print('reference model size: \t', ref_model_size)

        scratch_filepath = os.path.join(scratch_dirpath, model_architecture + '_' + model_name + '_log.csv')
        # to avoid messages about serialization on cpu
        torch.nn.Module.dump_path = 'False'


        ##############################################################
        # -------------------------------------------------------------------
        # NER Setup ---------------------------------------------------------
        # -------------------------------------------------------------------
        ###############################################################
        ###################################################
        ### Prepare the list of clean example files
        examples_dirpath = ''
        examples_dirpath = os.path.join(model_dirpath, array_model_dir_names[idx], 'clean_example_data')
        # read the config file and load the tokenizer
        # create the list of example filenames from the provided clean data
        example_filenames = ''
        example_filenames = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if
                                  fn.endswith('.txt')]

        example_filenames.sort()  # ensure file ordering

        # load the config file to retrieve parameters
        #model_dirpath, _ = os.path.split(model_filepath[idx])
        with open(os.path.join(model_dirpath, array_model_dir_names[idx], 'config.json')) as json_file:
            config = json.load(json_file)
            embedding_name = None
            if config['embedding']:
                embedding_name = config['embedding']

            if embedding_name == 'DistilBERT':
                tokenizer_filepath = os.path.join(tokenizer_dirpath, 'DistilBERT-distilbert-base-cased.pt')
            elif embedding_name == 'BERT':
                tokenizer_filepath = os.path.join(tokenizer_dirpath,'BERT-bert-base-uncased.pt')
            elif embedding_name == 'MobileBERT':
                tokenizer_filepath = os.path.join(tokenizer_dirpath,'MobileBERT-google-mobilebert-uncased.pt')
            elif embedding_name == 'RoBERTa':
                tokenizer_filepath = os.path.join(tokenizer_dirpath,'RoBERTa-roberta-base.pt')
            else:
                print("Unknown embedding name:", embedding_name)
                sys.exit(2)

            print('Source dataset name = "{}"'.format(config['source_dataset']))

            if 'data_filepath' in config.keys():
                print('Source dataset filepath = "{}"'.format(config['data_filepath']))

            # Load the provided tokenizer
            # TODO: Should use this for evaluation server

            tokenizer = torch.load(tokenizer_filepath)

            # Or load the tokenizer from the HuggingFace library by name
            embedding_flavor = config['embedding_flavor']
            if config['embedding'] == 'RoBERTa':
                tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True,
                                                                            add_prefix_space=True)
            else:
                tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)

            # set the padding token if its undefined
            if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # identify the max sequence length for the given embedding
            if config['embedding'] == 'MobileBERT':
                max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
            else:
                max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

        ###################################################
        # we need only one to initiate the model and extract the graph
        input_sample = []
        input_sample.append(example_filenames[0])
        #input_sample.append(example_filenames[1])
        #print('input_sample:', input_sample)

        preprocessed_data = extended_dataset_ner(input_sample, tokenizer, max_input_length,num_iterations=1)
        #dataset = extended_dataset_nlp(self.example_filenames, self.tokenizer, self.max_input_length, num_iterations=self.num_duplicate_data_iterations)
        #preprocessed_data = self._preprocess_data(dataset)

        use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
        # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
        # Note, should NOT use_amp when operating with MobileBERT

        # predict the text sentiment
        if use_amp:
            with torch.cuda.amp.autocast():
                # Classification model returns loss, logits, can ignore loss if needed
                _, logits = model(preprocessed_data.input_ids, attention_mask=preprocessed_data.attention_mask,
                                  labels=preprocessed_data.labels_tensor)
        else:
            _, logits = model(preprocessed_data.input_ids, attention_mask=preprocessed_data.attention_mask,
                              labels=preprocessed_data.labels_tensor)

        preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
        numpy_logits = logits.cpu().flatten().detach().numpy()

        #print('INFO: logits prediction:', logits)

        # if the network graph is needed
        #dot = make_dot(model(Variable(input)), params=dict(model.named_parameters()))
        dot = make_dot(logits, params=dict(model.named_parameters()))
        out_dot_filename = scratch_dirpath + array_model_dir_names[idx] + '_' + model_name + '_graph.dot'
        #dot.view(out_dot_filename)
        print('saving file:', out_dot_filename)
        dot.save(out_dot_filename)


if __name__ == "__main__":
    import argparse

    print('torch version: %s \n' % (torch.__version__))

    parser = argparse.ArgumentParser(
        description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_dirpath', type=str, help='Directory path to all pytorch genrated data and models to be evaluated.',
                        required=True)
    # parser.add_argument('--reference_dirpath', type=str,
    #                     help='Directory path to all reference data used for evaluating models (i.e., the graphs)).',
    #                     required=True)
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        required=True)
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        required=True)
    parser.add_argument('--model_format', type=str,
                        help='Model file format (suffix)  which might be useful for filtering a folder containing a model .',
                        required=False)
    # parser.add_argument('--tokenizer_filepath', type=str,
    #                     help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.',
    #                     default='')
    # parser.add_argument('--embedding_filepath', type=str,
    #                     help='File path to the pytorch model (.pt) file containing the correct embedding to be used with the model_filepath.',
    #                     default='')
    # parser.add_argument('--embedding_dirpath', type=str,
    #                     help='Directory path for all embeddings (used if embedding_filepath or tokenizer_filepath is not used)',
    #                     default=None)
    parser.add_argument('--tokenizer_dirpath', type=str,
                        help='Directory path for all tokenizers (used if embedding_filepath or tokenizer_filepath is not used)',
                        default=None)
    # parser.add_argument('--examples_dirpath', type=str,
    #                     help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
    #                     default='')
    # parser.add_argument('--num_duplicate_data_iterations', type=int,
    #                     help='Number of iterations to run when processing data (values >1 will duplicate the data).',
    #                     default='1')

    args = parser.parse_args()

    model_dirpath = os.path.dirname(args.model_dirpath)
    print('args %s \n % s \n %s \n %s \n %s \n' % (
        model_dirpath, args.tokenizer_dirpath, args.result_filepath, args.scratch_dirpath, args.model_format))

    graph_extractor(model_dirpath, args.tokenizer_dirpath, args.result_filepath, args.scratch_dirpath, args.model_format)
