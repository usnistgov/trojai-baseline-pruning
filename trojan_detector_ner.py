import torch
import os
import csv
import numpy as np
import sys
import time
import copy
import transformers
import json
import statistics
from collections import OrderedDict
import configargparse
import psutil


from model_classifier_ner import model_classifier
from extended_dataset_ner import extended_dataset_ner
#from remove_prune_ner import remove_prune_model
from reset_prune_ner import reset_prune_model
from trim_prune_ner import trim_prune_model
from linear_regression import read_regression_coefficients, linear_regression_prediction


class TrojanDetectorNER:
    def __init__(self, model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath,
                 examples_dirpath, pruning_method, sampling_method, ranking_method,
                 num_samples, num_images_used, linear_regression_filepath,
                 trim_pruned_amount=0.5, trim_pruned_multiplier=4.0, trim_pruned_divisor=1000.0,
                 reset_pruned_divisor=1000.0, remove_pruned_num_samples_threshold=5,
                 remove_pruned_divisor=1000.0, mean_acc_pruned_model_threshold=0.01,
                 prob_trojan_in_model_min_threshold=0, prob_trojan_in_model_max_threshold=1.0,
                 example_img_format='txt', use_cuda=True, num_duplicate_data_iterations=1,
                 batch_size=100, num_workers=4):
        self.execution_time_start = time.perf_counter()
        self.model_filepath = model_filepath

        # check if it is a directory of filepath
        text = str(tokenizer_filepath)
        if text.endswith(".pt"):
            self.tokenizer_filepath = tokenizer_filepath
            self.tokenizer_dirpath = ''
        else:
            self.tokenizer_dirpath = tokenizer_filepath
            self.tokenizer_filepath = ''

        if self.tokenizer_filepath == '':
            print("INFO 1: self.tokenizer_dirpath=", self.tokenizer_dirpath)
        else:
            print("INFO 1: self.tokenizer_filepath=", self.tokenizer_filepath)

        self.result_filepath = result_filepath
        self.scratch_dirpath = scratch_dirpath
        self.examples_dirpath = examples_dirpath
        self.example_img_format = example_img_format
        if torch.cuda.is_available() and use_cuda:
            self.use_cuda = use_cuda
            self.use_amp = True
        else:
            self.use_cuda = False
            self.use_amp = False

        self.model_dirpath = os.path.dirname(model_filepath)
        self.gt_model_label_filepath = os.path.join(self.model_dirpath, 'ground_truth.csv')
        self.gt_model_label = -1

        # parameters for data loader
        self.num_duplicate_data_iterations = num_duplicate_data_iterations
        self.batch_size = batch_size
        self.num_workers = num_workers

        #################################
        self.pruning_method = pruning_method  # remove or reset or trim
        self.sampling_method = sampling_method  # random or targeted or uniform sampling
        self.ranking_method = ranking_method  # 'L1'  # L1, L2, Linf, STDEV
        self.num_samples = num_samples  # nS=5 or  10 or 15 or 25 was tested
        # set the number of images used
        self.num_images_used = num_images_used  # nD=10,20,30,40 was tested

        self.linear_regression_filepath = linear_regression_filepath  # None

        # Prune params
        self.trim_pruned_amount = trim_pruned_amount  # 0.5
        self.trim_pruned_multiplier = trim_pruned_multiplier  # 4.0
        self.trim_pruned_divisor = trim_pruned_divisor  # 1000.0

        # Reset params
        self.reset_pruned_divisor = reset_pruned_divisor  # 1000.0

        # Remove params
        self.remove_pruned_num_samples_threshold = remove_pruned_num_samples_threshold  # 5
        self.remove_pruned_divisor = remove_pruned_divisor  # 1000.0

        # Trojan detection params
        self.mean_acc_pruned_model_threshold = mean_acc_pruned_model_threshold  # 0.01
        self.prob_trojan_in_model_min_threshold = prob_trojan_in_model_min_threshold  # 0
        self.prob_trojan_in_model_max_threshold = prob_trojan_in_model_max_threshold  # 1.0

        # these values are computed from each architecture by 1/(min number of filters per layer) - rounded up at the second decimal
        # this guarantees that at least one filter is removed from each layer
        # # TODO: Update coef for 'LstmLinear', 'GruLinear', 'Linear', 'FCLinear'
        # TODO update coef for 'BERTLinear', 'DistilBERTLinear', 'MobileBERTLinear', 'RoBERTaLinear'
        self.min_one_filter = {"shufflenet1_0": 0.05, "shufflenet1_5": 0.05, "shufflenet2_0": 0.05,
                               "inceptionv1(googlenet)": 0.07, "inceptionv3": 0.04, "resnet18": 0.03,
                               "resnet34": 0.03, "resnet50": 0.03, "resnet101": 0.03, "resnet152": 0.03,
                               "wide_resnet50": 0.03, "wide_resnet101": 0.03,
                               "squeezenetv1_0": 0.21, "squeezenetv1_1": 0.15, "mobilenetv2": 0.07,
                               "densenet121": 0.04, "densenet161": 0.03, "densenet169": 0.04, "densenet201": 0.04,
                               "vgg11_bn": 0.03, "vgg13_bn": 0.03, "vgg16_bn": 0.03,
                               "GruLinear": 0.1, "LstmLinear": 0.1, "Linear": 0.1, "FCLinear": 0.1,
                               "BERTLinear": 0.1, "DistilBERTLinear": 0.1, "MobileBERTLinear":0.1, "RoBERTaLinear":0.1}
        # -------------------------------------------------------------------
        # NER Setup ---------------------------------------------------------
        # -------------------------------------------------------------------

        # decide which model architecture is represented by the provided AI Model
        a = model_classifier(model_filepath)
        # determine the AI model architecture based on loaded class type
        self.model_class_str, self.model_architecture = a.classify_architecture("cpu")
        self.model_size, self.model_type, self.min_model_size_delta = a.get_filesize(self.model_architecture)
        self.model_name = a.get_model_name(self.model_type)
        #self.model_name, self.model_type, self.min_model_size_delta = a.classify_type(self.model_architecture)
        print('model_type: %s \t' % self.model_type)
        print('model name: %s \n' % self.model_name)
        print('file size delta between a model and the reference model: %s\n' % self.min_model_size_delta)
        #model_architecture = a.switch_architecture(self.model_type)
        print('classified the model as:\t', self.model_architecture)
        print('model size: \t', a.model_size)
        self.ref_model_size = a.model_size + self.min_model_size_delta
        print('reference model size: \t', self.ref_model_size)

        print('pruning_method (PM):', self.pruning_method, ' sampling method (SM):', self.sampling_method,
              ' ranking method (RM):',
              self.ranking_method)
        print('num_samples (nS):', self.num_samples, ' num_sentiment_text_used (nD):', self.num_images_used)

        self.scratch_filepath = os.path.join(self.scratch_dirpath, self.model_name + '_log.csv')

        # to avoid messages about serialization on cpu
        torch.nn.Module.dump_path = 'False'

        ###############################################################
        # read the config file and load the tokenizer
        # create the list of example filenames from the provided clean data
        self.example_filenames = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
        self.example_filenames.sort()  # ensure file ordering

        # load the config file to retrieve parameters
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)

        embedding_name = None
        if config['embedding']:
            embedding_name = config['embedding']

        if self.tokenizer_filepath == '' and self.tokenizer_dirpath != '':
            if embedding_name == 'DistilBERT':
                self.tokenizer_filepath = os.path.join(self.tokenizer_dirpath, 'DistilBERT-distilbert-base-cased.pt')
            elif embedding_name == 'BERT':
                self.tokenizer_filepath = os.path.join(self.tokenizer_dirpath, 'BERT-bert-base-uncased.pt')
            elif embedding_name == 'MobileBERT':
                self.tokenizer_filepath = os.path.join(self.tokenizer_dirpath,
                                                       'MobileBERT-google-mobilebert-uncased.pt')
            elif embedding_name == 'RoBERTa':
                self.tokenizer_filepath = os.path.join(self.tokenizer_dirpath, 'RoBERTa-roberta-base.pt')
            else:
                print("Unknown embedding name:", embedding_name)
                sys.exit(2)

        print('INFO 3: tokenizer_dirpath:', self.tokenizer_dirpath)
        print('INFO 3: tokenizer_filepath:', self.tokenizer_filepath)

        print('Source dataset name = "{}"'.format(config['source_dataset']))
        if 'data_filepath' in config.keys():
            print('Source dataset filepath = "{}"'.format(config['data_filepath']))

        # Load the provided tokenizer
        # TODO: Should use this for evaluation server
        self.tokenizer = torch.load(self.tokenizer_filepath)

        # Or load the tokenizer from the HuggingFace library by name
        # embedding_flavor = config['embedding_flavor']
        # if config['embedding'] == 'RoBERTa':
        #    self.tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True,
         #                                                          add_prefix_space=True)
        # else:
        #    self.tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)

        # set the padding token if its undefined
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # identify the max sequence length for the given embedding
        if config['embedding'] == 'MobileBERT':
            self.max_input_length = self.tokenizer.max_model_input_sizes[self.tokenizer.name_or_path.split('/')[1]]
        else:
            self.max_input_length = self.tokenizer.max_model_input_sizes[self.tokenizer.name_or_path]

        use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
        # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
        # Note, should NOT use_amp when operating with MobileBERT

        #######################################################################################
        # read the ground truth label
        if os.path.isfile(self.gt_model_label_filepath):
            # read the label
            with open(self.gt_model_label_filepath) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    print('ground_truth.csv - is poisoned?:', row[0])
                    self.gt_model_label = int(row[0])
        else:
            print('missing ground truth label for the model')
            self.gt_model_label = -1

        with open(self.scratch_filepath, 'a') as fh:
            fh.write("{}, ".format(self.model_name))
            fh.write("model_filepath, {}, ".format(self.model_filepath))
            fh.write("model_size, {}, ".format(a.model_size))
            fh.write("ref_model_size, {:.4f}, ".format(self.ref_model_size))
            fh.write("delta_model_size, {:.4f}, ".format(self.min_model_size_delta))
            fh.write("gt_model_label, {}, ".format(self.gt_model_label))

        self.example_filenames = self._configure_example_filenames()
        self.sampling_probability = self._configure_prune_sampling_probability()

        if os.path.isfile(self.linear_regression_filepath):
            self.trained_coef = read_regression_coefficients(self.linear_regression_filepath, self.model_name)
            print('loaded pre-computed linear regression coefficients; number of coeff:', len(self.trained_coef ))
        else:
            self.trained_coef = None


    # Updates trojan detection parameters based on optimal configuration CSV. This overrides values taken from command-line.
    def update_configuration_from_optimal_configuration_csv_filepath(self, optimal_configuration_csv_filepath):
        if os.path.isfile(optimal_configuration_csv_filepath):
            coef = [-1]
            with open(optimal_configuration_csv_filepath) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')

                architecture_name_index = -1
                num_images_used_index = -1
                num_samples_index = -1
                pruning_method_index = -1
                ranking_method_index = -1
                sampling_method_index = -1
                pruning_probability_index = -1
                row_index = 0

                for row in readCSV:
                    # search header for indices
                    if row_index == 0:
                        col_index = 0
                        flag = True
                        start = end = -1

                        for elem in row:
                            elem = elem.strip().lower()
                            if 'architecture' in elem:
                                architecture_name_index = col_index
                            if 'number of eval images' in elem:
                                num_images_used_index = col_index
                            if 'number of samples' in elem:
                                num_samples_index = col_index
                            if 'pruning method' in elem:
                                pruning_method_index = col_index
                            if 'ranking method' in elem:
                                ranking_method_index = col_index
                            if 'sampling method' in elem:
                                sampling_method_index = col_index
                            if 'pruning probability' in elem:
                                pruning_probability_index = col_index

                            # parse the header to determine start and end of the coefficients
                            if elem.startswith('b') and len(elem) < 4:  # support for nS<99
                                if flag:
                                    start = col_index
                                    flag = False
                                else:
                                    end = col_index

                            col_index = col_index + 1
                        coef = [-1] * (end + 1 - start)
                    else:
                        if self.model_name == row[architecture_name_index]:
                            print('Found model architecture: {}'.format(self.model_name))
                            self.num_images_used = int(row[num_images_used_index])
                            self.num_samples = int(row[num_samples_index])
                            self.pruning_method = row[pruning_method_index]
                            self.ranking_method = row[ranking_method_index]
                            self.sampling_method = row[sampling_method_index]
                            self.sampling_probability = float(row[pruning_probability_index])

                            for col_index in range(start, end + 1):
                                if row[col_index] != '':
                                    coef[col_index - start] = float(row[col_index])

                    row_index = row_index + 1
                self.trained_coef = [value for value in coef if value != -1]

                print('Found coef: {}'.format(self.trained_coef))
                print('pruning_method (PM):', self.pruning_method, ' sampling method (SM):', self.sampling_method,
                      ' ranking method (RM):',
                      self.ranking_method)
                print('num_samples (nS):', self.num_samples, ' num_images_used (nD):', self.num_images_used)
                print('Sampling probability: {}'.format(self.sampling_probability))

    # The function will gather the image file names from the examples directory, that are available for each model
    def _configure_example_filenames(self):
        self.example_filenames = [os.path.join(self.examples_dirpath, fn) for fn in os.listdir(self.examples_dirpath) if
                                  fn.endswith(self.example_img_format) and not fn.__contains__('_tokenized')]

        self.example_filenames.sort()

        num_images_avail = len(self.example_filenames)
        with open(self.scratch_filepath, 'a') as fh:
            fh.write("num_ner_text_avail, {}, ".format(num_images_avail))
            fh.write("num_ner_text_used, {}, ".format(self.num_images_used))
        print('number of ner_text available for eval per model:', num_images_avail)

        if num_images_avail < self.num_images_used:
            print('WARNING - corrected: ', num_images_avail, ' is less than ', self.num_images_used)
            self.num_images_used = num_images_avail


        step = num_images_avail // self.num_images_used
        temp_idx = []
        for i in range(step // 2, num_images_avail, step):
            if len(temp_idx) < self.num_images_used:
                temp_idx.append(i)

        example_filenames = [self.example_filenames[i] for i in temp_idx]

        print('returning: len(example_filenames): ', len(example_filenames))

        return example_filenames

    def _configure_prune_sampling_probability(self):
        # adjustments per model type for trim method
        sampling_probability = 0.0

        if 'trim' in self.pruning_method:
            # sampling_probability = 0.4
            # print('SET sampling_probability:', sampling_probability)
            sampling_probability = self.trim_pruned_multiplier * self.trim_pruned_amount * np.ceil(
                self.trim_pruned_divisor / self.num_samples) / self.trim_pruned_divisor  # 0.2 # 1.0/num_samples #0.2 # before 0.4

        if 'reset' in self.pruning_method:
            sampling_probability = 0.9
            print('SET reset sampling_probability:', sampling_probability)
            # sampling_probability = np.ceil(self.reset_pruned_divisor / self.num_samples) / self.reset_pruned_divisor

        # adjustments per model type for remove method
        if 'remove' in self.pruning_method:
            if self.num_samples <= self.remove_pruned_num_samples_threshold:
                for key, value in self.min_one_filter.items():
                    if key in self.model_architecture:
                        sampling_probability = value
            else:
                # this is the setup for nS>6
                sampling_probability = np.ceil(
                    self.remove_pruned_divisor / self.num_samples) / self.remove_pruned_divisor

            # there is a random failure of densenet models for sampling_probability larger than 0.02
            # if 'densenet' in self.model_name and sampling_probability > 0.02:
            #     # this value is computed as if for num_samples = 50 --> sampling_probability = 0.02
            #     sampling_probability = 0.02

        return sampling_probability

    def _eval(self, model, preprocessed_data, device):

        use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
        # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
        # Note, should NOT use_amp when operating with MobileBERT

        sum = 0.0

        for index in range(0, preprocessed_data.num_samples):
            # loop over all available examples
            input_ids, attention_mask, labels, labels_mask, labels_tensor, original_words = preprocessed_data.getarrayitem(index, device)

            # predict the text sentiment
            # if use_amp:
            #     with torch.cuda.amp.autocast():
            #         # Classification model returns loss, logits, can ignore loss if needed
            #         _, logits = model(preprocessed_data.input_ids, attention_mask=preprocessed_data.attention_mask, labels=preprocessed_data.labels_tensor)
            # else:
            #     _, logits = model(preprocessed_data.input_ids, attention_mask=preprocessed_data.attention_mask, labels=preprocessed_data.labels_tensor)

            if use_amp:
                with torch.cuda.amp.autocast():
                    # Classification model returns loss, logits, can ignore loss if needed
                    _, logits = model(input_ids, attention_mask=attention_mask,
                                      labels=labels_tensor)
            else:
                _, logits = model(input_ids, attention_mask=attention_mask,
                                  labels=labels_tensor)

            preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
            numpy_logits = logits.cpu().flatten().detach().numpy()

            n_correct = 0
            n_total = 0
            predicted_labels = []
            for i, m in enumerate(labels_mask):
                if m:
                    predicted_labels.append(preds[i])
                    n_total += 1
                    n_correct += preds[i] == labels[i]

            #print('Predictions: {} from Text: "{}"'.format(predicted_labels, original_words))
            assert len(predicted_labels) == len(original_words)
            # print('  logits: {}'.format(numpy_logits))
            sum += n_correct / float(n_total)
            # predicted_labels = []
            # for i, m in enumerate(preprocessed_data.labels_mask):
            #     if m:
            #         predicted_labels.append(preds[i])
            #         n_total += 1
            #         n_correct += preds[i] == preprocessed_data.labels[i]

        # print('Predictions: {} from Text: "{}"'.format(predicted_labels, preprocessed_data.original_words))
        # assert len(predicted_labels) == len(preprocessed_data.original_words)
        #print('  logits: {}'.format(numpy_logits))

        #return n_correct / float(n_total)
        return sum/ float(preprocessed_data.num_samples)

    def prune_model(self):

        preprocessed_data = extended_dataset_ner(self.example_filenames, self.tokenizer, self.max_input_length,
                                       num_iterations=self.num_duplicate_data_iterations)
        #preprocessed_data = self._preprocess_data(dataset)

        #######################################
        # load a model
        #memory_stats('DEBUG: before model_orig')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = 'cpu'
        try:
            model_orig = torch.load(self.model_filepath, map_location=device)
            #memory_stats('DEBUG: after load model_orig')

            # testing
            # acc_pruned_model = self._eval(model_orig, preprocessed_data, device)
            # print('DEBUG: acc_pruned_model:', acc_pruned_model)

            #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # load the classification model and move it to the GPU
            #model_orig = torch.load(self.model_filepath, map_location=torch.device(device))

        except:
            print("Unexpected loading error:", sys.exc_info()[0])
            # close the line
            with open(self.scratch_filepath, 'a') as fh:
                fh.write("\n")
            raise
        #memory_stats('DEBUG: after model_orig')

        params = sum([np.prod(p.size()) for p in model_orig.parameters()])
        print("Before Number of Parameters: %.1fM" % (params / 1e6))
        acc_model = 1.0  # <- TODO: is this right? eval(model_orig, test_loader, result_filepath, model_name)
        print("Before Acc=%.4f\n" % (acc_model))

        #####################################
        acc_pruned_model_shift = []
        pruning_shift = []

        timings = OrderedDict()
        copy_times = list()
        prune_times = list()
        eval_times = list()

        loop_start = time.perf_counter()

        for sample_shift in range(self.num_samples):
            copy_start = time.perf_counter()
            # copy model on CPU !!!
            model = copy.deepcopy(model_orig)
            copy_time = time.perf_counter() - copy_start

            #memory_stats('DEBUG: after deepcopy of model_orig')
            copy_times.append(copy_time)
            print('INFO: ', self.pruning_method, ' pruning method for sample_shift:', sample_shift)
            prune_start = time.perf_counter()

            # Temporary hack for the prune method = remove because  the shufflenet architecture is not supported
            # if 'shufflenet' in self.model_name and 'remove' in self.pruning_method:
            #     self.pruning_method = 'reset'

            try:
                if 'remove' in self.pruning_method:
                    print("pruning method = remove is currently not supported because recursive graph pruning is not supported")
                    # remove_prune_model(model, self.model_architecture, output_transform, sample_shift, self.sampling_method,
                    #              self.ranking_method, self.sampling_probability, self.num_samples)
                if 'reset' in self.pruning_method:
                    print("entering pruning method = reset ")
                    reset_prune_model(model, device, self.model_name, sample_shift, self.sampling_method, self.ranking_method,
                                      self.sampling_probability,
                                      self.num_samples)
                if 'trim' in self.pruning_method:
                    print("pruning method = trim is currently not supported ")
                    # trim_prune_model(model, device, self.model_name, sample_shift, self.sampling_method, self.ranking_method,
                    #            self.sampling_probability,
                    #            self.num_samples, self.trim_pruned_amount)

            except:
                # this is relevant to PM=Remove because it fails for some configurations to prune the model correctly
                print("Unexpected pruning error:", sys.exc_info()[0])
                # close the line
                with open(self.scratch_filepath, 'a') as fh:
                    fh.write("\n")
                raise

            #memory_stats('DEBUG: after pruning deepcopy of model_orig')

            prune_time = time.perf_counter() - prune_start
            prune_times.append(prune_time)

            eval_start = time.perf_counter()
            # load the classification model and move it to the GPU

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # push the pruned model to GPU
            model.to(device)
            acc_pruned_model = self._eval(model, preprocessed_data, device)
            #memory_stats('DEBUG: after eval of pruned model')

            eval_time = time.perf_counter() - eval_start
            eval_times.append(eval_time)
            print('model: ', self.model_filepath, ' acc_model: ', acc_model, ' acc_pruned_model: ', acc_pruned_model)
            acc_pruned_model_shift.append(acc_pruned_model)
            pruning_shift.append(sample_shift)

            del model
            torch.cuda.empty_cache()
            #print('torch.cuda.memory_snapshot():', torch.cuda.memory_snapshot())
            #memory_stats('DEBUG: after torch.cuda.empty_cache()')


        loop_time = time.perf_counter() - loop_start
        timings['loop'] = loop_time
        timings['avg copy'] = statistics.mean(copy_times)
        timings['avg prune'] = statistics.mean(prune_times)
        timings['max eval'] = max(eval_times)
        timings['min eval'] = min(eval_times)
        timings['avg eval'] = statistics.mean(eval_times)

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
        if mean_acc_pruned_model > self.mean_acc_pruned_model_threshold:
            coef_var = stdev_acc_pruned_model / mean_acc_pruned_model
        else:
            coef_var = 0.0

        prob_trojan_in_model = coef_var
        if prob_trojan_in_model < self.prob_trojan_in_model_min_threshold:
            prob_trojan_in_model = 0

        if prob_trojan_in_model > self.prob_trojan_in_model_max_threshold:
            prob_trojan_in_model = 1.0
        print('coef of variation:', coef_var, ' prob_trojan_in_model:', prob_trojan_in_model)

        # round 2 - linear regression coefficients applied to the num_samples (signal measurement)
        # this function should be enabled if  the estimated multiple linear correlation coefficients should be applied
        if self.trained_coef is not None:
            print('INFO: applying pre-computed linear regression coefficients:', self.trained_coef)
            prob_trojan_in_model = linear_regression_prediction(self.trained_coef, acc_pruned_model_shift)

        # stop timing the execution
        execution_time_end = time.perf_counter()

        with open(self.scratch_filepath, 'a') as fh:
            # fh.write("model_filepath, {}, ".format(model_filepath))
            fh.write("number of params, {}, ".format((params / 1e6)))
            fh.write("{}, ".format(self.model_architecture))
            fh.write("{}, ".format(self.pruning_method))
            fh.write("{}, ".format(self.sampling_method))
            fh.write("{}, ".format(self.ranking_method))
            fh.write("{}, ".format(self.num_samples))
            fh.write("{}, ".format(self.num_images_used))
            fh.write("{:.4f}, ".format(self.sampling_probability))
            for i in range(len(acc_pruned_model_shift)):
                fh.write("{:.8f}, ".format(acc_pruned_model_shift[i]))

            fh.write("mean, {:.4f}, ".format(mean_acc_pruned_model))
            fh.write("stdev, {:.4f}, ".format(stdev_acc_pruned_model))
            fh.write("min, {:.4f}, ".format(min_acc_pruned_model))
            fh.write("max, {:.4f}, ".format(max_acc_pruned_model))
            fh.write("coef_var, {:.4f}, ".format(coef_var))
            fh.write("num_min2max_ordered, {}, ".format(num_min2max_ordered))
            fh.write("num_max2min_ordered, {}, ".format(num_max2min_ordered))
            fh.write("slope, {:.4f}, ".format(slope))
            fh.write("prob_trojan_in_model, {:.4f}, ".format(prob_trojan_in_model))
            fh.write("execution time [s], {},".format((execution_time_end - self.execution_time_start)))
            for key in timings.keys():
                fh.write('{} [s], {},'.format(key, timings[key]))
            fh.write('\n')

        # write the result to a file
        with open(self.result_filepath, 'w') as fh:
            fh.write("{}".format(prob_trojan_in_model))

        del model_orig
        return prob_trojan_in_model

    @staticmethod
    def processParameters(default_config_file='./config.txt'):
        parser = configargparse.ArgumentParser(default_config_files=[default_config_file],
                                               description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
        parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

        parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                            default='./model/model.pt')
        parser.add_argument('--tokenizer_filepath', type=str,
                            help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.',
                            default='./model/tokenizer.pt')
        parser.add_argument('--result_filepath', type=str,
                            help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                            default='./output.txt')
        parser.add_argument('--scratch_dirpath', type=str,
                            help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                            default='./scratch')
        parser.add_argument('--examples_dirpath', type=str,
                            help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                            default='./model/clean_example_data')

        # method specific parameters
        parser.add_argument('--pruning_method', type=str,
                            help='The pruning method to use (trim, reset, or remove)',
                            default='reset')
        parser.add_argument('--sampling_method', type=str,
                            help='The sampling method to use (random, targeted, or uniform)',
                            default='targeted')
        parser.add_argument('--ranking_method', type=str,
                            help='The ranking method to use (L1, L2, Linf, or STDEV).',
                            default='L1')
        parser.add_argument('--num_samples', type=int,
                            help='The number of samples (models) to execute with.',
                            default='15')
        parser.add_argument('--num_images_used', type=int,
                            help='The number of images to inference with.',
                            default='100')
        parser.add_argument('--linear_regression_filepath', type=str,
                            help='The linear regression filepath. Used to apply the results of the linear regression.',
                            default='./linear_regression_data/r7_reset_L1_targeted_15_100_0p9_layer0p005.csv')
        parser.add_argument('--trim_pruned_amount', type=float,
                            help='Amount used when calculating the sampling probability for trim.',
                            default='0.5')
        parser.add_argument('--trim_pruned_multiplier', type=float,
                            help='Multiplier used when calculating the sampling probability for trim.',
                            default='4.0')
        parser.add_argument('--trim_pruned_divisor', type=float,
                            help='Divisor used when calculating the sampling probability for trim.',
                            default='1000.0')
        parser.add_argument('--reset_pruned_divisor', type=float,
                            help='Divisor used when calculating the sampling probability for reset.',
                            default='1000.0')
        parser.add_argument('--remove_pruned_num_samples_threshold', type=int,
                            help='The threshold for comparing the number of samples when calculating the sampling probability for remove.',
                            default='5')
        parser.add_argument('--remove_pruned_divisor', type=float,
                            help='Divisor used when calculating the sampling probability for remove.',
                            default='1000.0')
        parser.add_argument('--mean_acc_pruned_model_threshold', type=float,
                            help='Mean threshold used when comparing accuracy of pruned model.',
                            default='0.01')
        parser.add_argument('--prob_trojan_in_model_min_threshold', type=float,
                            help='Minimum threshold for comparing probability of a trojaned model.',
                            default='0.0')
        parser.add_argument('--prob_trojan_in_model_max_threshold', type=float,
                            help='Maximum threshold for comparing probability of a trojaned model.',
                            default='1.0')
        parser.add_argument('--num_duplicate_data_iterations', type=int,
                            help='Number of iterations to run when processing data (values >1 will duplicate the data).',
                            default='1')
        parser.add_argument('--batch_size', type=int,
                            help='Batch size of data to run through the model.',
                            default='100')
        parser.add_argument('--num_workers', type=int,
                            help='Number of parallel workers to load data.',
                            default='4')
        parser.add_argument('--no_cuda',
                            help='Specifies to disable using CUDA',
                            dest='use_cuda', action='store_false')
        parser.add_argument('--optimal_configuration_csv_filepath',
                            help='Specifies an optimal configuration CSV file, '
                                 'this file overrides the following configuration settings: '
                                 'number of eval images, number of samples, pruning method, '
                                 'ranking method, sampling method, and the linear regression filepath',
                            default=None)
        parser.set_defaults(use_cuda=True)

        args = parser.parse_args()
        parser.print_values()

        #model_dirpath = os.path.dirname(args.model_filepath)
        model_filepath = args.model_filepath
        tokenizer_filepath = args.tokenizer_filepath
        result_filepath = args.result_filepath
        scratch_dirpath = args.scratch_dirpath
        examples_dirpath = args.examples_dirpath

        print('model_filepath = {}'.format(model_filepath))
        print('tokenizer_filepath = {}'.format(tokenizer_filepath))
        print('result_filepath = {}'.format(result_filepath))
        print('scratch_dirpath = {}'.format(scratch_dirpath))
        print('examples_dirpath = {}'.format(examples_dirpath))


        trojanDetector = TrojanDetectorNER(model_filepath,
                                           tokenizer_filepath,
                                           result_filepath,
                                           scratch_dirpath, examples_dirpath,
                                           args.pruning_method, args.sampling_method,
                                           args.ranking_method, args.num_samples,
                                           args.num_images_used, args.linear_regression_filepath,
                                           args.trim_pruned_amount, args.trim_pruned_multiplier,
                                           args.trim_pruned_divisor, args.reset_pruned_divisor,
                                           args.remove_pruned_num_samples_threshold, args.remove_pruned_divisor,
                                           args.mean_acc_pruned_model_threshold,
                                           args.prob_trojan_in_model_min_threshold,
                                           args.prob_trojan_in_model_max_threshold, 'txt', args.use_cuda,
                                           args.num_duplicate_data_iterations, args.batch_size, args.num_workers)

        if args.optimal_configuration_csv_filepath is not None:
            print('INFO: loading optimal configuration:')
            trojanDetector.update_configuration_from_optimal_configuration_csv_filepath(
                args.optimal_configuration_csv_filepath)

        return trojanDetector


# used for debugging the virtual memory problem
def memory_stats(text_for_print: str):
    print(text_for_print)
    # gives a single float value
    print('cpu_percent():', psutil.cpu_percent())
    # gives an object with many fields
    print('virtual_memory():', psutil.virtual_memory())
    # you can convert that object to a dictionary
    dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    print('virtual_memory().percent:', psutil.virtual_memory().percent)
    # you can calculate percentage of available memory
    percent_available = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
    print('percent memory available:', percent_available)

if __name__ == '__main__':
    entries = globals().copy()

    print('torch version: %s \n' % (torch.__version__))

    trojan_detector = TrojanDetectorNER.processParameters()
    trojan_detector.prune_model()
