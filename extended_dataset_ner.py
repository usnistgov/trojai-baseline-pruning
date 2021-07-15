
"""
Disclaimer
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
"""
__author__ = "Tim Blattner"
__copyright__ = "Copyright 2020, The IARPA funded TrojAI project"
__credits__ = ["Peter Bajcsy", "Michael Majurski", "Tim Blattner", "Derek Juba", "Walid Keyrouz"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Peter Bajcsy"
__email__ = "peter.bajcsy@nist.gov"
__status__ = "Research"

import torch
from skimage import io, transform
from itertools import repeat

from nltk.corpus import wordnet
import random

"""
This class supports creating datasets in PyTorch
The code was adopted from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

class extended_dataset_ner(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  #def __init__(self, list_IDs, labels):
  def __init__(self, list_filenames, tokenizer, max_input_length, num_iterations=1):

      #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

      input_ids = []
      attention_mask = []
      labels = []
      labels_mask = []
      original_words = []

      num_samples = 0
      for fn in list_filenames:
          # For this example we parse the raw txt file to demonstrate tokenization.
          if fn.endswith('_tokenized.txt'):
              continue

          # load the example
          _original_words = []
          original_labels = []
          with open(fn, 'r') as fh:
              lines = fh.readlines()
              for line in lines:
                  split_line = line.split('\t')
                  word = split_line[0].strip()
                  label = split_line[2].strip()

                  _original_words.append(word)
                  original_labels.append(int(label))

          # Select your preference for tokenization
          #input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
          # input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words,
          #                                                                                   original_labels,
          #                                                                                   max_input_length)
          _input_ids, _attention_mask, _labels, _labels_mask = manual_tokenize_and_align_labels(tokenizer, _original_words,
                                                                                            original_labels,
                                                                                            max_input_length)

          input_ids.append(_input_ids)
          attention_mask.append(_attention_mask)
          labels.append(_labels)
          labels_mask.append(_labels_mask)
          original_words.append((_original_words))
          num_samples += 1
          ###############################################
          # input_ids = torch.as_tensor(input_ids)
          # attention_mask = torch.as_tensor(attention_mask)
          # labels_tensor = torch.as_tensor(labels)
          #
          # if device != 'cpu':
          #     input_ids = input_ids.to(device)
          #     attention_mask = attention_mask.to(device)
          #     labels_tensor = labels_tensor.to(device)
          #
          # # Create just a single batch
          # input_ids = torch.unsqueeze(input_ids, axis=0)
          # attention_mask = torch.unsqueeze(attention_mask, axis=0)
          # labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

      #############################################################
      'Initialization'
      self.input_ids = input_ids
      self.attention_mask = attention_mask
      #self.labels_tensor = labels_tensor
      self.original_words = original_words
      self.labels_mask = labels_mask
      self.labels = labels
      self.num_samples = num_samples

      self.tokenizer = tokenizer
      self.max_input_length = max_input_length
      self.num_iterations = num_iterations
      print('Total length = {}'.format(len(self.input_ids)))

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.input_ids)

  def getarrayitem(self, index, device):
      if index < 0 or index >= self.num_samples:
          print("ERROR: index is out of range:", index, " range=[", 0, ", ", self.num_samples)

      input_ids = torch.as_tensor(self.input_ids[index])
      attention_mask = torch.as_tensor(self.attention_mask[index])
      labels_tensor = torch.as_tensor(self.labels[index])

      if device != 'cpu':
          input_ids = input_ids.to(device)
          attention_mask = attention_mask.to(device)
          labels_tensor = labels_tensor.to(device)

      # Create just a single batch
      input_ids = torch.unsqueeze(input_ids, axis=0)
      attention_mask = torch.unsqueeze(attention_mask, axis=0)
      labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

      self.labels_tensor = labels_tensor

      return input_ids, attention_mask, self.labels[index],  self.labels_mask[index], self.labels_tensor, self.original_words[index]


# Adapted from: https://github.com/huggingface/transformers/blob/2d27900b5d74a84b4c6b95950fd26c9d794b2d57/examples/pytorch/token-classification/run_ner.py#L318
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# Note, this requires 'fast' tokenization
def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True,
                                 max_length=max_input_length)
    labels = []
    label_mask = []

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx

    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask


# Alternate method for tokenization that does not require 'fast' tokenizer (all of our tokenizers for this round have fast though)
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# This is a similar version that is used in trojai.
def manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    labels = []
    label_mask = []
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    tokens = []
    attention_mask = []

    # Add cls token
    tokens.append(cls_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)

    for i, word in enumerate(original_words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = original_labels[i]

        # Variable to select which token to use for label.
        # All transformers for this round use bi-directional, so we use first token
        token_label_index = 0
        for m in range(len(token)):
            attention_mask.append(1)

            if m == token_label_index:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)

    if len(tokens) > max_input_length - 1:
        tokens = tokens[0:(max_input_length - 1)]
        attention_mask = attention_mask[0:(max_input_length - 1)]
        labels = labels[0:(max_input_length - 1)]
        label_mask = label_mask[0:(max_input_length - 1)]

    # Add trailing sep token
    tokens.append(sep_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return input_ids, attention_mask, labels, label_mask

