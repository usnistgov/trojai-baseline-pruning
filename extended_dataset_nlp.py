
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

class extended_dataset_nlp(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  #def __init__(self, list_IDs, labels):
  def __init__(self, list_filenames, tokenizer, max_input_length, num_iterations=1):

      labels = []
      all_filenames = []

      for i in repeat(None, num_iterations):
          for fn in list_filenames:
              #print('processing image:', fn)
              # extract the label from the file name
              # example file name: class_0_example_1.png
              if 'class_' in fn:
                  start = fn.rfind('class_') + 6
                  end = fn.rfind('_example', start)
                  image_class = int(fn[start:end])
              else:
                  print('ERROR: could not parse the image label from the image file name')
                  image_class = 0

              #print('image_class:', image_class)
              labels.append(image_class)
              all_filenames.append(fn)

      'Initialization'
      self.labels = labels
      self.list_IDs = all_filenames
      self.tokenizer = tokenizer
      self.max_input_length = max_input_length
      self.num_iterations = num_iterations
      print('Total length = {}'.format(len(self.list_IDs)))

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        with open(ID, 'r') as fh:
            text = fh.read()

        #print('text before:', text)
        flagAntonym = False
        flagReplacement = False
        if self.num_iterations > 1:
            words = text.split()
            # remove commas or periods at the end of some words
            # they cannot be removed because the periods or commas could be triggers and the assembly would have to change
            # for i in range(len(words)):
            #     if words[i].endswith(',') or words[i].endswith('.'):
            #         words[i] = words[i][:-1]

            #print('words', words)
            max_length_word = len(max(words))
            flagReplacement = True

            # choose whether synonyms or antonyms are replaced
            selection = random.randint(0,1)
            if selection == 0:
                flagAntonym = True
                countAntonyms = 0
                max_length_word = max_length_word - 2 # consider more words in order to avoid zero countAntonyms
            else:
                flagAntonym = False
                countSynonyms = 0

            #print('max_legth_words', max_length_word)
            # print('flagAntonym:', str(flagAntonym))
            # print('flagSynonym:', str(flagSynonym))

            for i in range(len(words)):
                if len(words[i]) >= max_length_word:
                    #print('selected word:', words[i])
                    antonyms = []
                    synonyms = []
                    for syn in wordnet.synsets(words[i]):
                        for l in syn.lemmas():
                            # include only synonyms that are distinct!
                            if words[i] not in l.name():
                                synonyms.append(l.name())
                            if l.antonyms():
                                antonyms.append(l.antonyms()[0].name())

                    if flagAntonym and len(antonyms) > 0:
                        # replace antonyms
                        idx = random.randint(0, len(antonyms)-1)
                        words[i] = antonyms[idx]
                        countAntonyms += 1
                        #print('replaced antonym:', words[i])

                    if not flagAntonym:
                        # replace synonyms
                        if len(synonyms) > 0:
                            flagSynonym = True
                            idx = random.randint(0, len(synonyms)-1)
                            words[i] = synonyms[idx]
                            countSynonyms += 1
                            #print('replaced synonym:', words[i])

            # re-assemble the sentence
            text = ''
            for i in range(len(words)):
                text += words[i] +' '

        # if flagAntonym:
        #    print('replaced antonyms - count:', countAntonyms)
        # else:
        #    print('replaced synonyms - count:', countSynonyms)
        # print('text after:', text)

        results = self.tokenizer(text, max_length=self.max_input_length - 2, padding=True, truncation=True, return_tensors='pt')

        # if self.transform is not None:
        #     results = self.transform(results)

        y = self.labels[index]

        # flip the label if iterations > 1, antonyms replaced words and the count is larger than zero
        if flagReplacement and flagAntonym and countAntonyms > 0:
            #print('label before:', y)
            if y == 0:
                y = 1
            else:
                y = 0
            #print('label after:', y)

        return results.data['input_ids'], results.data['attention_mask'], y, ID
