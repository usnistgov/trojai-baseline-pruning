"""
Disclaimer
This software was developed by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government and is being made available as a public service. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States.  This software may be subject to foreign copyright.  Permission in the United States and in foreign countries, to the extent that NIST may hold copyright, to use, copy, modify, create derivative works, and distribute this software and its documentation without fee is hereby granted on a non-exclusive basis, provided that this notice and disclaimer of warranty appears in all copies.
THE SOFTWARE IS PROVIDED 'AS IS' WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND FREEDOM FROM INFRINGEMENT, AND ANY WARRANTY THAT THE DOCUMENTATION WILL CONFORM TO THE SOFTWARE, OR ANY WARRANTY THAT THE SOFTWARE WILL BE ERROR FREE.  IN NO EVENT SHALL NIST BE LIABLE FOR ANY DAMAGES, INCLUDING, BUT NOT LIMITED TO, DIRECT, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES, ARISING OUT OF, RESULTING FROM, OR IN ANY WAY CONNECTED WITH THIS SOFTWARE, WHETHER OR NOT BASED UPON WARRANTY, CONTRACT, TORT, OR OTHERWISE, WHETHER OR NOT INJURY WAS SUSTAINED BY PERSONS OR PROPERTY OR OTHERWISE, AND WHETHER OR NOT LOSS WAS SUSTAINED FROM, OR AROSE OUT OF THE RESULTS OF, OR USE OF, THE SOFTWARE OR SERVICES PROVIDED HEREUNDER.
"""
__author__ = "Tim Blattner"
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
from torchvision import transforms
import csv
import time
from trojan_detector import TrojanDetector

from linear_regression import read_regression_coefficients, linear_regression_prediction
from model_classifier_round2 import model_classifier
# from my_dataset import my_dataset
from extended_dataset import extended_dataset
from remove_prune import prune_model
from reset_prune import reset_prune_model
from trim_prune import trim_model

"""
This class is designed for detecting trojans in TrojAI Round 2 Challenge datasets
see https://pages.nist.gov/trojai/docs/data.html#round-2
This code is an adjusted version of the trojan detector for the Round 1 of the TrojAI challenge
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

####################################################################################
if __name__ == '__main__':
    entries = globals().copy()

    print('torch version: %s \n' % (torch.__version__))

    transform = None
        # transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.CenterCrop(224),
        # transforms.ToTensor()])

    trojan_detector = TrojanDetector.processParameters(transform, default_config_file='config_files/round4.config')
    trojan_detector.prune_model()
