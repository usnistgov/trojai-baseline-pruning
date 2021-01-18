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

""" 
This class returns the AI architecture type based on the model file size.
It was designed for TrojAI Round 1 Challenge datasets consisting of three AI architectures
resnet50, densent121, and inceptionV3
"""


class model_classifier:
    # centroid file size values per architecture model
    resnet50_size = float(94.381)  # in MB
    inceptionv3_size = float(87.47)  # in MB
    densenet121_size = float(28.368)  # in MB
    model_filepath = ''
    model_size = 0
    min_model_size_delta = sys.float_info.max #(1.7976931348623157e+308)

    def __init__(self, model_filepath: object) -> object:
        self.model_filepath = model_filepath
        self.model_size = 0

    ###########################
    def switch_architecture(self, argument):
        """
        @argument integer [1,3]
        :return: string
        """
        switcher = {
            1: "resnet50",
            2: "inceptionv3",
            3: "densenet121"
        }
        return switcher.get(argument, "Invalid architecture")


    ########################################################
    def classify_type(self):
        """
        This method classifies a model based on its file size
        :param model_filepath: path to the model
        :return: integer [1,3] with mapping defined in the class model_classifier
        """
        a = model_classifier(self.model_filepath)

        size = os.stat(self.model_filepath).st_size
        size = size / 1000000  # in MB
        print('Model size in MB: {}'.format(size))
        self.model_size = size

        if abs(size - a.resnet50_size) < abs(size - a.inceptionv3_size):
            if abs(size - a.resnet50_size) < abs(size - a.densenet121_size):
                print('RESNET 50 Model is analyzed')
                model_type = 1
                min_model_size_delta =  a.resnet50_size - size
            else:
                print('DENSENET 121 Model is analyzed')
                model_type = 3
                min_model_size_delta = a.densenet121_size - size
        elif abs(size - a.inceptionv3_size) < abs(size - a.densenet121_size):
            print('INCEPTIONV3 Model is analyzed')
            model_type = 2
            min_model_size_delta =  a.inceptionv3_size - size
        else:
            print('DENSENET Model is analyzed')
            model_type = 3
            min_model_size_delta =  a.densenet121_size - size

        #print('classified the model as:\t', a.switch_architecture(model_type))
        return model_type, min_model_size_delta

