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

import csv
import os

"""
This class can load  computed linear regression coefficients that convert the signal extracted from 
pruned models nto a probability of a model being trained with trojans.

There is also an option of hard-coding the coefficients and estimating the probability of a model being 
trained with trojans form those coefficients
"""

#  round 1
# order:     Intercept    Smallest L1	Largest L1	smaller L1	Middle L1	Large L1

# PM=RESET from run 13
r1_resnet50_coef = [0.320174976,	0.398784844,	0.282845294,	0.150766857,	-0.783193088,	-0.313062887]
r1_densenet121_coef = [-0.228030328,	0.460795499,	0.397504035,	-0.011112314,	0.090053982,	-0.410604721]
r1_inceptionv3_coef = [0.812242541	,-0.386845105	,0.018517737	,0.553447564	,-0.226538181,	-1.044530917]

########################################################################################
# round 2
# order:     Intercept,   Smallest L1,	Smaller L1,	Middle L1,	Large L1,  Largeflagst L1, Predicted Errors, Total classified models, expected ratio error
#conv + batchnorm RESET - derived from desktop run 11 (30 samples per model)
resnet18_coef =[1.405468687,	-0.693805054,	-0.254816291,	-0.313301938,	0.190925648,	-0.232343903,9,23,0.391304348]
resnet34_coef = [0.182248025,	0.795320153,	-0.132060188,	-1.48348815,	1.79110939,	-0.329270459,7,22,0.318181818]
resnet50_coef = [0.625603576,	-0.073081334,	-0.608279404,	0.814795372,	-0.115098854,	-1.93948493,9,22,0.409090909]
resnet101_coef = [-0.428548901,	2.245414576,	0.140453357,	-4.108781383,	6.320812896,	-5.303709915,4,17,0.235294118]
resnet152_coef = [3.579060432,	-2.053657612,	-0.872069566,	0.335902255,	-0.095363086,	-3.393795497,3,16,0.1875]
wide_resnet50_coef = [0.353011231,	0.819794032,	-0.692952163,	0.960427272,	-1.771237153,0.560487656,2,11,0.181818182]
wide_resnet101_coef = [0.574739885,	0.78493846,	2.402751746,	-1.030062656,	-17.84604154,	12.54632744, 3, 13, 0.230769231]
densenet121_coef = [1.329312232,	-0.7236845,	-0.465387464,	0.77143054,	1.164318783,	-2.259319427, 5, 16, 0.3125]
densenet161_coef = [2.093512075,	-2.032625038,	-0.456937124,	2.044136205,	-3.386358828,	3.419221041,3,25,0.12]
densenet169_coef = [1.678007044,	-1.749696208,	0.650937397,	-0.075956683,	-0.616096946,-0.517974907,6,21,0.285714286]
densenet201_coef = [3.927563217,-2.461180594,0.044285662,	-2.058776512,	1.664598121,-1.187018952,8,26,0.307692308]
inceptionv1_googlenet_coef = [-0.472642919,	1.512250451,	-0.515413729,	-0.151376682,	-0.413167591,	1.608774346,9,25,0.36]
inceptionv3_coef = [2.112718025,	-1.986934903,	-0.47734445,	1.101587547,	0.824035464,	0.942754044,4,15,0.266666667]
squeezenetv1_0_coef = [2.49585147,	-0.32917303,-2.576915933,	-0.213681555,	0.097615573,0.695980782, 1, 13, 0.076923077]
squeezenetv1_1_coef = [1.05084445,	0.187791027,	-1.709001047,	0.70417226,	-0.362736183,	0.239133205, 2,17, 0.117647059]
mobilenetv2_coef = [1.859163448,	-1.654698765,	-0.815951345,	0.900649963,	-0.033159165,	0.745316607, 7,22, 0.318181818]
shufflenet1_0_coef = [0.412394767,	-0.858929412,	-0.496782557,	3.14513177,	-0.537687767,	-2.072410641, 4, 17, 0.235294118]
shufflenet1_5_coef = [0.583011283,	2.975794479,	-5.002456736,	-1.424399725,	0.371193388,	1.885431256, 8,26, 0.307692308]
shufflenet2_0_coef = [0.340863079,	-0.863363666,	7.527884627,	-4.156295846,	-1.062087204,	0.144011985, 9, 22, 0.409090909]
vgg11_bn_coef = [0.230613878,	1.367439725,	-0.50955596,	-0.995895346,	0.706347991,	-0.76467083, 5,31,0.161290323]
vgg13_bn_coef = [-1.155497827,	2.423172395,	-0.881188419,	1.024100103,	-3.190232657,	0.113376547,1, 13, 0.076923077]
vgg16_bn_coef = [0.789774073,	0.393473139,	-0.335988859,	-1.61624675,	2.299162798,	-3.52933684, 3, 16, 0.1875]


"""
This method loads the multiple linear regression coefficients from a file generated
by linear_regression_fit.py for a target_model_name (architecture)
"""
def read_regression_coefficients(linear_regression_filepath, target_model_name ):
    coef = [-1]
    # check if the file with regression coefficients exists
    if os.path.isfile(linear_regression_filepath):
        # read the coefficients
        with open(linear_regression_filepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            index = 0
            architecture_name_index = -1
            for row in readCSV:
                print(index,'th row:', row)
                if index == 0:
                    # parse the header to determine start and end of the coefficients
                    col = 0
                    flag = True
                    start = end = -1
                    for elem in row:
                        elem = elem.strip()
                        if 'architecture' in elem.lower():
                            architecture_name_index = col

                        if elem.startswith('b') and len(elem) < 6: # TODO Be aware of the support for nS<9999 !!!!!!
                            if flag:
                                start = col
                                flag = False
                            else:
                                end = col
                        col = col + 1

                    print('LR coeff: start = ', start, ' end=', end)
                    if start == -1 or end == -1:
                        print('Error: missing LR coefficients in the file:', linear_regression_filepath)
                        return coef

                    coef = [-1] * (end+1-start)
                else:
                    # parse the 0th element to determine model name
                    model_name = row[architecture_name_index]
                    print('model_name:',model_name)
                    if model_name in target_model_name:
                        for col in range(start, end+1):
                            coef[col-start] = float(row[col])

                index = index + 1
    else:
        print('Error: missing file with LR coefficients:', linear_regression_filepath)
        return coef

    #sanity check that the model is in the LR file
    count = 0
    for val in range(len(coef)):
        if coef[val] == -1:
            count = count + 1
    if count == len(coef):
        print('ERROR: missing target_model_name:', target_model_name, ' in the file with LR coefficients:', linear_regression_filepath)

    print(coef)
    return coef

"""
This method predicts the probability of a model trained with trojan based on the precomputed coefficients
(estimated_coef). 
It assumes that the multiple linear regression coefficients (estimated_coef) have been loaded from a file.
"""
def linear_regression_prediction(estimated_coef, acc_pruned_model_shift):
    prob_trojan_in_model = 0.0
    if len(acc_pruned_model_shift) != len(estimated_coef)-1:
        print('ERROR: the number of accuracy samples per model does not match the number of estimated coefficients')
        print('len(acc_pruned_model_shift):', len(acc_pruned_model_shift), 'len(estimated_coef)-1:', (len(estimated_coef)-1) )
        return prob_trojan_in_model

    num_samples = len(acc_pruned_model_shift)
    # order:     Intercept    Smallest L1	Smaller L1	Middle L1	Large L1  Largeflagst L1
    prob_trojan_in_model = estimated_coef[0]
    for i in range(0,num_samples):
        prob_trojan_in_model += acc_pruned_model_shift[i] * estimated_coef[i+1]

    return prob_trojan_in_model

#####################################################################################################
# this method uses the linear regression coefficients estimated
# from the run on 1104 models in round2-train-dataset
def linear_regression_round2(model_name, acc_pruned_model_shift):
    prob_trojan_in_model = 0.0
    if len(acc_pruned_model_shift) != 5:
        print('ERROR: the number of accuracy samples per model is not equal 5, it is ', len(acc_pruned_model_shift))

    num_samples = len(acc_pruned_model_shift)
    # order:     Intercept    Smallest L1	Smaller L1	Middle L1	Large L1  Largeflagst L1
    if 'resnet18' in model_name:
        prob_trojan_in_model = resnet18_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * resnet18_coef[i+1]
    if 'resnet34' in model_name:
        prob_trojan_in_model = resnet34_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * resnet34_coef[i+1]
    if 'resnet50' in model_name:
        prob_trojan_in_model = resnet50_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * resnet50_coef[i+1]
    if 'resnet101' in model_name:
        prob_trojan_in_model = resnet101_coef[0]
        for i in range(0, 5):
            prob_trojan_in_model += acc_pruned_model_shift[i] * resnet101_coef[i + 1]
    if 'resnet152' in model_name:
        prob_trojan_in_model = resnet152_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * resnet152_coef[i+1]
    if 'wide_resnet50' in model_name:
        prob_trojan_in_model = wide_resnet50_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * wide_resnet50_coef[i+1]
    if 'wide_resnet101' in model_name:
        prob_trojan_in_model = wide_resnet101_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * wide_resnet101_coef[i+1]

    if 'densenet121' in model_name:
        prob_trojan_in_model = densenet121_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * densenet121_coef[i+1]
    if 'densenet161' in model_name:
        prob_trojan_in_model = densenet161_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * densenet161_coef[i+1]
    if 'densenet169' in model_name:
        prob_trojan_in_model = densenet169_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * densenet169_coef[i+1]
    if 'densenet201' in model_name:
        prob_trojan_in_model = densenet201_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * densenet201_coef[i+1]

    if 'inceptionv1' in model_name:
        prob_trojan_in_model = inceptionv1_googlenet_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * inceptionv1_googlenet_coef[i+1]
    if 'inceptionv3' in model_name:
        prob_trojan_in_model = inceptionv3_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * inceptionv3_coef[i+1]

    if 'squeezenetv1_0' in model_name:
        prob_trojan_in_model = squeezenetv1_0_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * squeezenetv1_0_coef[i+1]
    if 'squeezenetv1_1' in model_name:
        prob_trojan_in_model = squeezenetv1_1_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * squeezenetv1_1_coef[i+1]
    if 'mobilenetv2' in model_name:
        prob_trojan_in_model = mobilenetv2_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * mobilenetv2_coef[i+1]

    if 'shufflenet1_0' in model_name:
        prob_trojan_in_model = shufflenet1_0_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * shufflenet1_0_coef[i+1]
    if 'shufflenet1_5' in model_name:
        prob_trojan_in_model = shufflenet1_5_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * shufflenet1_5_coef[i+1]
    if 'shufflenet2_0' in model_name:
        prob_trojan_in_model = shufflenet2_0_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * shufflenet2_0_coef[i+1]

    if 'vgg11_bn' in model_name:
        prob_trojan_in_model = vgg11_bn_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * vgg11_bn_coef[i+1]
    if 'vgg13_bn' in model_name:
        prob_trojan_in_model = vgg13_bn_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * vgg13_bn_coef[i+1]
    if 'vgg16_bn' in model_name:
        prob_trojan_in_model = vgg16_bn_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * vgg16_bn_coef[i+1]

    return prob_trojan_in_model


# this method uses the linear regression coefficients estimated
# from the run on 1004 models in round2-train-dataset
def linear_regression_round1(model_name, acc_pruned_model_shift):
    prob_trojan_in_model = 0.0
    if len(acc_pruned_model_shift) != 5:
        print('ERROR: the number of accuracy samples per model is not equal 5, it is ', len(acc_pruned_model_shift))

    num_samples = len(acc_pruned_model_shift)
    # order:     Intercept    Smallest L1	Smaller L1	Middle L1	Large L1  Largest L1
    if 'resnet50' in model_name:
        prob_trojan_in_model = r1_resnet50_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * r1_resnet50_coef[i+1]

    if 'densenet121' in model_name:
        prob_trojan_in_model = r1_densenet121_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * r1_densenet121_coef[i+1]

    if 'inceptionv3' in model_name:
        prob_trojan_in_model = r1_inceptionv3_coef[0]
        for i in range(0,num_samples):
            prob_trojan_in_model += acc_pruned_model_shift[i] * r1_inceptionv3_coef[i+1]

    return prob_trojan_in_model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--linear_regression_filepath', type=str,
                        help='File path to the file where all estimated linera regression coefficients are stored for all architecture',
                        required=True)
    parser.add_argument('--architecture', type=str,
                        help='architecture of the AI model that was measured',
                        required=True)
    args = parser.parse_args()
    print('args %s \n %s \n' % (
        args.linear_regression_filepath, args.architecture))

    #argument for testing: --linear_regression_filepath .\round1_results\LR_results\run45_LR_results.csv --architecture densenet121
    trained_coef = read_regression_coefficients(args.linear_regression_filepath, args.architecture)
    # test failure
    measured_coef = [0.5,0.6,0.5,0.7,0.7,0.9,0.9]
    prob_trojan = linear_regression_prediction(trained_coef,measured_coef)
    print('TEST expected ERROR and prob = 0:',prob_trojan )
    # test success
    measured_coef = [0.5,0.6,0.5,0.7,0.7,0.9,0.9,0.9,0.9,0.9,0.9,0.5,0.5,0.5,0.5]
    prob_trojan = linear_regression_prediction(trained_coef,measured_coef)
    print('TEST expected SUCCESS and prob = 0.557:', prob_trojan)