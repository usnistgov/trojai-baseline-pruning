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

import argparse
import csv
import time
import os

import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm

import numpy as np
import statistics



def load_csv_file(csv_filepath: str, header: bool, column_idx_array):
    # start timer
    start = time.time()
    header_cnt = ''
    # init 2D array
    column_cnt = []
    for i in range(len(column_idx_array)):
        column_cnt.append([])

    if os.path.isfile(csv_filepath):
        # read the label
        with open(csv_filepath) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            idx = 0
            for row in readCSV:
                # test
                if idx == 0:
                    idx3 = 0
                    for entry in row:
                        print('entry[', idx3, ']=', entry)
                        idx3 += 1

                if header and idx == 0:
                    header_cnt = row[0]
                else:
                    #print('number of col:',len(row))
                    idx2 = 0
                    for column_idx in column_idx_array:
                        if column_idx < len(row):
                            column_cnt[idx2].append(row[column_idx])
                        else:
                            print('WARNING: row idx:', idx, ' does not contain column idx: ', str(column_idx))
                        idx2 += 1
                idx += 1
    else:
        print('ERROR: file: ', csv_filepath, ' is not a file')

    end = time.time()
    print('loaded file:', csv_filepath, ' in ', (end-start), ' [s]')

    return header_cnt,column_cnt


def batch_process_dir(csv_dirpath, result_filepath, dataset_round, number_of_pruned_models):
    print('csv_dirpath = {}'.format(csv_dirpath))
    print('result_filepath = {}'.format(result_filepath))

    # Identify all csv files in the csv directory
    csv_file_names = os.listdir(csv_dirpath)
    print('os.listdir(csv_dirpath):',os.listdir(csv_dirpath))
    csv_format = '.csv'
    csv_filepath = []
    idx = 0
    for fn in csv_file_names:
        if fn.endswith(csv_format):
            csv_filepath.append(os.path.join(csv_dirpath, fn))
            idx = idx + 1

    number_of_csv_files = idx
    print('number_of_csv_files:', number_of_csv_files)

    ################## loop over all models ##############################
    # init indices of accuracy and GT model labels in the input CSV files, the last one is for execution time
    column_idx_array = []
    is_header_present = False
    if dataset_round >= 2:
        is_header_present = False
        if number_of_pruned_models == 55:
            # nS = 55
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                                67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 98]
        elif number_of_pruned_models == 45:
            # nS = 45
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
                                67, 68, 88]
        elif number_of_pruned_models == 35:
            # nS = 35
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 78]
        elif number_of_pruned_models == 25:
            # nS = 25
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                                45, 46, 47, 48, 68]
        elif number_of_pruned_models == 15:
            # nS = 15
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 58]
        elif number_of_pruned_models == 10:
            # nS = 10
            column_idx_array = [10, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 53]
        elif number_of_pruned_models == 5:
            # nS = 5
            column_idx_array = [10, 24, 25, 26, 27, 28, 48] # for any run larger than 13
            #column_idx_array = [5, 17,18,19,20,21, 41] # for run 4, 5, and 6
        else:
            print('ERROR: unsupported number of pruned accuracies:', number_of_pruned_models)
            exit(-1)

    if dataset_round == 1:
        if number_of_pruned_models == 35:
            # nS = 35
            # run49,50,51
            column_idx_array = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 74]
        elif number_of_pruned_models == 25:
            # nS = 25
            # run46,47,48
            column_idx_array = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                41, 42, 43, 44, 64]
        elif number_of_pruned_models == 15:
            # nS = 15
            # run41 -> up
            column_idx_array = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 54]
            # run23, 24, 25
            #column_idx_array = [5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 53]
        elif number_of_pruned_models == 10:
            # nS = 10
            # run 26, 27, 28
            column_idx_array = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 49]
        elif number_of_pruned_models == 5:
            # nS = 5
            column_idx_array = [10, 20, 21, 22, 23, 24, 44] # good for run 29 and up
            #column_idx_array = [5, 11, 12, 13, 14, 15, 39]  # good for run 11, 12
            #column_idx_array = [5, 11, 12, 13, 14, 15, 35]  # good for run 10
            #column_idx_array = [5, 11, 12, 13, 14, 15, 41]  # good for run 17
        else:
            print('ERROR: unsupported number of pruned accuracies:', number_of_pruned_models)
            exit(-1)

    # create header for the resulting file
    with open(result_filepath, 'w') as fh:
        fh.write("architecture_name,")
        fh.write("csv_filepath,")
        for idx3 in range(number_of_pruned_models+1):
            fh.write("b{}, ".format(idx3))

        fh.write("num_models, poisoned_model_count, num_erroneous_models,total_error_percent [%], fp_error_percent,"
                 "fn_error_percent,avg_exec_time,stdev_exec_time, avg_cross_entropy_loss, stdev_cross_entropy_loss \n")


    start = time.time()
    for idx in range(0, number_of_csv_files):
        #  extract NN architecture name from the file base name
        filepath_head, filepath_tail = os.path.split(csv_filepath[idx])
        end_point = len(filepath_tail)
        if filepath_tail.endswith('_log.csv'):
            end_point = len(filepath_tail)-len('_log.csv')
        else:
            if filepath_tail.endswith('_log1.csv'):
                end_point = len(filepath_tail) - len('_log1.csv')
            else:
                print('ERROR: expected suffix _log.csv or _log1.csv')

        architecture_name = filepath_tail[:end_point]
        print('processing model_filepath:', csv_filepath[idx], ' based name:', filepath_tail, ' architecture_name:', architecture_name)
        start1 = time.time()

        header_cnt, table_col = load_csv_file(csv_filepath[idx],  is_header_present, column_idx_array)

        # convert all entries to floats
        for col_idx in range(len(column_idx_array)):
            for row_idx in range(len( table_col[col_idx])):
                #if not type(table_col[col_idx][row_idx]) is float:
                try:
                    table_col[col_idx][row_idx] = float(table_col[col_idx][row_idx])
                except:
                    print('ERROR in CSV col:', col_idx, ' (column_idx_array[col_idx]:',column_idx_array[col_idx], ') row:', row_idx, ' value is not a float:', str(table_col[col_idx]) )
                    raise TypeError("Only floats are allowed")
                    # print('ERROR in CSV col:', col_idx, ' row:', row_idx, ' value is not a float:', table_col[col_idx][row_idx] )
                    # table_col[col_idx][row_idx] = -1.0
                    # print('ERROR: value set to -1.0')
                    # continue


        ###########################################
        # number of poisoned models
        poisoned_model_count = (statistics._sum(table_col[0]))[1] # statistics returns count for 0 (clean) and count fo 1 (poisoned)
        print('poisoned_model_count:', poisoned_model_count)
        #analyze the execution time
        idx_time = len(table_col) - 1
        print('idx_time:', idx_time)
        avg_exec_time = statistics.mean(table_col[idx_time])
        stdev_exec_time = statistics.stdev(table_col[idx_time])
        print('INFO: avg_exec_time:', avg_exec_time, ' stdev:', stdev_exec_time)
        ###############################################################
        # compute multiple regression coefficients
        # follow https://datatofish.com/multiple-linear-regression-python/

        my_input_var =[]
        acc_table = {}
        my_str = ''
        for index in range(number_of_pruned_models):
            my_str += 'a[' + str(index) + '],'
            my_input_var.append(('a[' + str(index) + ']'))
            acc_table["a[" + str(index) + "]"] = table_col[index + 1]

        acc_table["gt_model_label"] = table_col[0]
        #my_input_var = (my_str + '.')[:-2] # create a copy and remove the last comma
        my_str += 'gt_model_label'
        print('INFO: my_str:', my_str, ' my_input_var:', my_input_var)

        # acc_table = {
        #     'a[0]': table_col[1],
        #     'a[1]': table_col[2],
        #     'a[2]': table_col[3],
        #     'a[3]': table_col[4],
        #     'a[4]': table_col[5],
        #     'gt_model_label': table_col[0]
        #     }

        #df = pd.DataFrame(acc_table,columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]','gt_model_label'])
        #df = pd.DataFrame(acc_table, columns=[my_str]) # This did not work !!!
        # X = df[my_input_var] This assignment did not work!!!
        # If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
        if number_of_pruned_models == 55:
            df = pd.DataFrame(acc_table,
                              columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                                       'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]',
                                       'a[19]','a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]', 'a[25]', 'a[26]', 'a[27]',
                                       'a[28]','a[29]','a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]', 'a[35]', 'a[36]',
                                       'a[37]', 'a[38]','a[39]','a[40]', 'a[41]', 'a[42]', 'a[43]', 'a[44]', 'a[45]',
                                       'a[46]', 'a[47]', 'a[48]','a[49]',
                                       'a[50]', 'a[51]', 'a[52]', 'a[53]', 'a[54]','gt_model_label'])
            X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                    'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]', 'a[25]', 'a[26]', 'a[27]', 'a[28]', 'a[29]',
                    'a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]', 'a[35]', 'a[36]', 'a[37]', 'a[38]', 'a[39]',
                    'a[40]', 'a[41]', 'a[42]', 'a[43]', 'a[44]', 'a[45]', 'a[46]', 'a[47]', 'a[48]','a[49]',
                    'a[50]', 'a[51]', 'a[52]', 'a[53]', 'a[54]']]
        elif number_of_pruned_models == 45:
            df = pd.DataFrame(acc_table,columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]','a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]', 'a[25]', 'a[26]', 'a[27]', 'a[28]', 'a[29]',
                                                 'a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]', 'a[35]', 'a[36]', 'a[37]', 'a[38]', 'a[39]',
                                                 'a[40]', 'a[41]', 'a[42]', 'a[43]', 'a[44]','gt_model_label'])
            X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]','a[25]', 'a[26]', 'a[27]', 'a[28]', 'a[29]',
                                                 'a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]', 'a[35]', 'a[36]', 'a[37]', 'a[38]', 'a[39]',
                                                 'a[40]', 'a[41]', 'a[42]', 'a[43]', 'a[44]']]
        elif number_of_pruned_models == 35:
            df = pd.DataFrame(acc_table,columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]','a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]', 'a[25]', 'a[26]', 'a[27]', 'a[28]', 'a[29]',
                                                 'a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]','gt_model_label'])
            X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]','a[25]', 'a[26]', 'a[27]', 'a[28]', 'a[29]',
                                                 'a[30]', 'a[31]', 'a[32]', 'a[33]', 'a[34]']]
        elif number_of_pruned_models == 25:
            df = pd.DataFrame(acc_table,columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]','a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]', 'gt_model_label'])
            X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                    'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'a[15]', 'a[16]', 'a[17]', 'a[18]', 'a[19]',
                                                 'a[20]', 'a[21]', 'a[22]', 'a[23]', 'a[24]']]
        elif number_of_pruned_models == 15:
                df = pd.DataFrame(acc_table,
                                  columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]',
                                           'a[9]',
                                           'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]', 'gt_model_label'])
                X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                        'a[10]', 'a[11]', 'a[12]', 'a[13]', 'a[14]']]
        elif number_of_pruned_models == 10:
            df = pd.DataFrame(acc_table,
                              columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]',
                                       'gt_model_label'])
            X = df[['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]', 'a[5]', 'a[6]', 'a[7]', 'a[8]', 'a[9]']]
        elif number_of_pruned_models == 5:
            df = pd.DataFrame(acc_table,columns=['a[0]', 'a[1]', 'a[2]', 'a[3]', 'a[4]','gt_model_label'])
            X = df[['a[0]','a[1]', 'a[2]', 'a[3]', 'a[4]']]
        else:
            print('ERROR: unsupported number_of_pruned_models:', number_of_pruned_models )
            exit(-1)

        Y = df['gt_model_label']

        # with sklearn
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)

        print('Intercept: \n', regr.intercept_)
        print('Coefficients: \n', regr.coef_)

        # prediction with sklearn
        # a1 = 0.9
        # a2 = 0.8
        # print('Predicted model label: \n', regr.predict([[a1, a2]]))

        sum_fp = 0
        sum_fn = 0
        crossEntropyLossSum = 0.0
        crossEntropyLossSum2 = 0.0
        epsilon = 1.0e-12
        num_models = len(table_col[0])
        for model_idx in range(len(table_col[1])):
            gt_label = table_col[0][model_idx]
            # extract pruned model accuracies
            idx2 = 0
            a = []
            for acc_idx in range(1,len(column_idx_array)-1):
                a.append(table_col[acc_idx][model_idx])
                idx2 += 1
            print('model  idx:', model_idx, ' accuracy val:', a)
            predict_prob = 0.0
            if number_of_pruned_models == 5:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4]]])
            elif number_of_pruned_models == 10:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4],a[5], a[6], a[7], a[8], a[9]]])
            elif number_of_pruned_models == 15:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4],a[5], a[6], a[7], a[8], a[9],a[10],
                                            a[11], a[12], a[13], a[14] ]])
            elif number_of_pruned_models == 25:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4],a[5], a[6], a[7], a[8], a[9],a[10],
                                            a[11], a[12], a[13], a[14],a[15], a[16], a[17], a[18], a[19],a[20],
                                            a[21], a[22], a[23], a[24] ]])
            elif number_of_pruned_models == 35:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4],a[5], a[6], a[7], a[8], a[9],a[10],
                                            a[11], a[12], a[13], a[14],a[15], a[16], a[17], a[18], a[19],a[20],
                                            a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29],a[30],
                                            a[31], a[32], a[33], a[34] ]])
            elif number_of_pruned_models == 45:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10],
                                              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20],
                                              a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30],
                                              a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40],
                                              a[41], a[42], a[43], a[44]]])
            elif number_of_pruned_models == 55:
                predict_prob = regr.predict([[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10],
                                              a[11], a[12], a[13], a[14], a[15], a[16], a[17], a[18], a[19], a[20],
                                              a[21], a[22], a[23], a[24], a[25], a[26], a[27], a[28], a[29], a[30],
                                              a[31], a[32], a[33], a[34], a[35], a[36], a[37], a[38], a[39], a[40],
                                              a[41], a[42], a[43], a[44], a[45], a[46], a[47], a[48], a[49], a[50],
                                              a[51], a[52], a[53], a[54]]])
            else:
                print('ERROR: unsupported number_of_pruned_models:', number_of_pruned_models)
                continue


            print('Predicted model label: ',predict_prob, 'gt model label:', gt_label)

            # compute classification metric
            # cross entropy loss following https://pages.nist.gov/trojai/docs/overview.html?highlight=cross%20entropy
            #  CrossEntropyLoss=−(y∗log(p)+(1−y)∗log(1−p)), y = 0 or 1, p in [0,1]
            # based on wiki page: https://en.wikipedia.org/wiki/Cross_entropy, the log is actually ln
            # the cross entropy scores range from 0 (confident and correct) to about 36 (confident and wrong).
            if predict_prob < epsilon:
                predict_prob = epsilon # to avoid log(0)
            if (1.0 - predict_prob) < epsilon:
                predict_prob = 1.0 - epsilon # to avoid log(1-1)
            cross_entropy_loss = -(gt_label * np.log(predict_prob) + (1.0 - gt_label)*np.log(1.0 - predict_prob) )
            crossEntropyLossSum += cross_entropy_loss
            crossEntropyLossSum2 += cross_entropy_loss*cross_entropy_loss

            # classification errors
            residual = gt_label - predict_prob
            if residual <= -0.5:
                sum_fp += 1
            if residual >= 0.5:
                sum_fn += 1

        print('sum_fp:', sum_fp, ' sum_fn:', sum_fn)
        total_error_percent = round(100.0 * (sum_fp + sum_fn)/num_models,2)
        fp_error_percent = round(100.0 * sum_fp/num_models,2)
        fn_error_percent = round(100.0 * sum_fn/num_models, 2)

        print('sum_fp:', sum_fp, ' sum_fn:', sum_fn)
        avg_cross_entropy_loss = crossEntropyLossSum/num_models
        # print('DEBUG: crossEntropyLossSum2/num_models: ', crossEntropyLossSum2/num_models )
        # print('DEBUG: avg_cross_entropy_loss^2: ', avg_cross_entropy_loss*avg_cross_entropy_loss)
        val = crossEntropyLossSum2/num_models - avg_cross_entropy_loss*avg_cross_entropy_loss
        # sanity check since these values can have rounding errors: example 9.999557570501278e-25 - 9.999557570501283e-25 < 0
        if val < 0:
            stdev_cross_entropy_loss = 0.0
        else:
            stdev_cross_entropy_loss = np.sqrt(val)

        avg_cross_entropy_loss = int(10000*avg_cross_entropy_loss)/10000
        stdev_cross_entropy_loss = int(10000*stdev_cross_entropy_loss)/10000

        fp_error_percent = round(100.0 * sum_fp/num_models,2)
        fn_error_percent = round(100.0 * sum_fn/num_models, 2)

        end = time.time()
        # save the results
        with open(result_filepath, 'a') as fh:
            fh.write("{}, ".format(architecture_name))
            fh.write("{}, ".format(csv_filepath[idx]))
            fh.write("{}, ".format(regr.intercept_))
            for b_coeff in regr.coef_:
                fh.write("{}, ".format(b_coeff))

            fh.write("{}, ".format(num_models))
            fh.write("{}, ".format(poisoned_model_count))
            fh.write("{}, ".format(sum_fp+sum_fn))
            fh.write("{}, ".format(total_error_percent))
            fh.write("{}, ".format(fp_error_percent))
            fh.write("{}, ".format(fn_error_percent))
            fh.write("{}, ".format(avg_exec_time))
            fh.write("{}, ".format(stdev_exec_time))
            fh.write("{}, ".format(avg_cross_entropy_loss))
            fh.write("{}, ".format(stdev_cross_entropy_loss))

            # fh.write("csv_filename,{}, ".format(csv_filepath[idx]))
            # fh.write("header_cnt,{}, ".format(header_cnt))
            # fh.write("b0,{}, ".format(regr.intercept_))
            # idx3 = 1
            # for b_coeff in regr.coef_:
            #     fh.write("b{}, {}, ".format(idx3, b_coeff))
            #     idx3 += 1
            #
            # fh.write("num_models,{}, ".format(num_models))
            # fh.write("num_erroneous_model,{}, ".format(sum_fp+sum_fn))
            # fh.write("total_error_percent,{}, ".format(total_error_percent))
            # fh.write("fp_error_percent,{}, ".format(fp_error_percent))
            # fh.write("fn_error_percent,{}, ".format(fn_error_percent))
            # fh.write("avg_exec_time,{}, ".format(avg_exec_time))
            # fh.write("stdev_exec_time,{}, ".format(stdev_exec_time))
            fh.write("\n")


if __name__=='__main__':

    parser = argparse.ArgumentParser(
        description='Linear regression from accuracies of pruned models to NN clean or poisoned labels.')
    parser.add_argument('--accuracies_dirpath', type=str, help='Dir path to the csv files to be evaluated.',
                        required=True)
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written.',
                        required=True)


    args = parser.parse_args()
    print('args %s \n %s \n' % (
        args.accuracies_dirpath, args.result_filepath))
    # --accuracies_dirpath C:\PeterB\Projects\TrojAI\python\trojai-pruning\scratch_LR --result_filepath C:\PeterB\Projects\TrojAI\python\trojai-pruning\scratch_r2\LR_results.csv

    dataset_round = 4
    number_of_pruned_models = 45
    batch_process_dir(args.accuracies_dirpath, args.result_filepath, dataset_round, number_of_pruned_models)
