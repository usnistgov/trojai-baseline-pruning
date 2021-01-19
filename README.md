# Pruning-based Baseline Approach to Trojan Detection in AI Models

## Goal
This code is for classifying convolutional neural network (CNN) models into those trained with Trojans (TwT) and those trained without Trojans (TwoT).
It is based on Round 1 and Round 2 TrojAI Challenge datasets posted at [URL](https://pages.nist.gov/trojai/docs/data.html#).

### Inputs
Trained CNN models with metadata about presence or absence of Trojans (clean or poisoned model).

Images without Trojans that encode a predicted classification label in file names.
 
### Outputs
Probability of the input CNN model being trained with Trojan (being poisoned)

## Installation

Follow the installation of conda at [URL](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

```sh
conda create -n env_2 python=3.4
```

```sh
conda env list
```

```sh
conda activate env_2
```

```sh
pip install -r requirements.txt
```

Note: The file 'requirements.txt' defines the minimum installation requirements. The file was
created on Windows OS, but the requirements extracted on Linux Ubuntu OS 
have additional OS specific libraries. 

## Execution
The execution consists of three steps:

1. extract accuracy measurements over all pruned model version
from all available models by running 
trojan_detector.py (Round 1) or trojan_detector_round2.py (Round 2)
    * The six execution parameters are set in these main classes
    * This step includes estimating the CNN model architecture using 
    model_classifier.py (Round 1) or model_classifier_round2.py (Round2) against
    the reference architectures file sizes and graphs provided in reference_data folder
2. estimate multiple linear regression coefficients from the extracted accuracy
measurements and ground truth labels provided for Round 1 and 2 data sets by 
running linear_regression_fit.py
    * Example files are provided in linear_regression_data folder
3. compute probability of a CNN model being trained with trojan by running 
trojan_detector.py (Round 1) or trojan_detector_round2.py (Round 2) with 
the estimated linear regression coefficients handled in linear_regression.py
    * The final probability value is saved in a file defined by --result_filepath argument

The main classes for trojan detection are in trojan_detector.py (Round 1) and 
trojan_detector_round2.py (Round 2), which follow the arguments required 
by the TrojAI challenge. 
They are designed to take one trained AI model and output the probability of the CNN model being trained with trojan.

**Example:**

```sh
python3 ./trojan_detector_round2.py --model_filepath ./trojai/datasets/round2/id-00000001/model.pt  --result_filepath ./trojai/datasets/round2/scratch_r2/test_python_output.txt --scratch_dirpath .trojai/datasets/round2/scratch_r2 --examples_dirpath ./trojai/datasets/round2/id-00000001/example_data
```

One can execute the trojan detection on a folder containing many AI models with example images
 by using the classes my_folder.py (Round 1) and my_folder_round2.py (Round 2). 
 The execution assumes certain organization of files in folders.
 
**Example:** 

```sh
python3 ./my_folders_round2.py --model_dirpath ./trojai/datasets/round2/round2-train-dataset/  --result_filepath ./trojai/datasets/round2/scratch_r2/output.txt --scratch_dirpath ./trojai/datasets/round2/scratch_r2/ 
```

Parts of the code can be reused for classifying AI models according to their architecture 
using the model file size and graphs.

**Example:**

```sh
python3 ./model_classifier_round2.py --model_dirpath ./trojai/datasets/round2/round2-train-dataset/  --result_filepath ./trojai/datasets/round2/scratch_r2/model_names.txt --model_format .pt
```

```sh
python3 ./graph_classifier_round2.py --model_dirpath ./trojai/datasets/round2/round2-train-dataset/  --reference_dirpath /trojai-pruning/reference_data --result_filepath ./trojai/datasets/round2/scratch_r2/graph_output.txt --scratch_dirpath ./trojai/datasets/round2/scratch_r2/ --model_format .pt
```

## Algorithmic Parameters for the Pruning-based Approach

The code supports multiple configurations of pruning:
- Pruning Method: Remove, Reset, and Trim 
- Sampling Method: Random, Uniform, and Targeted
- Ranking Method: L1, L2, L_infinity, and Stdev
- Number of images used: nD (used for evaluating accuracy of pruned models)
- Number of samples: nS (i.e., pruned models to be evaluated)
- Sampling proportion: p (proportion of filters to be removed in each layer)

**Note 1:** The current implementation removes conv2D and batch normalization modules 
in each layer. The configurations are set in the main trojan detection classes.

**Note 2:** The execution does not require GPU.

**Note 3:** ShuffleNet architectures in the TrojAI Round 2 dataset are not supported 
for the configuration with the pruning method equal to Remove. The explanation is available 
at the [URL](https://github.com/VainF/Torch-Pruning/issues/9) that was the source 
for the graph dependency implementation. 

**Note 4:** The configuration with the pruning method equal to Remove is also sensitive
to the parameter p (sampling proportion) for DenseNet architectures. For p > 0.02, some 
DenseNet models are pruned incorrectly since the forward pass (accuracy evaluation) fails.

**Note 5:** The torch.clamp method (see [URL](https://pytorch.org/docs/stable/generated/torch.clamp.html))
is suspected to have a memory leak. The code for capturing the RAM before and after is 
commented out in the class trim_prune.py. The memory leak is a problem when running
the code on 1000+ models, and therefore it is recommended to use the shell scripts
(evaluate_models.sh and evaluate_models_round2.sh) when processing thousands of models.


