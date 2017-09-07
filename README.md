# *FingerNet* : An Universal Deep Convolutional Network for Extracting Fingerprint Representation

By Yao Tang, Fei Gao, JuFu Feng and YuHang Liu at Peking University

### Introduction

**FingerNet** is an universal deep ConvNet for extracting fingerprint representations including orientation field, segmentation, enhenced fingerprint and minutiae. It can produce reliable results on both rolled/slap and latent fingerprints.

Here is a Python implementation of FingerNet. This code has been tested on Ubuntu 14.04 and Python2.7.

### License

FingerNet is released under the MIT License (refer to the LICENSE file for details).

### Citing FingerNet

If you find FingerNet useful in your research, please consider citing:

    @inproceedings{tang2017FingerNet,
        Author = {Tang, Yao and Gao, Fei and Feng, Jufu and Liu, Yuhang},
        Title = {FingerNet: An Unified Deep Network for Fingerprint Minutiae Extraction},
        booktitle = {Biometrics (IJCB), 2017 IEEE International Joint Conference on},
        Year = {2017}
        organization={IEEE}
    }

### Main Results
| training data | test data | precision| recall | conv-time/img| post-time/img|
|:-------------:|:---------:|:--------:|:------:|:------------:|:------------:|
|  CISL24218    |  FVC2004  |   76%    |  80%   |  674ms       |   285ms      |
|  CISL24218    |  NISTSD27 |   63%    |  63%   |  183ms       |   885ms      |

**Note**: 
0. conv-time/img can be faster if using batch size greater than 1.
0. post-time/img contains orientation selection, segmentation dilation and minutiae nms.
0. CISL24218 is a in-house database which is unavailable to public. It contains around 8000 matched roll and latent fingerprints with manual minutiae and around 10000 latent fingerprints with only manual minutiae. Both FVC2002DB2A and NISTSD27 are non-intersect with CISL24218. 

### Contents
0. [Requirements: software](#requirements-software)
0. [Requirements: hardware](#requirements-hardware)
0. [Predicting Demo](#predicting-demo)
0. [Preparation for Training & Testing](#preparation-for-training-and-testing)
0. [Training](#training)
0. [Testing](#testing)
0. [Acknowledgement](#acknowledgement)


### Requirements: software

0. `Python 2.7`: cv2, numpy, scipy, matplotlib, pydot, graphviz
0. `Tensorflow 1.0.1`
0.  `Keras 2.0.2`

### Requirements: hardware

GPU: Titan, Titan Black, Titan X, K20, K40, K80.

0. FingerNet predicting
    - 2GB GPU memory for FVC2002DB2A
    - 5GB GPU memory for NISTSD27

### Predicting Demo

0.  Run `cd` in shell to directory `src/`
0.  Run `python train_test_deploy.py 0 deploy` to test demo images provided in `datasets/`.
    - You may use different GPU by changing 0 to desired GPU ID. 
    - You will see the timing information as below. We get the following running time on single-core of K80 the demo images:
    ```Shell
    Predicting images:
    images 1 / 1: B101L9U
    load+conv: 4.872s, seg-postpro+nms: 0.223, draw: 2.249
    Average: load+conv: 4.872s, oir-select+seg-post+nms: 0.223, draw: 2.249
    Predicting CISL24218:
    CISL24218 1 / 1: A0100003009991600022036_2
    load+conv: 2.219s, seg-postpro+nms: 0.439, draw: 2.247
    Average: load+conv: 2.219s, oir-select+seg-post+nms: 0.439, draw: 2.247
    Predicting FVC2002DB2A:
    FVC2002DB2A 1 / 1: 1_1
    load+conv: 1.718s, seg-postpro+nms: 0.548, draw: 3.309
    Average: load+conv: 1.718s, oir-select+seg-post+nms: 0.548, draw: 3.309
    Predicting NIST4:
    NIST4 1 / 1: F0001_01
    load+conv: 1.006s, seg-postpro+nms: 1.640, draw: 3.947
    Average: load+conv: 1.006s, oir-select+seg-post+nms: 1.640, draw: 3.947
    Predicting NIST14:
    NIST14 1 / 1: F0000001
    load+conv: 6.211s, seg-postpro+nms: 3.271, draw: 4.643
    Average: load+conv: 6.211s, oir-select+seg-post+nms: 3.271, draw: 4.643
    ```
    - The visual results might be different from those in the paper due to numerical variations.    
0. Change `deploy_set=['*/',..., '*/']` in line 44-45 in `train_test_deploy.py` to desired dataset folders to test other fingerprint datasets.

### Preparation for Training and Testing

0.  Move raw fingerprint training images to `datasets/dataset-name/images/`
    - Training images should be of `bmp` format
0.  Move segmentation labels to `datasets/dataset-name/seg_labels/`
    - Segmentation labels should be of `png` format and have the same size and name with its corresponding fingerprint images.
    - `255` indicates foreground while `0` for background.
0. Move orientation labels to `datasets/dataset-name/ori_labels/`
    - Orientation labels are rolled/slap fingerprint aligned to training images and of same size, same name and `bmp` format.
0. Move minutiae labels to `datasets/dataset-name/mnt_labels/`
    - Minutiae labels should be of same name and `mnt` format.
    - `.mnt` structure is as follow:
        - line 1: image-name
        - line 2: number-of-minutiae-N, image-W, image-H
        - next N lines: minutia-x, minutia-y, minutiae-o 
0. Change `train_set=['*/', ...'*/']` in line 42 in `train_test_deploy.py` to `train_set=['../datasets/dataset-name/',]`

### Training:

0. Run `python train_test_deploy.py 0 train` to finetune your model. 
    - **Note**: Maximum epoch is set to 100. Early stop if model have converged.

### Testing

0. Run `python train_test_deploy.py 0 test` to test your model.
    - Different from Predicting, Testing requires datasets to have at least mnt labels and segmentation labels. 
    - Change `test_set=['*/', ...'*/']` in line 44 in `train_test_deploy.py` to test other datasets.
