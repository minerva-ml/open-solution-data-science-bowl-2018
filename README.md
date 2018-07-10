# Data Science Bowl 2018: open solution

This is an open solution to the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) based on [winning solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) from topcoders.

## Goal
Implement winning solution described by topcoders and reproduce their results.

## Disclaimer
In this open source solution you will find references to the neptune.ml. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Results
`0.577` **Local CV**

`0.457` **Stage 1 LB**

# Solution write-up
## Preprocessing
* Overlay binary masks for each image is produced
* Borders are produced using dilated watershed lines
* Normalization as on ImageNet

## Augmentations
* Flips u/d and l/r
* Rotations with symmetric padding
* piecewise affine transformation
* perspective transform
* inverting colors
* contrast normalization
* elastic transformation
* adding random value to pixels (elementwise and uniformly, in RGB and HSV)
* multiplying pixels by random value (elementwise and uniformly, in RGB and HSV)
* channel shuffle
* Gaussian, average and median blurring
* sharpen, emboss


## Network
* Unet with pretrained Resnet101 or Resnet152 encoders
* First network with softmax activation function and 3 channels: [background, masks - borders, borders] for predicting borders
* Second network with sigmoid activation function and 2 channels: [masks, borders] for predicting full masks

## Training
* Adam optimizer
* Initial lr 1e-4
* Batch size of 36 (2 GPUs) or 72 (4 GPUs)
* Training on random crops of size 256x256
* Inference on full images padded to minimal size fitting to network (i.e. dimensions must be divisible by 64)
* TTA (flips, rotations)

## Loss function
* 1st network: Cross Entropy with Dice (not on background)
* 2nd network: BCE with Dice
* Averaging Dice Loss over number of classes didn't change the results

## Postprocessing
* Different thresholds are used for masks (2nd network) for retrieving seeds and final masks
* Seeds for watershed are calculated as masks (2nd network) - borders (1st network)
* Small mask instances and seeds are dropped
* Watershed using labeled seeds as markers and masks (2nd network) as masks

## External data
We included data from:
* https://nucleisegmentationbenchmark.weebly.com/dataset.html
* https://data.broadinstitute.org/bbbc/BBBC020/
* https://zenodo.org/record/1175282#.W0So1RgwhhG
* custom made images without nuclei on them
But, up to now, including external data did not improve our score


# Installation
Check [Installation page](https://github.com/neptune-ml/data-science-bowl-2018/wiki/Installation) on our Wiki, for instructions.

#### Fast track:
1. get repository, install [PyTorch](http://pytorch.org/) then remaining requirements
2. register to [Neptune](https://neptune.ml/ 'machine learning lab')
3. run experiment:
```bash
$ neptune login
$ neptune send main.py --worker gcp-gpu-large --environment pytorch-0.2.0-gpu-py3 -- train_evaluate_predict_pipeline --pipeline_name unet_multitask
```
4. collect submit from `/output/dsb/experiments/submission.csv` directory.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com) is our primary way of communication.
2. Read project's [Wiki](https://github.com/neptune-ml/data-science-bowl-2018/wiki), where we publish descriptions about the code, pipelines and neptune.
3. You can submit an [issue](https://github.com/neptune-ml/data-science-bowl-2018/issues) directly in this repo.

## Contributing
Check [CONTRIBUTING](CONTRIBUTING.md) for more information.