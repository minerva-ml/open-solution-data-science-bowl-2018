# Intro
This is an open solution to the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) based on the [topcoders winning solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) from [ods.ai](http://ods.ai).

## Goal
Implement winning solution described by topcoders and reproduce their results using our Tech stack, mainly [steppy](https://github.com/neptune-ml/steppy) and [steppy-toolkit](https://github.com/neptune-ml/steppy-toolkit).

## Results (so far)
`0.577` **Local CV**

`0.457` **Stage 1 LB**

# Solution write-up
## Preprocessing
* Overlay binary masks for each image is produced
* Borders are produced using dilated watershed lines
* Normalization as on ImageNet

**Differences with topcoders solution:**
* Borders width doesn't depend on nuclei size

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

**Differences with topcoders solution:**
* No color to gray and gray to color
* We didn't know how often and how hard were these augmentations, if they were OneOf or SomeOf etc.

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

**Differences with topcoders solution:**
* No info about inference in the write up, maybe it was done using sliding window not on full images.
* Larger batchsize.

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

## Not implemented from topcoders solution
* 2nd level model
* model ensembling
