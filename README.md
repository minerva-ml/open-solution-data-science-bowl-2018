# Data Science Bowl 2018: mask-rcnn notebook

Idea behind this sub-project is to let you experiment with the state-of-the-art mask-rcnn in an easy-to-use environment.
The notebook `main.ipynb` should get you to around 0.44 on the LB.

## Installation

1. Register and login to https://neptune.ml/
2. install neptune-cli on your machine by running:
```bash
pip install neptune-cli
```
3. run
```bash
neptune login
```
4. upload Mask-RCNN project that is just a fork of https://github.com/matterport/Mask_RCNN by running
```bash
neptune data upload Mask-RCNN
```
5. Go to https://neptune.ml/ and click `start notebook` button at the top right
* select `python 3.5` 
* select `tensorflow 1.4.0`
* click browse files and upload the `main.ipynb` notebook
* type Mask_RCNN in the `files` tab

<img src="https://gist.githubusercontent.com/jakubczakon/10e5eb3d5024cc30cdb056d5acd3d92f/raw/577a2614cd1041f6251b6096029272f3547d78df/readme_neptune_rcnn.png" width="200" height="400" />



6. Click through the notebook and get 0.44 or tweatk parameters and get even more!

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com) is our primary way of communication.
2. Read project's [Wiki](https://github.com/neptune-ml/data-science-bowl-2018/wiki), where we publish descriptions about the code, pipelines and neptune.
3. You can submit an [issue](https://github.com/neptune-ml/data-science-bowl-2018/issues) directly in this repo.

## Contributing
Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
