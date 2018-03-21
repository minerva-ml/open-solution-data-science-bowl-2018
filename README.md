# Data Science Bowl 2018: mask-rcnn notebook

Idea behind this sub-project is to let you experiment with the state-of-the-art mask-rcnn in an easy-to-use environment.
The notebook `main.ipynb` should get you to around **0.44** on the LB.

## Installation

1. Register and login to https://neptune.ml/ 
2. Create a project called Data Science Bowl so that you get project-key DSB
3. install neptune-cli on your machine by running:
```bash
pip install neptune-cli
```
4. run
```bash
neptune login
```
5. upload Mask-RCNN project that is just a fork of https://github.com/matterport/Mask_RCNN by running
```bash
neptune data upload Mask_RCNN --recursive --project-key DSB
```
6. Go to https://neptune.ml/ and click `start notebook` button at the top right
* select `gcp-gpu-medium`
* select `python 3.5` 
* select `tensorflow 1.4`
* click browse files and upload the `main.ipynb` notebook
* type `Mask_RCNN` in the table at the bottom (_files_ tab)

7. Click through the notebook and get **0.44** submission from `/output/submission.csv` or tweak parameters and get even more!
<img src="https://gist.githubusercontent.com/jakubczakon/10e5eb3d5024cc30cdb056d5acd3d92f/raw/577a2614cd1041f6251b6096029272f3547d78df/readme_neptune_rcnn.png" width="600" height="600" />

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com) is our primary way of communication.
2. Read project's [Wiki](https://github.com/neptune-ml/data-science-bowl-2018/wiki), where we publish descriptions about the code, pipelines and neptune.
3. You can submit an [issue](https://github.com/neptune-ml/data-science-bowl-2018/issues) directly in this repo.

## Contributing
Check [CONTRIBUTING](CONTRIBUTING.md) for more information.
