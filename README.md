# Data Science Bowl 2018: open solution

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/neptune-ml/data-science-bowl-2018/blob/master/LICENSE)

This is an open solution to the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018).

[Live preview of our experiments](https://app.neptune.ml/neptune-ml/Data-Science-Bowl-2018) with all code :computer:, parameters :bookmark_tabs: and results :chart_with_upwards_trend:.

## Goals
1) Deliver open, ready-to-use and extendable solution to this competition. This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.
2) Encourage more Kagglers to start working on the Data Science Bowl, test their ideas and learn advanced data science.

## Disclaimer
In this open source solution you will find references to the neptune.ml. It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Installation
Check [Installation page](https://github.com/neptune-ml/data-science-bowl-2018/wiki/Installation) on our Wiki, for instructions.

#### Fast track:
1. get repository, install [PyTorch](http://pytorch.org/) then remaining requirements
2. register to [neptune.ml](https://neptune.ml)
3. run experiment:
```bash
$ neptune account login
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
