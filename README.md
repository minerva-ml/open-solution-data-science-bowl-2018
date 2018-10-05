# Data Science Bowl 2018: open solution

This is an open solution to the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) based on the [topcoders winning solution](https://www.kaggle.com/c/data-science-bowl-2018/discussion/54741) from [ods.ai](http://ods.ai).

## More competitions :sparkler:
Check collection of [public projects :gift:](https://app.neptune.ml/-/explore), where you can find multiple Kaggle competitions with code, experiments and outputs.

## Disclaimer
In this open source solution you will find references to the [neptune.ml](https://neptune.ml). It is free platform for community Users, which we use daily to keep track of our experiments. Please note that using neptune.ml is not necessary to proceed with this solution. You may run it as plain Python script :wink:.

## Installation
Check [Installation page](https://github.com/neptune-ml/data-science-bowl-2018/wiki/Installation) on our Wiki, for detailed instructions.

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
