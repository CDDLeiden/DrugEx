# Welcome to the DrugEx Tutorial

This tutorial is a collection of Jupyter notebooks that show how various DrugEx models can be trained and optimized for different tasks. It is divided into several parts that reflect the most common workflows in DrugEx:

1. [Data Preparation](datasets.ipynb)
2. [Pretraining](pretraining.ipynb) -- this tutorial is optional since very rarely you will have to pretrain your own models
3. [Finetuning](finetuning.ipynb)
4. [Optimization with Reinforcement Learning](rl_optimization.ipynb)
5. [Generating Molecules](generation.ipynb)

You can download example data sets from [this link](https://drive.google.com/file/d/1_t4Br4iGGj953qyYvoLmwIWXONqSXYNJ/view?usp=share_link). Just unpack the file in this folder and you should be good to go.

The type of model you are trying to build determines what parts of the API you will need in your workflow. Therefore, make sure you have an overview of the available models and understand their specifics (details can be found in the [main readme file](../README.md) or the project [documentation](https://cddleiden.github.io/DrugEx/docs/)).
