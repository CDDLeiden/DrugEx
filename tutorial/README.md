# Welcome to the DrugEx Tutorial

This tutorial is a collection of Jupyter notebooks that show how various DrugEx models can be trained and optimized. The tutorial is divided by model type. If you are new to these types of models, we recommend you start with the recurrent neural network (RNN) in the first tutorial:

1. [Recurrent Neural Network](RNN.ipynb)
2. [Graph Transformer](Graph-Transformer.ipynb)
3. [SMILES Sequence Transformer](SMILES-Transformer.ipynb)

Before you begin, you should install the DrugEx package and its dependencies and also the environment for this tutorial:

```bash
pip install git+https://github.com/CDDLeiden/DrugEx.git@master
pip install -r requirements.txt
```

Subsequently, you can download the tutorial models and data with the download script accessible from command line:

```bash
drugex dowload
```