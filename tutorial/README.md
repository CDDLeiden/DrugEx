# Welcome to the DrugEx Tutorial

This tutorial is a collection of Jupyter notebooks that show how various DrugEx models can be trained and optimized. The tutorial is divided by model type. If you are new to these types of models, we recommend you start with the recurrent neural network (RNN) in the first tutorial:

1. [Recurrent Neural Network](Sequence-RNN.ipynb) 
   - start with this tutorial if you also need broader introduction to generative AI for molecules)
2. [Graph Transformer](Graph-Transformer.ipynb)
   - this is a more advanced tutorial and knowledge of some terms from the previous one is assumed
<!-- 3. [SMILES Sequence Transformer](SMILES-Transformer.ipynb) -->

## Installation

Before you begin, you should install the DrugEx package and the dependencies for this tutorial. You will find everything you need in the `requirements.txt` file and can use pip to install everything:

```bash
pip install -r requirements.txt
```

Subsequently, you should also download the tutorial models and data with the download script accessible from command line:

```bash
drugex download
```

### Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/CDDLeiden/DrugEx/)

If you are using Google Colab, just click the logo above and select one of the shown notebooks. Then make sure you are using a runtime with GPU access and evaluate the following cell in the opened notebook:

```python
!wget https://raw.githubusercontent.com/CDDLeiden/DrugEx/master/tutorial/colab.sh
!bash colab.sh
```

 Then you should be able to import the drugex package:

```python
import drugex
drugex.__version__
```
