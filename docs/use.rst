..  _usage:

Usage
=====

The command-line interface can be used to preprocess data and build models. In order to obtain a final model and generate novel compounds you will need to run multiple scripts. 
The description of the functionality of each script can be displayed with the :code:`--help` argument. For example, the help message for the :code:`drugex.dataset` script can be shown as follows:

..  code-block::

    python -m drugex.dataset --help

A basic command-line workflow to fine-tune and optimize a graph-based model is given below (see :ref:`cli-example`). 
In addition to this we also show a few other workflows to show some of the other functionalities.

The command downloads the data and models required for running the CLI examples and saves them in the tutorial/CLI folder.

..  code-block:: bash
    python -m drugex.download -o tutorial/CLI

If you want more control over the inputs and outputs or want to customize DrugEx a bit more, you can also use the Python API directly (see :ref:`api-docs`). 
You can find a tutorial with Jupyter notebooks illustrating some common use cases in the project `source code <https://github.com/CDDLeiden/DrugEx/tree/master/tutorial>`_.

..  _cli-example:

CLI Example
===========

.. _basics:

Basics
------

Fine-tuning a Pretrained Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will use DrugEx' CLI to fine-tune a pretrained graph transformer (trained on the Papyrus data set `05.5`). 
This pretrained model has been trained on a diverse set of molecules.
Fine-tuning will give us a model that can generate molecules that should more closely resemble the compounds in the data set of interest. 
You can find the model used here `archived on Zenodo <https://doi.org/10.5281/zenodo.7085421>`_ or among the other data files for this tutorial 'CLI/generators/'. 
You can find links to more pretrained models on the `project GitHub <https://github.com/CDDLeiden/DrugEx>`_.

Here, we want to bias the model towards generating compounds that are more related to known ligands of the Adenosine receptors. 
To use the CLI all the input data should be in the :code:`data` folder of the base directory :code:`-b tutorial/CLI`. 
For fine-tuning this input is a file with compounds (:code:`A2AR_LIGANDS.tsv`) 
Before we begin the fine-tuning, we have to preprocess the training data, as follows:

..  code-block:: bash

    # input is in tutorial/CLI/data/A2AR_LIGANDS.tsv
    python -m drugex.dataset -b tutorial/CLI -i A2AR_LIGANDS.tsv -mc SMILES -o arl -mt graph

This will tell DrugEx to preprocess compounds saved in the :code:`-mc SMILES` column of the :code:`-i A2AR_LIGANDS.tsv` file for a :code:`-mt graph` type transformer

Preprocessing molecules for the graph based models includes fragmentation and encoding. This is done because the transformer takes fragmented molecules as input. 
For the graph-based transformers these inputs also need to be encoded into a graph representation.

The resulting files will be saved in the data folder and given a prefix (:code:`-o arl`). 
You can use this prefix to load the compiled data files in the next step, which is fine-tuning the pretrained generator on the preprocessed molecules with the :code:`train` script:

..  code-block:: bash

    # pretrained model placed in tutorial/CLI/generators/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg
    python -m drugex.train -m FT -b tutorial/CLI -i arl -o arl -pt pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5 -mt graph -e 1 -bs 32 -gpu 0,1

This tells DrugEx to use the generated file (prefixed with :code:`-i arl`) to fine-tune (:code:`-m FT`) a pretrained model with model states saved in the :code:`-pt Papyrus05.5_graph_trans_PT.pkg` file.
The training will take 1 epoch :code:`-e 1` (for a real application more epochs are required) with a batch size of 32 in parallel on GPUs with IDs 0 and 1. 
The best model will be saved to :code:`./generators/arl_graph_trans_FT.pkg`.


Optimization with Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, want to generate drug-like molecules that are active on A2AR and have a high Syntehtic Accessibility Score (SAScore).
To achieve this, reinforcement learning (RL) is used to tune the generator model to generate molecules with desired properties. 
For this task the RL framework is composed of the agent (generator) and environment (predictor and SAScorer).
The predictor model (a Random Forest QSAR model for binary A2A bioactivity predictions) has been `created using QSPRpred <https://github.com/CDDLeiden/QSPRPred>`_

During RL a combination of two generators with the same architecture is used to create molecules; the agent that is optimized during RL for exploitation and 
the prior that is kept fixed for exploration. 
At each iteration, generated molecules are scored based on the environment and send a back to the agent for tuning.

.. code-block:: bash

    # pretrained model placed in tutorial/CLI/generators/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg
    # predictor model placed in tutorial/CLI/qspr/models
    python -m drugex.train -m RL -b tutorial/CLI -i arl -o arl -ag arl_graph_trans_FT -pr pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT -ea RF -ta pchembl_value_Median -sas -e 3 -bs 32 -gpu 0,1

This tells DrugEx to create molecules from input fragments encoded in preprocessed data file (prefixed with :code:`arl`)
and optimize the initial agent (the fine-tuned model) (:code:`-ag arl_graph_trans_FT`) with RL (:code:`-m RL`). 
Molecules are scored with a desirability function that favour molecules predicted to be active on A2AR (:code:`-ta pchembl_value_Median`) as predicted using a RF model (:code:`-ea RF`)
and have a high synthetic accessibility (:code:`-sas`).
Exploration of chemical space is forced by the use of a fixed prior-generator (:code:`-pr Papyrus05.5_graph_trans_PT`). 
The training will take a maximum of 3 epochs with a batch size of 32 in parallel on GPUs with IDs 0 and 1. 
The best model will be saved to :code:`./generators/arl_graph_trans_RL.pkg`.

Design new molecules
^^^^^^^^^^^^^^^^^^^^

In this example, we use the optimized agent model to design new compounds that should be active on A2AR and have high synthetic accessibility.

.. code-block:: bash

    python -m drugex.designer -b tutorial/CLI -i arl_test_graph.txt -g arl_graph_trans_RL

This tells DrugEx to generate new molecules based on the input fragment in :code:`arl_test_graph.txt` with the :code:`arl_graph_trans_RL.pkg` model.
The new compounds are saved to :code:`./new_molecules/arl_graph_trans_RL.tsv`.


Advanced
--------

Using different generator architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can vary the type of model to use with the :code:`-a` and :code:`-mt` parameters. 

Recurrent neural network
""""""""""""""""""""""""
The most simple model is the RNN-based generator. This model gets the 'go' token as input and from there generates SMILES strings. 
Therefore, this model does not use input fragments for training or sampling. To preprocess the data for training an RNN-based generator the molecules 
are standardized and encoded based on the vocabulary of the pretrained model :code:`-vf Papyrus05.5_smiles_voc.txt`, but no fragmentation is done :code:`-nof`. 
To fine-tune an RNN-based generator on the A2AR set, the algorithm needs to be specified :code:`-a rnn`.
Here the generator is fine-tuned on the A2AR set and then used to generate new compounds. 

..  code-block:: bash

    # pretrained model placed in tutorial/CLI/generators/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.pkg
    # pretrained model voc files placed in tutorial/CLI/data/Papyrus05.5_smiles_voc.txt
    python -m drugex.dataset -b tutorial/CLI -i A2AR_LIGANDS.tsv -mc SMILES -o rnn-example -nof -vf Papyrus05.5_smiles_voc.txt
    python -m drugex.train -m FT -b tutorial/CLI -i rnn-example -pt pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT -vfs Papyrus05.5  -mt smiles -a rnn -e 3 -bs 32 -gpu 0
    python -m drugex.designer -b tutorial/CLI -g rnn-example_smiles_rnn_FT -vfs Papyrus05.5 -gpu 0

Sequence-based transformer
""""""""""""""""""""""""""
For working with a SMILES-based transformer; you need to preprocess the data by specifying :code:`-mt smiles` indicating that the inputs are encoded as SMILES. 
By default the transformer algorithm (:code:`-a trans`) is used for training.

..  code-block:: bash

    # pretrained model placed in tutorial/CLI/generators/pretrained/smiles-trans/Papyrus05.5_smiles_trans_PT/Papyrus05.5_smiles_trans_PT.pkg
    python -m drugex.dataset -b tutorial/CLI -i A2AR_LIGANDS.tsv -mc SMILES -o ast -mt smiles
    python -m drugex.train -m FT -i ast -pt pretrained/smiles-trans/Papyrus05.5_smiles_trans_PT/Papyrus05.5_smiles_trans_PT -mt smiles -a trans -e 3 -bs 32 -gpu 0,1


Pretraining a Generator
^^^^^^^^^^^^^^^^^^^^^^^

 Pretraining :code:`-m PT` of a model from scartch works exactly the same way as fine-tuning, 
 the only difference is that the generator will not be initialized with pretrained model weights.

 ..  code-block:: bash

    python -m drugex.dataset -b tutorial/CLI -i A2AR_LIGANDS.tsv -mc SMILES -o example_pt -mt graph
    python -m drugex.train -m PT -b tutorial/CLI -i example_pt -mt graph -e 3 -bs 32 -gpu 0,1


Scaffold-based Reinforcement learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tuning of the transformer-based generators can also be done on one scaffold or a subset of scaffolds. Here we show an example of this on the previously trained and fine-tuned A2AR generators.
First
.. code-block:: bash
    # input is in tutorial/CLI/data/xanthine.tsv
    python -m drugex.dataset -b tutorial/CLI -i xanthine.tsv -mc SMILES -o scaffold_based -mt graph -s
    python -m drugex.train -m RL -b tutorial/CLI -i scaffold_based_graph.txt -o scaffold_based -ag arl_graph_trans_FT -pr pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT -ta pchembl_value_Median -sas -e 3 -bs 32 -gpu 0,1
    python -m drugex.designer -b tutorial/CLI -i scaffold_based_graph.txt -g scaffold_based_graph_trans_RL
    