..  _usage:

Usage
=====

In this document, the use of command line interface (CLI) will be described. If you want more control over the inputs and outputs or want to customize DrugEx itself, you can also use the Python API directly (see :ref:`api-docs`). You can find a complete `tutorial <https://github.com/CDDLeiden/DrugEx/tree/master/tutorial>`_ illustrating some common use cases for each model type on the project's GitHub.

The command-line is a simple interface that can be used to preprocess data and build models quickly. In order to obtain a final model and generate novel compounds you will need to run multiple scripts, however.
The description of the functionality of each script can be displayed with the :code:`--help` argument. For example, the help message for the :code:`drugex.dataset` script can be shown as follows:

..  code-block:: bash

    python -m drugex.dataset --help

On Linux and MacOS, you do not need to call python explicitly and the following will suffice:

.. code-block:: bash

    drugex dataset --help

A basic command-line workflow to fine-tune and optimize a graph-based model is given below (see :ref:`cli-example`). 
However, we also show a few other workflows to show some of the other functionalities.

Before you start, make sure you have downloaded the example data and models in the :code:`tutorial/CLI/examples` folder:

..  code-block:: bash

    python -m drugex.download -o tutorial/CLI/examples # ran from the repository root

.. note:: All of the examples below also assume you are also executing them from the repository root.

.. warning:: All of the commands below are intended as quick examples and it is unlikely the resulting models will be useful in any way. In production settings, the models should of course be trained for many more epochs.

..  _cli-example:

CLI Example
===========

.. _basics:

Basics
------

Fine-tuning a Pretrained Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will use the DrugEx CLI to fine-tune a pretrained graph transformer (trained on the latest version of the Papyrus data set).
This pretrained model has been trained on a diverse set of molecules.
Fine-tuning will give us a model that can generate molecules that should more closely resemble the compounds in the data set of interest. 
You can find the model used here `archived on Zenodo <https://doi.org/10.5281/zenodo.7085421>`_ or among the other data files for this tutorial 'CLI/generators/'. 
You can find links to more pretrained models on the `project GitHub <https://github.com/CDDLeiden/DrugEx>`_.

Here, we want to bias the model towards generating compounds that are more related to known ligands of the Adenosine receptors. 
To use the CLI all the input data should be in the :code:`data` folder of the base directory :code:`-b tutorial/CLI`. 
For fine-tuning this input is a file with compounds (:code:`A2AR_LIGANDS.tsv`) 
Before we begin the fine-tuning, we have to preprocess the training data, as follows:

..  code-block:: bash

    # input is in tutorial/CLI/examples/data/A2AR_LIGANDS.tsv
    export BASE_DIR=tutorial/CLI/examples
    python -m drugex.dataset -b ${BASE_DIR} -i A2AR_LIGANDS.tsv -mc SMILES -o arl -mt graph

This will tell DrugEx to preprocess compounds saved in the :code:`-mc SMILES` column of the :code:`-i A2AR_LIGANDS.tsv` file for a :code:`-mt graph` type transformer

Preprocessing molecules for the graph based models includes fragmentation and encoding. This is done because the transformer takes fragmented molecules as input. 
For the graph-based transformers these inputs also need to be encoded into a graph representation.

The resulting files will be saved in the data folder and given a prefix (:code:`-o arl`). You can use this prefix to load the compiled data files in the next step. If you made and error somewhere or got an exception, you may also notice some :code:`backup_{number}` folders being created in the data folder. These are backups of the data files before the last step. You can use them to go back to the previous results if you accidentally overwrite them.

Now that we have our data sets prepared, we can finetune the pretrained generator on the preprocessed molecules with the :code:`train` script:

..  code-block:: bash

    python -m drugex.train -tm FT -b ${BASE_DIR} -i arl -o arl -ag ${BASE_DIR}/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg -mt graph -e 2 -bs 32 -gpu 0

This tells DrugEx to use the generated file (prefixed with :code:`-i arl`) to fine-tune (:code:`-m FT`) a pretrained model with model states saved in the :code:`-pt Papyrus05.5_graph_trans_PT.pkg` file.
The training will only be 2 epochs, :code:`-e 2`, with a batch size of 32, :code:`-bs 32` and it will be done on GPU 0, :code:`-gpu 0`. You can also specify multiple GPUs with the :code:`-gpu` argument (i.e :code:`-gpu 0,1`). The best model will be saved to :code:`${BASE_DIR}/generators/arl_graph_trans_FT.pkg`. However, you will find more output files with the :code:`.log` and :code:`.tsv` extensions in :code:`${BASE_DIR}`. These files contain the training and validation losses and the molecules generated at each epoch.


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

    python -m drugex.train -tm RL -b ${BASE_DIR} -i arl -o arl -ag arl_graph_trans_FT -pr ${BASE_DIR}/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg -p models/qsar/A2AR_RandomForestClassifier/A2AR_RandomForestClassifier_meta.json -ta A2AR_RandomForestClassifier -sas -e 2 -bs 32 -gpu 0

This tells DrugEx to create molecules from input fragments encoded in preprocessed data file (prefixed with :code:`arl`)
and optimize the initial agent (the fine-tuned model) (:code:`-ag arl_graph_trans_FT`) with RL (:code:`-m RL`). In this case we are using two desirability functions to score molecules:

* **Pretrained QSAR Model** (:code:`-p .../A2AR_RandomForestClassifier_meta.json`): This model is located in the :code:`tutorial/CLI/examples/models/qsar/` folder and is used to predict the bioactivity of the generated molecules on A2AR, which is indicated by adding it by name to the list of active targets with :code:`-ta A2AR_RandomForestClassifier`. This model was build using the :code:`QSPRpred` package and you can check out the Jupyter Notebook used to create it in the Python `tutorial <https://github.com/CDDLeiden/DrugEx/tree/master/tutorial/qsar.ipynb>`_

* **SAScore** (:code:`-sas`): This is a synthetic accessibility score that will prevent DrugEx from generating molecules that are too difficult to synthesize.

The rate between exploration and exploitation of known chemical space is forced by the use of a fixed prior-generator (:code:`-pr Papyrus05.5_graph_trans_PT`) and its influence can be tuned with the :code:`-eps, --epsilon` parameter.
The best model found during RL will be saved as :code:`${BASE_DIR}/generators/arl_graph_trans_RL.pkg`.

Design new molecules
^^^^^^^^^^^^^^^^^^^^

In this example, we use the optimized agent model to design new compounds that should be active on A2AR and have high synthetic accessibility.

.. code-block:: bash

    python -m drugex.generate -b ${BASE_DIR} -i arl_test_graph.txt -g arl_graph_trans_RL -gpu 0

This tells DrugEx to generate new molecules based on the input fragment in :code:`arl_test_graph.txt` with the :code:`arl_graph_trans_RL.pkg` model.
The new compounds are saved to :code:`${BASE_DIR}/new_molecules/arl_graph_trans_RL.tsv` and are also scored with the original environment used to create the model.


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

    python -m drugex.dataset -b ${BASE_DIR} -i A2AR_LIGANDS.tsv -mc SMILES -o rnn-example -nof -vf Papyrus05.5_smiles_voc.txt
    python -m drugex.train -tm FT -b ${BASE_DIR} -i rnn-example -ag ${BASE_DIR}/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT/Papyrus05.5_smiles_rnn_PT.pkg -vfs Papyrus05.5_smiles_voc.txt -mt smiles -a rnn -e 2 -bs 32 -gpu 0
    python -m drugex.generate -b ${BASE_DIR} -g rnn-example_smiles_rnn_FT -vfs Papyrus05.5_smiles_voc.txt -gpu 0 -n 30 --keep_undesired

Sequence-based transformer
""""""""""""""""""""""""""
For working with a SMILES-based transformer; you need to preprocess the data by specifying :code:`-mt smiles` indicating that the inputs are encoded as SMILES. 
By default the transformer algorithm (:code:`-a trans`) is used for training.


.. warning:: Note that the pretrained model for this model is not fetched by the tutorial utility at this point so you will have download its files separately. This model is also still more experimental and will likely not perform as well as the previous models.

..  code-block:: bash

    python -m drugex.dataset -b ${BASE_DIR} -i A2AR_LIGANDS.tsv -mc SMILES -o ast -mt smiles
    python -m drugex.train -tm FT -i ast -ag ${BASE_DIR}/models/pretrained/smiles-trans/Papyrus05.5_smiles_trans_PT/Papyrus05.5_smiles_trans_PT.pkg -mt smiles -a trans -e 2 -bs 32 -gpu 0


Pretraining a Generator
^^^^^^^^^^^^^^^^^^^^^^^

Pretraining :code:`-m PT` of a model from scratch works exactly the same way as finetuning,
the only difference is that the generator will not be initialized with pretrained model weights.

..  code-block:: bash

    python -m drugex.dataset -b ${BASE_DIR} -i A2AR_LIGANDS.tsv -mc SMILES -o example_pt -mt graph
    python -m drugex.train -tm PT -b ${BASE_DIR} -i example_pt -mt graph -e 2 -bs 32 -gpu 0

Scaffold-based Reinforcement learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tuning of the transformer-based generators can also be done on one scaffold or a subset of scaffolds. There are two ways to do it, either by using a subset of fragments-molecule pairs containing the selected scaffold or using the directly the scaffold as input. If your training data contains molecules with the selected scaffold we recommend former methods as its more stable with policy gradient-based reinforcement learning.

Here we show examples of these approaches on the previously trained and fine-tuned A2AR generators. We will use the molecule xanthine as a scaffold, in both examples.

With subset of molecules containing the scaffold
""""""""""""""""""""""""""""""""""""""""""""""""
First the molecules from the given dataset are fragmented and encoding while only selecting fragments-molecule pairs (:code:`-s <scaffold>`) containing the xanthine in the input fragements, then we proceed with RL with this subset of molecules.

.. code-block:: bash

    python -m drugex.dataset -b ${BASE_DIR} -i A2AR_LIGANDS.tsv -mc SMILES -o arl_xanthine -mt graph -sf c1[nH]c2c(n1)nc(nc2O)O 
    python -m drugex.train -tm RL -b ${BASE_DIR} -i arl_xanthine -o arl_xanthine -ag arl_graph_trans_FT -pr ${BASE_DIR}/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg -p models/qsar/A2AR_RandomForestClassifier/A2AR_RandomForestClassifier_meta.json -ta A2AR_RandomForestClassifier -sas -e 2 -bs 32 -gpu 0
    python -m drugex.generate -b ${BASE_DIR} -i arl_xanthine -g arl_xanthine_graph_trans_RL -gpu 0 -n 5

If you want the fragments-molecule pairs consist of ones with exclusively the selected scaffold as the input fragment add the argument :code:`-sfe` 

With input scaffold
"""""""""""""""""""
First this molecule is encoded, then reinforcement learning is done with this scaffold as input. Lastly a new molecule is generated containing this scaffold.

..  code-block:: bash

    # input is in tutorial/CLI/data/xanthine.tsv
    python -m drugex.dataset -b ${BASE_DIR} -i xanthine.tsv -mc SMILES -o scaffold_based -mt graph -s
    python -m drugex.train -tm RL -b ${BASE_DIR} -i scaffold_based_graph.txt -o scaffold_based -ag arl_graph_trans_FT -pr ${BASE_DIR}/models/pretrained/graph-trans/Papyrus05.5_graph_trans_PT/Papyrus05.5_graph_trans_PT.pkg -p models/qsar/A2AR_RandomForestClassifier/A2AR_RandomForestClassifier_meta.json -ta A2AR_RandomForestClassifier -sas -e 2 -bs 32 -gpu 0
    python -m drugex.generate -b ${BASE_DIR} -i scaffold_based_graph.txt -g scaffold_based_graph_trans_RL -gpu 0 -n 5

.. note:: The not fully converged model here will have trouble producing the scaffold that we need so the generate command may take a long time.
    
