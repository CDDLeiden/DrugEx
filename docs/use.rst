..  _usage:

Usage
=====

You can use the command-line interface to preprocess data and build models. You will need to run multiple scripts to be able to obtain a final model. 
The description of the functionality of each script can be displayed with the :code:`--help` argument. For example, the help message for the :code:`drugex.dataset` script can be shown as follows:

..  code-block::

    python -m drugex.dataset --help

A simple command-line workflow to fine-tune and optimize a graph-based model is given below (see :ref:`cli-example`). 

If you want more control over the inputs and outputs or want to customize DrugEx a bit more, you can also use the Python API directly (see :ref:`api-docs`). 
You can find a tutorial with Jupyter notebooks illustrating some common use cases in the project `source code <https://github.com/CDDLeiden/DrugEx/tree/master/tutorial>`_.

..  _cli-example:

CLI Example
===========

.. _basics:

Basics
------

Finetuning a Pretrained Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will finetune an already pretrained graph transformer, but before we do that we have to generate the appropriate training data. 
Let's assume we want to bias the model towards generating compounds that are more related to known ligands of the Adenosine receptors. 
DrugEx assumes that all input data are saved in the :code:`data` folder of the directory it is executed from. 
Therefore, we place the compounds that will serve as a template for the finetuning inside this folder and execute DrugEx as follows:

..  code-block:: bash

    # input is in ./data/AR_ligands.tsv
    drugex dataset -i AR_ligands.tsv -mc CANONICAL_SMILES -o arl -mt graph

This will tell DrugEx to preprocess compounds saved in the :code:`CANONICAL_SMILES` column of the :code:`AR_ligands.tsv` file 
(the file can be found in :code:`tutorial/jupyter/data`).
The resulting input files will be saved in the data folder and given a prefix (:code:`arl`). 
You can then finetune the pretrained molecule generator preprocessed molecules with the :code:`train` script:

..  code-block:: bash

    # pretrained model placed in ./generators/chembl27_graph.pkg
    drugex train -m FT -i arl -o arl -pt chembl27_graph -mt graph -e 200 -bs 32 -gpu 0,1

This tells DrugEx to use the generated file (prefixed with :code:`arl`) to finetune (:code:`-m FT`) a pretrained model with states saved in the :code:`chembl27_graph.pkg` file. 
The training will take a maximum of 200 epochs with a batch size of 32 in parallel on GPUs with IDs 0 and 1. 
The best model will be saved to :code:`./generators/arl_graph_trans_FT.pkg`.

Optimization with Reinforcement Learning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we generate drug-like molecules that should be active on A2B (UniprotID: P29275) and inactive on A2A (UniprotID: P29274).
The reinforcement learning (RL) framework is used to create the exploitation-exploration model tuned to generate molecules with desired properties. 
The RL framework is composed of the agent-generator and environment-predictor parts.

QSAR models
"""""""""""

First, we create the QSAR models used in the environment-predictor with

.. code-block:: bash

    # input is in ./data/AR_ligands.tsv
    drugex environ -i A2AR_raw.tsv -m RF -r False -t P29274 P29275 -c -s 

This tells DrugEx to use data from :code:`AR_ligands.tsv` to create and train two Random Forrest (:code:`-m RF`) QSAR models
for binary (:code:`-r False`) A2A and A2B (:code:`-t P29274 P29275`) bioactivity predictions. 
The model will be saved to :code:`./envs/single/RF_CLS_P29274.pkg` and model evalution to :code:`./envs/single/RF_CLS_P29274.[cv/ind].tsv`.

Reinforcement Learning
""""""""""""""""""""""

Then, we use a combination of two generators of the same architecture, the agent that is optimized during RL for exploitation and 
the prior that is kept fixed for exploration, to create molecules at each iteration that are scored with the environment-predictor 
that send a back to the agent with 

.. code-block:: bash

    # pretrained model placed in ./generators/chembl27_graph.pkg
    drugex train -m RL -i arl -o arl -ag arl_graph_trans_FT -pr chembl27_graph -ta P29275 -ti P29274 -qed -e 200 -bs 32 -gpu 0,1

This tells DrugEx to create molecules from input fragments encoded in preprocessed data file (prefixed with :code:`a2a`)
and optimize the initial agent-generator (:code:`-ag arl_graph_trans_FT`) with RL (:code:`-m RL`). 
Molecules are scores with a desirability function that favour molecules predicted to be active on A2B (:code:`-ta P29275`), 
inactive on A2A (:code:`-t P29274`) and full criteria of drug-likeness (:code:`-qed`).
Exploration of chemical space is forced by the use of a fixed prior-generator (:code:`-pr chembl27_graph`). 
The training will take a maximum of 200 epochs with a batch size of 32 in parallel on GPUs with IDs 0 and 1. 
The best model will be saved to :code:`./generators/arl_graph_trans_RL.pkg`.

Design new molecules
^^^^^^^^^^^^^^^^^^^^

In this example, we use the optimized exploitation-exploration model to design new compounds that should be active on A2B and inactive on A2A with

.. code-block:: bash

    drugex design -i arl_test_graph.txt -g arl_graph_trans_RL

This tells DrugEx to generate a new molecule per input fragment in :code:`arl_test_graph.txt` with the :code:`arl_graph_trans_RL.pkg` model.
The new compounds are saved to :code:`./new_molecules/arl_graph_trans_RL.tsv`.



..  Advanced
    --------

    Pretraining the Generator
    ^^^^^^^^^^^^^^^^^^^^^^^^^

    Optimizing the QSAR models
    ^^^^^^^^^^^^^^^^^^^^^^^^^^

    Scaffold-based Reinforcement learning
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    All options
    -----------

    dataset
    ^^^^^^^

    environ
    ^^^^^^^

    train
    ^^^^^

    designer
    ^^^^^^^^
