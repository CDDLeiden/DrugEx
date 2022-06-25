..  _usage:

Usage
=====

You can use the command line interface to preprocess data and build models. You will need to run multiple scripts to be able to obtain a final model. The description of the functionality of each script can be displayed with the :code:`--help` argument. For example, the help message for the :code:`drugex.dataset` script can be shown as follows:

..  code-block::

    python -m drugex.dataset --help

A simple command line workflow to fine-tune and optimize a graph-based model is given below (see :ref:`cli-example`).

If you want more control over the inputs and outputs or want to customize DrugEx a bit more, you can also use the Python API directly (see :ref:`api-docs`). You can find a tutorial with Jupyter notebooks illustrating some common use cases in the project `source code <https://github.com/CDDLeiden/DrugEx/tree/master/tutorial>`_.

..  _cli-example:

CLI Example
~~~~~~~~~~~

In this example, we will finetune an already pretrained graph transformer, but before we do that we have to generate the appropriate training data. Let's assume we want to bias the model towards generating compounds that are more related to known ligands of the A2A receptor. DrugEx assumes that all input data are saved in the :code:`data` folder of the directory it is executed from. Therefore, we place the compounds that will serve as a template for the finetuning inside this folder and execute DrugEx as follows:

..  code-block:: bash

    # input is in ./data/A2AR_raw.tsv
    python -m drugex.dataset -i A2AR_raw.tsv -mc CANONICAL_SMILES -o ft_a2a -mt graph

This will tell DrugEx to preprocess compounds saved in the the :code:`CANONICAL_SMILES` column of the :code:`A2AR_raw.tsv` file. The resulting input files will be saved in the data folder and given a prefix (starting with :code:`ft_a2a`) and an ID after a successful run. You can then use this ID (with the :code:`-p` parameter) to supply these files to the finetuning function of the :code:`train` script:

..  code-block:: bash

    # pretrained model placed in ./generators/chembl27_graph.pkg
    python -m drugex.train -i ft_a2a_4:4_brics -p 0001 -m FT -o ft_a2a_model -pt chembl27_graph.pkg -a graph -e 200 -bs 32 -gpu 0,1,2

This tells DrugEx to use the generated file (prefixed with :code:`ft_a2a_4:4_brics` and ID :code:`0001`) to finetune (:code:`-m FT`) a pretrained model with states saved in the :code:`chembl27_graph.pkg` file). The training will take a maxiumum of 200 epochs with batch size of 32 in parallel on GPUs with IDs 0,1 and 2.

Reinforcement Learning
----------------------

..  todo:: Add examples of using the environ script to create an environment and then using it for RL.

