"""
tests

Created by: Martin Sicho
On: 31.05.22, 10:20
"""
import os.path
from unittest import TestCase

import pandas as pd

from drugex.corpus.corpus import SequenceCorpus
from drugex.corpus.vocabulary import VocSmiles, VocGraph
from drugex.datasets.fragments import GraphFragmentEncoder, FragmentPairsSplitter, SequenceFragmentEncoder
from drugex.datasets.splitters import RandomTrainTestSplitter
from drugex.datasets.processing import MoleculeEncoder, SmilesDataSet, Standardization, FragmentEncoder, \
    GraphFragDataSet, SmilesFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter


class TrainingTestCase(TestCase):

    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    pretraining_file = os.path.join(test_data_dir, 'ZINC_raw_small.txt')
    finetuning_file = os.path.join(test_data_dir, 'A2AR_raw_small.txt')

    N_PROC = 2

    def getSmilesPretrain(self):
        return self.standardize(pd.read_csv(self.pretraining_file, header=0, sep='\t')['CANONICAL_SMILES'].tolist())

    def getSmilesFinetune(self):
        return self.standardize(pd.read_csv(self.finetuning_file, header=0, sep='\t')['CANONICAL_SMILES'].tolist())

    def standardize(self, smiles):
        return Standardization(n_proc=self.N_PROC).applyTo(smiles)

    def test_rnn_nofrags(self):
        pre_smiles = self.getSmilesPretrain()
        ft_smiles = self.getSmilesFinetune()

        # get training data
        encoder = MoleculeEncoder(
                SequenceCorpus,
                {
                    'vocabulary': VocSmiles()
                },
                n_proc=self.N_PROC
        )
        pre_data_set = SmilesDataSet('pretrain')
        encoder.applyTo(pre_smiles, pre_data_set)
        ft_data_set = SmilesDataSet('finetune')
        encoder.applyTo(ft_smiles, ft_data_set)

        # get common vocabulary
        vocabulary = pre_data_set.getVoc() + ft_data_set.getVoc()

        # pretraining
        splitter = RandomTrainTestSplitter(0.1)
        pr_loader_train, pr_loader_test = pre_data_set.asDataLoader(32, splitter=splitter)
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

    def test_graph_frags(self):
        pre_smiles = self.getSmilesPretrain()
        ft_smiles = self.getSmilesFinetune()

        # create and encode fragments
        splitter = FragmentPairsSplitter(0.1, 1e4, unique_only=True)
        encoder = FragmentEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=GraphFragmentEncoder(
                VocGraph(n_frags=4)
            ),
            pairs_splitter=splitter,
            n_proc=self.N_PROC
        )

        # get training data
        pr_data_set_test = GraphFragDataSet('pretrain')
        pr_data_set_train = GraphFragDataSet('pretrain')
        encoder.applyTo(pre_smiles, encodingCollectors=[pr_data_set_test, pr_data_set_train])
        ft_data_set_test = GraphFragDataSet('finetune')
        ft_data_set_train = GraphFragDataSet('finetune')
        encoder.applyTo(ft_smiles, encodingCollectors=[ft_data_set_test, ft_data_set_train])

        # get vocabulary -> for graph models it is always the default one
        vocabulary = VocGraph()

        # pretraining
        pr_loader_train = pr_data_set_train.asDataLoader(128)
        pr_loader_test = pr_data_set_test.asDataLoader(256)
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

    def test_smiles_frags(self):
        pre_smiles = self.getSmilesPretrain()
        ft_smiles = self.getSmilesFinetune()

        # create and encode fragments
        splitter = FragmentPairsSplitter(0.1, 1e4, unique_only=True)
        encoder = FragmentEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=SequenceFragmentEncoder(
                VocSmiles()
            ),
            pairs_splitter=splitter,
            n_proc=self.N_PROC
        )

        # get training data
        pr_data_set_test = SmilesFragDataSet('pretrain')
        pr_data_set_train = SmilesFragDataSet('pretrain')
        encoder.applyTo(pre_smiles, encodingCollectors=[pr_data_set_test, pr_data_set_train])
        ft_data_set_test = SmilesFragDataSet('finetune')
        ft_data_set_train = SmilesFragDataSet('finetune')
        encoder.applyTo(ft_smiles, encodingCollectors=[ft_data_set_test, ft_data_set_train])

        # get vocabulary (we will join all generated vocabularies to make sure the one used to create data loaders contains all tokens)
        vocabulary = pr_data_set_test.getVoc() + pr_data_set_train.getVoc() + ft_data_set_train.getVoc() + ft_data_set_test.getVoc()
        pr_data_set_test.setVoc(vocabulary)
        pr_data_set_train.setVoc(vocabulary)
        ft_data_set_train.setVoc(vocabulary)
        ft_data_set_test.setVoc(vocabulary)

        # pretraining
        pr_loader_train = pr_data_set_train.asDataLoader(32)
        pr_loader_test = pr_data_set_test.asDataLoader(split_converter=SmilesFragDataSet.TargetSplitConverter(32, vocabulary))
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)






