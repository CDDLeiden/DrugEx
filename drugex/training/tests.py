"""
tests

Created by: Martin Sicho
On: 31.05.22, 10:20
"""
import os.path
from unittest import TestCase

import pandas as pd

from drugex.corpus.corpus import SequenceCorpus
from drugex.corpus.vocabulary import VocSmiles, VocGraph, VocGPT
from drugex.datasets.fragments import GraphFragmentEncoder, FragmentPairsSplitter, SequenceFragmentEncoder
from drugex.datasets.splitters import RandomTrainTestSplitter
from drugex.datasets.processing import MoleculeEncoder, SmilesDataSet, Standardization, FragmentEncoder, \
    GraphFragDataSet, SmilesFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training.environment import DrugExEnvironment
from drugex.training.interfaces import TrainingMonitor
from drugex.training.models import GPT2Model, EncDec, Seq2Seq, RNN, GraphModel
from drugex.training.models.explorer import GraphExplorer, SmilesExplorerNoFrag, SmilesExplorer
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.predictors import Predictor
from drugex.training.scorers.properties import Property
from drugex.training.trainers import Pretrainer, FineTuner, Reinforcer


class TestModelMonitor(TrainingMonitor):

    def __init__(self):
        self.model = None
        self.execution = {
            'model' : False,
            'progress' : False,
            'performance' : False,
            'end' : False,
            'close' : False,
        }

    def saveModel(self, model):
        self.model = model
        self.execution['model'] = True

    def saveProgress(self, current_step, current_epoch, total_steps, total_epochs, *args, **kwargs):
        print("Test Progress Monitor:")
        print(current_step, current_epoch, total_steps, total_epochs)
        print(args)
        print(kwargs)
        self.execution['progress'] = True

    def savePerformanceInfo(self, current_step, current_epoch, loss, *args, **kwargs):
        print("Test Performance Monitor:")
        print(current_step, current_epoch, loss)
        print(args)
        print(kwargs)
        self.execution['performance'] = True

    def endStep(self, step, epoch):
        print(f"Finished step {step} of epoch {epoch}.")
        self.execution['end'] = True

    def close(self):
        print("Training done.")
        self.execution['close'] = True

    def getModel(self):
        return self.model

    def allMethodsExecuted(self):
        return all([self.execution[key] for key in self.execution])


class TrainingTestCase(TestCase):

    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    pretraining_file = os.path.join(test_data_dir, 'ZINC_raw_small.txt')
    finetuning_file = os.path.join(test_data_dir, 'A2AR_raw_small.txt')

    N_PROC = 2

    activity_threshold = 6.5
    pad = 3.5
    scorers = [
        Property(
            "MW",
            modifier=ClippedScore(lower_x=1000, upper_x=500)
        ),
        Predictor.fromFile(
            os.path.join(os.path.dirname(__file__), "test_data/RF_REG_P29274_0006.pkg"),
            type="REG",
            modifier=ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold)
        ),

    ]
    thresholds = [0.5, 0.99]

    def getTestEnvironment(self):
        return DrugExEnvironment(self.scorers, thresholds=self.thresholds)

    def getSmilesPretrain(self):
        return self.standardize(pd.read_csv(self.pretraining_file, header=0, sep='\t')['CANONICAL_SMILES'].tolist())

    def getSmilesFinetune(self):
        return self.standardize(pd.read_csv(self.finetuning_file, header=0, sep='\t')['CANONICAL_SMILES'].tolist())

    def standardize(self, smiles):
        return Standardization(n_proc=self.N_PROC).applyTo(smiles)

    def setUpSmilesFragData(self):
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

        pr_loader_train = pr_data_set_train.asDataLoader(32)
        pr_loader_test = pr_data_set_test.asDataLoader(split_converter=SmilesFragDataSet.TargetSplitConverter(32, vocabulary))
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

        ft_loader_train = pr_data_set_train.asDataLoader(32)
        ft_loader_test = pr_data_set_test.asDataLoader(split_converter=SmilesFragDataSet.TargetSplitConverter(32, vocabulary))
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        self.smiles_frags_data = pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary

    def setUp(self):
        self.setUpSmilesFragData()

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

        algorithm = RNN(vocabulary, is_lstm=True)
        pretrainer = Pretrainer(algorithm)
        monitor = TestModelMonitor()
        pretrainer.fit(pr_loader_train, pr_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        pr_model = monitor.getModel()

        # fine-tuning
        splitter = RandomTrainTestSplitter(0.1)
        ft_loader_train, ft_loader_test = ft_data_set.asDataLoader(32, splitter=splitter)
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        finetuner = FineTuner(algorithm)
        monitor = TestModelMonitor()
        finetuner.fit(ft_loader_train, ft_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        ft_model = monitor.getModel()

        # RL
        environment = self.getTestEnvironment()
        explorer = SmilesExplorerNoFrag(pr_model, env=environment, mutate=ft_model, crover=pr_model)
        reinforcer = Reinforcer(explorer)
        monitor = TestModelMonitor()
        reinforcer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=2)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())


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

        algorithm = GraphModel(vocabulary)
        pretrainer = Pretrainer(algorithm)
        monitor = TestModelMonitor()
        pretrainer.fit(pr_loader_train, pr_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        pr_model = monitor.getModel()

        # fine-tuning
        ft_loader_train = ft_data_set_train.asDataLoader(128)
        ft_loader_test = ft_data_set_test.asDataLoader(256)
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        tuner = FineTuner(pretrainer.getModel())
        monitor = TestModelMonitor()
        tuner.fit(ft_loader_train, ft_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        ft_model = monitor.getModel()

        # reinforcement learning
        environment = self.getTestEnvironment()
        explorer = GraphExplorer(pr_model, environment, mutate=ft_model)
        reinforcer = Reinforcer(explorer)
        monitor = TestModelMonitor()
        reinforcer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=2)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

    def test_smiles_frags_vec(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.smiles_frags_data

        # pretraining
        algorithm = EncDec(vocabulary, vocabulary)
        pretrainer = Pretrainer(algorithm)
        monitor = TestModelMonitor()
        pretrainer.fit(pr_loader_train, pr_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

        # fine-tuning
        finetuner = FineTuner(pretrainer.getModel())
        monitor = TestModelMonitor()
        finetuner.fit(ft_loader_train, ft_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

        # FIXME: RL for these models currently not working

    def test_smiles_frags_attn(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.smiles_frags_data

        # pretraining
        algorithm = Seq2Seq(vocabulary, vocabulary)
        pretrainer = Pretrainer(algorithm)
        monitor = TestModelMonitor()
        pretrainer.fit(pr_loader_train, pr_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

        # fine-tuning
        finetuner = FineTuner(pretrainer.getModel())
        monitor = TestModelMonitor()
        finetuner.fit(ft_loader_train, ft_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

        # FIXME: RL for these models currently not working

    def test_smiles_frags_gpt(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.smiles_frags_data

        # pretraining
        vocab_gpt = VocGPT(vocabulary.words)
        algorithm = GPT2Model(vocab_gpt, n_layer=12)
        pretrainer = Pretrainer(algorithm)
        monitor = TestModelMonitor()
        pretrainer.fit(pr_loader_train, pr_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        pr_model = monitor.getModel()

        # fine-tuning
        finetuner = FineTuner(pretrainer.getModel())
        monitor = TestModelMonitor()
        finetuner.fit(ft_loader_train, ft_loader_test, epochs=2, monitor=monitor)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())
        ft_model = monitor.getModel()

        # RL
        environment = self.getTestEnvironment()
        explorer = SmilesExplorer(pr_model, environment, mutate=ft_model, batch_size=32)
        reinforcer = Reinforcer(explorer)
        monitor = TestModelMonitor()
        reinforcer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=2)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())




