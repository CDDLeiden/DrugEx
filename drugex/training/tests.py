"""
tests

Created by: Martin Sicho
On: 31.05.22, 10:20
"""
import os.path
import tempfile
from collections import OrderedDict
from unittest import TestCase

import pandas as pd

from drugex.data.corpus.corpus import SequenceCorpus
from drugex.data.corpus.vocabulary import VocSmiles, VocGraph, VocGPT
from drugex.data.fragments import GraphFragmentEncoder, FragmentPairsSplitter, SequenceFragmentEncoder, FragmentCorpusEncoder
from drugex.data.processing import CorpusEncoder, Standardization, RandomTrainTestSplitter
from drugex.data.datasets import SmilesDataSet, SmilesFragDataSet, GraphFragDataSet
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.training.environment import DrugExEnvironment
from drugex.training.interfaces import TrainingMonitor
from drugex.training.models import GPT2Model, EncDec, Seq2Seq, RNN, GraphModel
from drugex.training.models.explorer import GraphExplorer, SmilesExplorerNoFrag, SmilesExplorer
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import ParetoSimilarity
from drugex.training.scorers.modifiers import ClippedScore
from drugex.training.scorers.predictors import Predictor
from drugex.training.scorers.properties import Property


class TestModelMonitor(TrainingMonitor):

    def __init__(self, submonitors=None):
        self.model = None
        self.execution = {
            'model' : False,
            'progress' : False,
            'performance' : False,
            'end' : False,
            'close' : False,
        }
        self.submonitors = [
            FileMonitor(tempfile.NamedTemporaryFile().name, verbose=True)
        ] if not submonitors else submonitors

    def passToSubmonitors(self, method, *args, **kwargs):
        for monitor in self.submonitors:
            method = getattr(monitor, method)(*args, **kwargs)

    def saveModel(self, model):
        self.model = model.getModel()
        self.execution['model'] = True
        self.passToSubmonitors('saveModel', model)

    def saveProgress(self, current_step=None, current_epoch=None, total_steps=None, total_epochs=None, *args, **kwargs):
        print("Test Progress Monitor:")
        print(current_step, current_epoch, total_steps, total_epochs)
        print(args)
        print(kwargs)
        self.execution['progress'] = True
        self.passToSubmonitors('saveProgress', current_step, current_epoch, total_steps, total_epochs, *args, **kwargs)

    def savePerformanceInfo(self, current_step=None, current_epoch=None, loss=None, *args, **kwargs):
        print("Test Performance Monitor:")
        print(current_step, current_epoch, loss)
        print(args)
        print(kwargs)
        self.execution['performance'] = True
        self.passToSubmonitors('savePerformanceInfo', current_step, current_epoch, loss, *args, **kwargs)

    def endStep(self, step, epoch):
        print(f"Finished step {step} of epoch {epoch}.")
        self.execution['end'] = True
        self.passToSubmonitors('endStep', step, epoch)

    def close(self):
        print("Training done.")
        self.execution['close'] = True
        self.passToSubmonitors('close')

    def getModel(self):
        return self.model

    def allMethodsExecuted(self):
        return all([self.execution[key] for key in self.execution])


class TrainingTestCase(TestCase):

    # input file information
    test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    pretraining_file = os.path.join(test_data_dir, 'ZINC_raw_small.txt')
    finetuning_file = os.path.join(test_data_dir, 'A2AR_raw_small.txt')

    # global calculation settings
    N_PROC = 2
    N_EPOCHS = 2
    SEED = 42
    MAX_SMILES = 16
    BATCH_SIZE = 8

    # environment objectives (TODO: we should test more options and combinations here)
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

    def setUp(self):
        self.monitor = TestModelMonitor()

    def getTestEnvironment(self, scheme=None):
        """
        Get the testing environment
        Returns:

        """

        scheme = ParetoSimilarity() if not scheme else scheme
        return DrugExEnvironment(self.scorers, thresholds=self.thresholds, reward_scheme=scheme)

    def getSmiles(self, _file):
        """
        Read and standardize SMILES from a file.
        Args:
            _file: input .tsv file with smiles

        Returns:

        """

        return self.standardize(pd.read_csv(_file, header=0, sep='\t')['CANONICAL_SMILES'].sample(self.MAX_SMILES, random_state=self.SEED).tolist())

    def standardize(self, smiles):
        """
        Standardize the input SMILES

        Args:
            smiles: smiles as `list`

        Returns:

        """
        return Standardization(n_proc=self.N_PROC).apply(smiles)

    def setUpSmilesFragData(self):
        """
        Create inputs for the fragment-based SMILES models.

        Returns:

        """

        pre_smiles = self.getSmiles(self.pretraining_file)
        ft_smiles = self.getSmiles(self.finetuning_file)

        # create and encode fragments
        splitter = FragmentPairsSplitter(0.1, 1e4, seed=self.SEED)
        encoder = FragmentCorpusEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=SequenceFragmentEncoder(
                VocSmiles(True)
            ),
            pairs_splitter=splitter,
            n_proc=self.N_PROC
        )

        # get training data
        pr_data_set_test = SmilesFragDataSet(self.getRandomFile())
        pr_data_set_train = SmilesFragDataSet(self.getRandomFile())
        encoder.apply(pre_smiles, encodingCollectors=[pr_data_set_test, pr_data_set_train])
        ft_data_set_test = SmilesFragDataSet(self.getRandomFile())
        ft_data_set_train = SmilesFragDataSet(self.getRandomFile())
        encoder.apply(ft_smiles, encodingCollectors=[ft_data_set_test, ft_data_set_train])

        # get vocabulary (we will join all generated vocabularies to make sure the one used to create data loaders contains all tokens)
        vocabulary = pr_data_set_test.getVoc() + pr_data_set_train.getVoc() + ft_data_set_train.getVoc() + ft_data_set_test.getVoc()
        pr_data_set_test.setVoc(vocabulary)
        pr_data_set_train.setVoc(vocabulary)
        ft_data_set_train.setVoc(vocabulary)
        ft_data_set_test.setVoc(vocabulary)

        pr_loader_train = pr_data_set_train.asDataLoader(self.BATCH_SIZE)
        pr_loader_test = pr_data_set_test.asDataLoader(self.BATCH_SIZE)
        # pr_loader_test = pr_data_set_test.asDataLoader(32, split_converter=SmilesFragDataSet.TargetCreator())
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

        ft_loader_train = pr_data_set_train.asDataLoader(self.BATCH_SIZE)
        ft_loader_test = pr_data_set_test.asDataLoader(self.BATCH_SIZE)
        # ft_loader_test = pr_data_set_test.asDataLoader(self.BATCH_SIZE, split_converter=SmilesFragDataSet.TargetCreator())
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        return pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary

    @staticmethod
    def getRandomFile():
        """
        Generate a random temporary file and return its path.

        Returns:
            path: pth to the file created
        """

        return tempfile.NamedTemporaryFile().name

    def fitTestModel(self, model, train_loader, test_loader):
        monitor = TestModelMonitor()
        model.fit(train_loader, test_loader, epochs=self.N_EPOCHS, monitor=monitor)
        pr_model = monitor.getModel()
        self.assertTrue(type(pr_model) == OrderedDict)
        self.assertTrue(monitor.allMethodsExecuted())
        model.loadStates(pr_model) # initialize from the best state
        return model, monitor

    def test_rnn_nofrags(self):
        """
        Test single network RNN.

        Returns:

        """

        pre_smiles = self.getSmiles(self.pretraining_file)
        ft_smiles = self.getSmiles(self.finetuning_file)

        # get training data
        encoder = CorpusEncoder(
                SequenceCorpus,
                {
                    'vocabulary': VocSmiles(False)
                },
                n_proc=self.N_PROC
        )
        pre_data_set = SmilesDataSet(self.getRandomFile())
        encoder.apply(pre_smiles, pre_data_set)
        ft_data_set = SmilesDataSet(self.getRandomFile())
        encoder.apply(ft_smiles, ft_data_set)

        # get common vocabulary
        vocabulary = pre_data_set.getVoc() + ft_data_set.getVoc()

        # pretraining
        splitter = RandomTrainTestSplitter(0.1)
        pr_loader_train, pr_loader_test = pre_data_set.asDataLoader(self.BATCH_SIZE, splitter=splitter)
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

        pretrained = RNN(vocabulary, is_lstm=True)
        pretrained, monitor = self.fitTestModel(pretrained, pr_loader_train, pr_loader_test)

        # fine-tuning
        splitter = RandomTrainTestSplitter(0.1)
        ft_loader_train, ft_loader_test = ft_data_set.asDataLoader(self.BATCH_SIZE, splitter=splitter)
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        finetuned = RNN(vocabulary, is_lstm=True)
        finetuned.loadStates(pretrained.getModel())
        finetuned, monitor = self.fitTestModel(finetuned, ft_loader_train, ft_loader_test)

        # RL
        environment = self.getTestEnvironment()
        explorer = SmilesExplorerNoFrag(pretrained, env=environment, mutate=finetuned, crover=pretrained)
        monitor = TestModelMonitor()
        explorer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=self.N_EPOCHS)
        self.assertTrue(type(monitor.getModel()) == OrderedDict)
        self.assertTrue(monitor.allMethodsExecuted())

    def test_graph_frags(self):
        """
        Test fragment-based graph model.

        Returns:

        """

        pre_smiles = self.getSmiles(self.pretraining_file)
        ft_smiles = self.getSmiles(self.finetuning_file)

        # create and encode fragments
        splitter = FragmentPairsSplitter(0.1, 1e4, seed=self.SEED)
        encoder = FragmentCorpusEncoder(
            fragmenter=Fragmenter(4, 4, 'brics'),
            encoder=GraphFragmentEncoder(
                VocGraph(n_frags=4)
            ),
            pairs_splitter=splitter,
            n_proc=self.N_PROC
        )

        # get training data
        pr_data_set_test = GraphFragDataSet(self.getRandomFile())
        pr_data_set_train = GraphFragDataSet(self.getRandomFile())
        encoder.apply(pre_smiles, encodingCollectors=[pr_data_set_test, pr_data_set_train])
        ft_data_set_test = GraphFragDataSet(self.getRandomFile())
        ft_data_set_train = GraphFragDataSet(self.getRandomFile())
        encoder.apply(ft_smiles, encodingCollectors=[ft_data_set_test, ft_data_set_train])

        # get vocabulary -> for graph models it is always the default one
        vocabulary = VocGraph()

        # pretraining
        pr_loader_train = pr_data_set_train.asDataLoader(self.BATCH_SIZE)
        pr_loader_test = pr_data_set_test.asDataLoader(self.BATCH_SIZE)
        self.assertTrue(pr_loader_train)
        self.assertTrue(pr_loader_test)

        pretrained = GraphModel(vocabulary)
        pretrained, monitor = self.fitTestModel(pretrained, pr_loader_train, pr_loader_test)

        # test molecule generation
        pretrained.sample_smiles([
            "c1ccncc1CCC",
            "CCO"
        ], num_samples=1)

        # fine-tuning
        ft_loader_train = ft_data_set_train.asDataLoader(self.BATCH_SIZE)
        ft_loader_test = ft_data_set_test.asDataLoader(self.BATCH_SIZE)
        self.assertTrue(ft_loader_train)
        self.assertTrue(ft_loader_test)

        finetuned = GraphModel(vocabulary)
        finetuned.loadStates(pretrained.getModel())
        finetuned, monitor = self.fitTestModel(finetuned, ft_loader_train, ft_loader_test)

        # reinforcement learning
        environment = self.getTestEnvironment()
        explorer = GraphExplorer(pretrained, environment, mutate=finetuned)
        monitor = TestModelMonitor()
        explorer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=self.N_EPOCHS)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())

    def test_smiles_frags_vec(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.setUpSmilesFragData()

        # pretraining
        pretrained = EncDec(vocabulary, vocabulary)
        pretrained, monitor = self.fitTestModel(pretrained, pr_loader_train, pr_loader_test)

        # fine-tuning
        finetuned = EncDec(vocabulary, vocabulary)
        finetuned.loadStates(pretrained.getModel())
        finetuned, monitor = self.fitTestModel(finetuned, ft_loader_train, ft_loader_test)

        # RL
        # FIXME: RL for these models currently not working
        # environment = self.getTestEnvironment()
        # explorer = SmilesExplorer(pretrained, environment, mutate=finetuned, batch_size=self.BATCH_SIZE)
        # monitor = TestModelMonitor()
        # explorer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=self.N_EPOCHS)
        # self.assertTrue(monitor.getModel())
        # self.assertTrue(monitor.allMethodsExecuted())

    def test_smiles_frags_attn(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.setUpSmilesFragData()

        # pretraining
        pretrained = Seq2Seq(vocabulary, vocabulary)
        pretrained, monitor = self.fitTestModel(pretrained, pr_loader_train, pr_loader_test)

        # fine-tuning
        finetuned = Seq2Seq(vocabulary, vocabulary)
        finetuned.loadStates(pretrained.getModel())
        finetuned, monitor = self.fitTestModel(finetuned, ft_loader_train, ft_loader_test)
        self.assertTrue(finetuned)
        self.assertTrue(monitor.getModel())

        # RL
        # FIXME: RL for these models currently not working
        # environment = self.getTestEnvironment()
        # explorer = SmilesExplorer(pretrained, environment, mutate=finetuned, batch_size=self.BATCH_SIZE)
        # monitor = TestModelMonitor()
        # explorer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=self.N_EPOCHS)
        # self.assertTrue(monitor.getModel())
        # self.assertTrue(monitor.allMethodsExecuted())

    def test_smiles_frags_gpt(self):
        pr_loader_train, pr_loader_test, ft_loader_train, ft_loader_test, vocabulary = self.setUpSmilesFragData()

        # pretraining
        vocab_gpt = VocGPT(vocabulary.words)
        pretrained = GPT2Model(vocab_gpt)
        pretrained, monitor = self.fitTestModel(pretrained, pr_loader_train, pr_loader_test)

        # fine-tuning
        finetuned = GPT2Model(vocab_gpt)
        finetuned.loadStates(pretrained.getModel())
        finetuned, monitor = self.fitTestModel(finetuned, ft_loader_train, ft_loader_test)

        # RL
        environment = self.getTestEnvironment()
        explorer = SmilesExplorer(pretrained, environment, mutate=finetuned, batch_size=self.BATCH_SIZE)
        monitor = TestModelMonitor()
        explorer.fit(ft_loader_train, ft_loader_test, monitor=monitor, epochs=self.N_EPOCHS)
        self.assertTrue(monitor.getModel())
        self.assertTrue(monitor.allMethodsExecuted())




