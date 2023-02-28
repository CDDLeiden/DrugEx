#!/usr/bin/env python
import os
import json
import argparse
import warnings

from qsprpred.models.interfaces import QSPRModel
from qsprpred.models.tasks import ModelTasks

from drugex import VERSION
from drugex.data.corpus.vocabulary import VocGraph, VocSmiles
from drugex.data.datasets import (GraphFragDataSet, SmilesDataSet,
                                  SmilesFragDataSet)
from drugex.logs.utils import backUpFiles, commit_hash, enable_file_logger
from drugex.training.environment import DrugExEnvironment
from drugex.training.explorers import (FragGraphExplorer, FragSequenceExplorer,
                                       SequenceExplorer)
from drugex.training.generators import (GraphTransformer, SequenceRNN,
                                        SequenceTransformer)
from drugex.training.monitors import FileMonitor
from drugex.training.rewards import (ParetoCrowdingDistance,
                                     ParetoTanimotoDistance, WeightedSum)
from drugex.training.scorers.modifiers import ClippedScore, SmoothHump
from drugex.training.scorers.properties import (LigandEfficiency,
                                                LipophilicEfficiency, Property,
                                                Uniqueness)
from drugex.training.scorers.similarity import (FraggleSimilarity,
                                                TverskyFingerprintSimilarity,
                                                TverskyGraphSimilarity)

from drugex.training.scorers.qsprpred import QSPRPredScorer

warnings.filterwarnings("ignore")
    
def GeneratorArgParser():
    """ Define and read command line arguments """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # I/O parameters
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains folders 'data' (and 'qspr' if a QSPRpred model is used for scoring) and where 'generators' will be created to save output files.")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Prefix (using separate files for training and validation) or \
                          full name of input file(s) (used both for training and validation) containing encoded molecules or fragments-molecule pairs.")  
    parser.add_argument('-vfs', '--voc_files', type=str, nargs='*', default=None,
                        help="List of vocabulary files name (with or without the '.txt.vocab' extension). If None, set to be the first word of input and if no vocabulary file is found, a default vocabulary is used.")
    parser.add_argument('-o', '--output', type=str, default=None,
                        help="Prefix of output files. If None, set to be the first word of input.")
    # Input models 
    parser.add_argument('-ag', '--agent_path', type=str, default=None,
                        help="Name of model (w/o .pkg extension) or full path to agent model. Only used in FT and RL modes.")
    parser.add_argument('-pr', '--prior_path', type=str, default=None,
                        help="Name of model (w/o .pkg extension) of full path to prior model. Only used in RL mode.")
    
    # General parameters
    parser.add_argument('-tm', '--training_mode', type=str, default='RL',
                        help="Mode, of the training: 'PT' for pretraining, 'FT' for fine-tuning and 'RL' for reinforcement learning") 
    parser.add_argument('-mt', '--mol_type', type=str, default='graph',
                        help="Molecule encoding type: 'smiles' or 'graph'")    
    parser.add_argument('-a', '--algorithm', type=str, default='trans',
                        help="Generator algorithm: 'trans' (graph- or smiles-based transformer model) or "\
                             "'rnn' (smiles-based recurrent neural network).")
    parser.add_argument('-gru', '--use_gru', action='store_true',
                        help="If on, use GRU units for the RNN model. Ignore if algorithm is not 'rnn'")
    parser.add_argument('-e', '--epochs', type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-gpu', '--use_gpus', type=str, default='1,2,3,4',
                        help="List of GPUs") 
    parser.add_argument('-pa', '--patience', type=int, default=50,
                        help="Number of epochs to wait before early stop if no progress on test set score.")
    
    
    # RL parameters
    parser.add_argument('-ns', '--n_samples', type=int, default=-1, 
                        help="During RL, n_samples and 0.2*n_samples random input fragments are used for training and validation at each epoch. If -1, all input data is used at once each epoch.") 
    parser.add_argument('-eps', '--epsilon', type=float, default=0.1,
                        help="Exploring rate: probability of using the prior model instead of the agent model.")
    parser.add_argument('-bet', '--beta', type=float, default=0.0,
                        help="Reward baseline used by the policy gradient algorithm.")
    parser.add_argument('-s', '--scheme', type=str, default='PRTD',
                        help="Multi-objective reward calculation scheme: 'WS' for weighted sum, 'PRTD' for Pareto ranking with Tanimoto distance or 'PRCD' for Pareto ranking with crowding distance.")

    # Affinity models
    parser.add_argument('-p', '--predictor', type=str, nargs='*', default=['RF'],
                        help="The path to the serialized metadata of a QSPRPred model (ie. 'RF_meta.json'). If different environments are required give environment of targets in order active, inactive, window.")
    parser.add_argument('-at', '--activity_threshold', type=float, default=6.5,
                        help="Activity threshold")
    parser.add_argument('-ta', '--active_targets', type=str, nargs='*', default=[],
                        help="Names of models that predict activity.")
    parser.add_argument('-ti', '--inactive_targets', type=str, nargs='*', default=[],
                        help="Names of models that predict inactivity.")
    parser.add_argument('-tw', '--window_targets', type=str, nargs='*', default=[],
                        help="Names of models for which selectivity window is calculated.")
    parser.add_argument('-le', '--ligand_efficiency', action='store_true', 
                        help="If on, use the ligand efficiency instead of the simple affinity as objective for active targets.")
    parser.add_argument('-le_ths', '--le_thresholds', type=float, nargs=2, default=[0.0, 0.5],
                        help='Thresholds used calculate ligand efficiency clipped scores in the desirability function.')
    parser.add_argument('-lipe', '--lipophilic_efficiency', action='store_true',
                        help="If on, use the ligand lipophilic efficiency instead of the simple affinity as objective for active targets.")
    parser.add_argument('-lipe_ths', '--lipe_thresholds', type=float, nargs=2, default=[4.0, 6.0],
                        help='Thresholds used calculate lipophilic efficiency clipped scores in the desirability function.')
    
    # Pre-implemented properties
    parser.add_argument('-qed', '--qed', action='store_true',
                        help="If on, QED is used in desirability function")
    parser.add_argument('-unq', '--uniqueness', action='store_true',
                        help="If on, molecule uniqueness is used in desirability function")
    parser.add_argument('-sas', '--sa_score', action='store_true',
                        help="If on, Synthetic Accessibility score is used in desirability function")       
    parser.add_argument('-ras', '--ra_score', action='store_true',
                        help="If on, Retrosynthesis Accessibility score is used in desirability function")
    parser.add_argument('-mw', '--molecular_weight', action='store_true',
                        help='If on, compounds with molecular weights outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-mw_ths', '--mw_thresholds', type=int, nargs='*', default=[200, 600],
                        help='Thresholds used calculate molecular weights clipped scores in the desirability function.')
    parser.add_argument('-logP', '--logP', action='store_true',
                        help='If on, compounds with logP values outside a range set by mw_thersholds are penalized in the desirability function')
    parser.add_argument('-logP_ths', '--logP_thresholds', type=float, nargs='*', default=[-5, 5],
                        help='Thresholds used calculate logP clipped scores in the desirability function')
    parser.add_argument('-tpsa', '--tpsa', action='store_true',
                        help='If on, topology polar surface area is used in desirability function')
    parser.add_argument('-tpsa_ths', '--tpsa_thresholds', type=float, nargs=2, default=[0, 140],
                        help='Thresholds used calculate TPSA clipped scores in the desirability function')
    parser.add_argument('-sim_mol', '--similarity_mol', type=str, default=None,
                        help='SMILES string of a reference molecule to which the similarity is used as an objective. Similarity metric and threshold set by --sim_metric and --sim_th.')
    parser.add_argument('-sim_type', '--similarity_type', type=str, default='fraggle',
                        help="'fraggle' for Fraggle similarity, 'graph' for Tversky similarity between graphs or fingerprints name ('AP', 'PHCO', 'BPF', 'BTF', 'PATH', 'ECFP4', 'ECFP6', 'FCFP4', 'FCFP6') for Tversky similarity between fingeprints")
    parser.add_argument('-sim_th', '--similarity_threshold', type=float, default=0.5,
                        help="Threshold for molecular similarity to reference molecule")
    parser.add_argument('-sim_tw', '--similarity_tversky_weights', nargs=2, type=float, default=[0.7, 0.3],
                        help="Weights (alpha and beta) for Tversky similarity. If both equal to 1.0, Tanimoto similarity.")       
    

    
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    parser.add_argument('-d', '--debug', action='store_true')

            
    args = parser.parse_args()
    
    # Setting output file prefix from input file
    if args.output is None:
        args.output = args.input.split('_')[0]

    # Setting vocabulary file names from input file
    if args.voc_files is None:
        args.voc_files = [args.input.split('_')[0]]
    
    # Setting target IDs as union of active, inactive and window targets
    args.targets = args.active_targets + args.inactive_targets + args.window_targets

    # Setting output file prefix 
    args.output_long = '_'.join([args.output, args.mol_type, args.algorithm, args.training_mode])
    args.output_file_base = f'{args.base_dir}/generators/{args.output_long}'

    args.use_gpus = [int(x) for x in args.use_gpus.split(',')]

    return args

class DataPreparation():

    """ 
    Class for preparing data for training and testing from input files.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    """
    def __init__(self, base_dir, voc_files, input, batch_size, n_samples, mt):

        """ 
        Parameters
        ----------
        base_dir : str
            Path to base directory.
        voc_files : list
            List of vocabulary file names.
        input : str
            Name of input file or prefix of input files.
        batch_size : int
            Batch size.
        n_samples : int
            Number of samples.
        mt : str
            Molecule type.
        """

        self.base_dir = base_dir
        self.data_dir = f'{base_dir}/data'
        self.voc_files = voc_files
        self.input = input
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.mt = mt

    def getVocPaths(self):
        """ 
        Get paths to vocabulary files. If none are found, use internal defaults.

        Returns
        -------
        list
            List of paths to vocabulary files.
        """

        voc_paths = []
        for voc_file in self.voc_files:
            if os.path.exists(f'{self.data_dir}/{voc_file}'):
                # Vocabulary from full path name
                voc_paths.append(f'{self.data_dir}/{voc_file}')
            elif os.path.exists(f'{self.data_dir}/{voc_file}_{self.mt}.txt.vocab'):
                # Vocabulary from file name without extension
                voc_paths.append(f'{self.data_dir}/{voc_file}_{self.mt}.txt.vocab')
            else:
                log.warning(f'Could not find vocabulary file {voc_file} or {voc_file}.txt.vocab in {self.data_dir}.')
                
        if len(voc_paths) == 0 :
            log.warning(f'No vocabulary files found. Using internal defaults for DrugEx v{VERSION}.')

        return voc_paths
    
    def getDataPaths(self):

        """ 
        Get paths to training and test data files.
        
        Returns
        -------
        Tuple[str, str]
            Paths to training and test data files.
        """

        # If exact data path was given as input, that data is both used for training and testing
        if os.path.exists(f'{self.data_dir}/{self.input}'):
            train_path = f'{self.data_dir}/{self.input}'
            test_path = train_path
        
        # Else if prefix was given, read separate train and test sets
        else:
            train_set = 'unique' if self.unique_frags else 'train'
            train_path = f'{self.data_dir}/{self.input}_{train_set}_{self.mol_type}.txt'
            test_path = f'{self.data_dir}/{self.input}_test_{self.mol_type}.txt'
 
        assert os.path.exists(train_path), f'{train_path} does not exist'
        assert os.path.exists(test_path), f'{test_path} does not exist'          
            
        log.info(f'Loading training data from {train_path}')
        log.info(f'Loading validation data from {test_path}')

        return train_path, test_path   

class FragGraphDataPreparation(DataPreparation):

    """ 
    Data preprocessing for graph-based fragment generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'graph'
    """

    # Initialize class
    def __init__(self, base_dir, voc_files, input, batch_size, n_samples, unique_frags=False):
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'graph')
        self.unique_frags = unique_frags
        self.mol_type = 'graph'

        """ 
        Parameters
        ----------
        base_dir : str
            Path to base directory.
        voc_files : list
            List of vocabulary file names.
        input : str
            Name of input file or prefix of input files.
        batch_size : int
            Batch size.
        n_samples : int
            Number of samples.
        unique_frags : bool
            If True, only unique fragments are used for training.
        """

    def __call__(self):

        """
        Get data loaders for training and testing and vocabulary.
        
        Returns
        -------
        Tuple[VocGraph, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        # Get vocabulary and data paths
        voc_paths = self.getVocPaths()
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = GraphFragDataSet(train_path)
        if voc_paths:
            # TODO: SOFTCODE number of fragments (and max lenght?)
            dataset_train.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
        # TODO: rediscuss *4
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size * 4, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = GraphFragDataSet(test_path)
        if voc_paths:
            # TODO: SOFTCODE number of fragments (and max lenght?)
            dataset_test.readVocs(voc_paths, VocGraph, max_len=80, n_frags=4)
        # TODO: rediscuss *10
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size * 10, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc() + dataset_test.getVoc()
        
        return voc, train_loader, valid_loader

class FragSmilesDataPreparation(DataPreparation):
    """ 
    Data preprocessing for SMILES-based fragment generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'smiles'
    """

    def __init__(self, base_dir, voc_files, input, batch_size, n_samples, unique_frags=False):
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'smiles')

        """ 
        Parameters
        ----------
        base_dir : str
            Path to base directory.
        voc_files : list
            List of vocabulary file names.
        input : str
            Name of input file or prefix of input files.
        batch_size : int
            Batch size.
        n_samples : int
            Number of samples.
        unique_frags : bool
            If True, only unique fragments are used for training.
        """
        self.unique_frags = unique_frags
        self.mol_type = 'smiles'


    def __call__(self):
        """
        Get data loaders for training and testing and vocabulary.
        
        Returns
        -------
        Tuple[VocSmiles, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """
            
        # Get vocabulary and data paths
        voc_paths = self.getVocPaths()
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = SmilesFragDataSet(train_path)
        # TODO: SOFTCODE max_len ?
        dataset_train.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=True)
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = SmilesFragDataSet(test_path)
        dataset_test.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=True)
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc()
        
        return voc, train_loader, valid_loader
    
class SmilesDataPreparation(DataPreparation):
    """ 
    Data preprocessing for SMILES-based molecule generation.

    Attributes
    ----------
    base_dir : str
        Path to base directory.
    data_dir : str
        Path to data directory.
    voc_files : list
        List of vocabulary file names.
    input : str
        Name of input file or prefix of input files.
    batch_size : int
        Batch size.
    n_samples : int
        Number of samples.
    unique_frags : bool
        If True, only unique fragments are used for training.
    mol_type : str
        Type of molecule representation: 'smiles'
    """

    def __init__(self, base_dir, voc_files, input, batch_size, n_samples, unique_frags=False):
        super().__init__(base_dir, voc_files, input, batch_size, n_samples, 'corpus')
        """ 
        Parameters
        ----------
        base_dir : str
            Path to base directory.
        voc_files : list
            List of vocabulary file names.
        input : str
            Name of input file or prefix of input files.
        batch_size : int
            Batch size.
        n_samples : int
            Number of samples.
        unique_frags : bool
            If True, only unique fragments are used for training.
        """
        self.unique_frags = False
        self.mol_type = 'smiles'

    def __call__(self):
        """
        Get data loaders for training and testing and vocabulary.
        
        Returns
        -------
        Tuple[VocGraph, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """
            
        # Get vocabulary and data paths
        voc_paths = self.getVocPaths()
        train_path, test_path = self.getDataPaths()

        # Get training data loader
        dataset_train = SmilesDataSet(train_path)
        # TODO: SOFTCODE max_len ?
        if voc_paths:
            dataset_train.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=False)
        train_loader = dataset_train.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples)

        # Get test data loader
        dataset_test = SmilesDataSet(test_path)
        dataset_test.readVocs(voc_paths, VocSmiles, max_len=100, encode_frags=False)
        valid_loader = dataset_test.asDataLoader(batch_size=self.batch_size, n_samples=self.n_samples, n_samples_ratio=0.2)

        # Get vocabulary
        voc = dataset_train.getVoc()
        
        return voc, train_loader, valid_loader

class SetUpGenerator():

    """ 
    Set up generator object for training or molecule generation
    """

    def __init__(self, args):

        """ 
        Set up generator object for training or molecule generation
        
        Parameters
        ----------
        args : object
            Command line arguments
        """
        # Set up attributes from command line arguments
        for key, value in args.__dict__.items():
            setattr(self, key, value)

    def setGeneratorAlgorithm(self, voc):

        """ 
        Set generator algorithm based on molecule type and generator algorithm type
        
        Parameters
        ----------
        voc : object
            Vocabulary object
        
        Returns
        -------
        generator : object
            DrugEx Generator object
        """

        if self.mol_type == 'graph' and self.algorithm == 'trans':
            generator = GraphTransformer(voc, use_gpus=self.use_gpus)
        
        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            generator = SequenceTransformer(voc, use_gpus=self.use_gpus)
        
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            generator = SequenceRNN(voc, is_lstm= not self.use_gru, use_gpus=self.use_gpus)

        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return generator
    
    def prepareInputs(self):
        """ 
        Prepare inputs for training or molecule generation
        
        Returns
        -------
        Tuple[Vocabulary, DataLoader, DataLoader]
            Vocabulary, training data loader, and test data loader.
        """

        if self.mol_type == 'graph' and self.algorithm == 'trans':
            voc, train_loader, test_loader = FragGraphDataPreparation(self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples, self.unique_frags)()

        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            voc, train_loader, test_loader = FragSmilesDataPreparation(self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples, self.unique_frags)()
        
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            voc, train_loader, test_loader = SmilesDataPreparation(self.base_dir, self.voc_files, self.input, self.batch_size, self.n_samples)()
        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return voc, train_loader, test_loader  
    
    def loadStatesFromFile(self, generator, generator_path):
        """
        Load pretrained weights from file

        Parameters
        ----------
        generator : object
            DrugEx Generator object
        generator_path : str
            Path to pretrained weights
        
        Returns
        -------
        generator : object
            DrugEx Generator object
        """

        # Load pretrained weights
        if os.path.exists(f'{generator_path}'):
            # Load pretrained weights from absolute path
            generator.loadStatesFromFile(f'{generator_path}')
        elif os.path.exists(f'{self.base_dir}/generators/{generator_path}.pkg'):
            # Load pretrained weights from generators folder
            generator.loadStatesFromFile(f'{self.base_dir}/generators/{generator_path}.pkg')
        else:
            raise ValueError(f'Could not find pretrained model at {generator_path} or {self.base_dir}/generators/{generator_path}.pkg')
              
        return generator
    
class Pretrain(SetUpGenerator):

    """ 
    Pretrain generator
    """

    # Initialize class
    def __init__(self, args):
        super().__init__(args)
        """ 
        Parameters
        ----------
        args : Namespace
            Command line arguments
        """
        self.unique_frags = False

    def __call__(self):
        """ Pretrain generator """

        # Get vocabulary and data loaders
        voc, train_loader, test_loader = self.prepareInputs()

        # Set generator algorithm
        agent = self.setGeneratorAlgorithm(voc)

        # Set monitoring
        monitor = FileMonitor(self.output_file_base, verbose=True)

        # Fit 
        log.info('Pretraining generator...')
        agent.fit(train_loader, test_loader, epochs=self.epochs, monitor=monitor, patience=self.patience)


class Finetune(SetUpGenerator):
    """ 
    Finetune generator
    """

    def __init__(self, args):
        super().__init__(args)
        """
        Parameters
        ----------
        args : Namespace
            Command line arguments
        """
        self.unique_frags = False

    def __call__(self):
        """ Finetune generator """

        # Get vocabulary and data loaders
        voc, train_loader, test_loader = self.prepareInputs()

        # Set generator algorithm and load pretrained weights
        agent = self.setGeneratorAlgorithm(voc)
        agent = self.loadStatesFromFile(agent, self.agent_path)
        
        # Set monitoring
        monitor = FileMonitor(self.output_file_base, verbose=True)

        # Fit 
        log.info('Finetuning generator...')
        agent.fit(train_loader, test_loader, epochs=self.epochs, monitor=monitor, patience=self.patience)


class Reinforce(SetUpGenerator):
    """ Train generator with reinforcement learning """

    # Initialize class
    def __init__(self, args):
        """ 
        Parameters
        ----------
        args : Namespace
            Command line arguments
        """
        super().__init__(args)
        self.unique_frags = True

    def setExplorer(self, agent, prior, env):
        """
        Set explorer algorithm and parameters

        Parameters
        ----------
        agent : object
            DrugEx Generator object to be fitted
        prior : object
            DrugEx Generator object to be used as prior
        env : object
            DrugEx Environment object to score molecules

        Returns
        -------
        explorer : object
            DrugEx Explorer object
        """

        kwargs = {
            'mutate': prior,
            'batch_size': self.batch_size,
            'epsilon': self.epsilon,
            'beta': self.beta,
            'n_samples': self.n_samples,
            'use_gpus': self.use_gpus
        }

        if self.mol_type == 'graph' and self.algorithm == 'trans':
            explorer = FragGraphExplorer(agent, env, **kwargs)        
        elif self.mol_type == 'smiles' and self.algorithm == 'trans':
            explorer = FragSequenceExplorer(agent, env, **kwargs)
        elif self.mol_type == 'smiles' and self.algorithm == 'rnn':
            explorer = SequenceExplorer(agent, env, **kwargs)
        else:
            raise ValueError(f'Unknown generator algorithm {self.algorithm} for molecule type {self.mol_type}')

        return explorer        

    def __call__(self):
        """ Train generator with reinforcement learning """
            
        # Get vocabulary and data loaders
        voc, train_loader, test_loader = self.prepareInputs()
    
        # Set agent algorithm and load pretrained weights
        agent = self.setGeneratorAlgorithm(voc)
        agent = self.loadStatesFromFile(agent, self.agent_path)

        # Set prior algorithm and load pretrained weights
        prior = self.setGeneratorAlgorithm(voc)
        prior = self.loadStatesFromFile(prior, self.prior_path)

        # Set environment
        environment = CreateEnvironment(
            self.base_dir,
            self.predictor,
            self.scheme,
            active_targets=self.active_targets,
            inactive_targets=self.inactive_targets,
            window_targets=self.window_targets,
            activity_threshold=self.activity_threshold,
            qed=self.qed,
            unique=self.uniqueness, 
            sa_score=self.sa_score,
            ra_score=self.ra_score,
            mw=self.molecular_weight,
            mw_ths=self.mw_thresholds,
            logP=self.logP,
            logP_ths=self.logP_thresholds,
            tpsa=self.tpsa,
            tpsa_ths=self.tpsa_thresholds,
            sim_smiles=self.similarity_mol,
            sim_type=self.similarity_type,
            sim_th=self.similarity_threshold,
            sim_tw=self.similarity_tversky_weights,
            le=self.ligand_efficiency,
            le_ths=self.le_thresholds,
            lipe=self.lipophilic_efficiency,
            lipe_ths=self.lipe_thresholds,
        )

        # Set evolver
        evolver = self.setExplorer(agent, prior, environment)

        # Set monitoring
        monitor = FileMonitor(self.output_file_base, verbose=True)

        # Fit
        log.info('Training generator with reinforcement learning...')
        evolver.fit(train_loader, test_loader, epochs=self.epochs, monitor=monitor, patience=self.patience)

def getModifiers(task, scheme, activity_threshold):
    """
    Gets the modifiers for a given task, scheme and activity threshold combination.

    Arguments:
        task (str): The task to get the modifiers for.
        scheme (str): The scheme to get the modifiers for.
        activity_threshold (float): The activity threshold to use for the modifiers when relevant.

    Returns:
        list: The active modifiers for the given task.
    """
    pad = 3.5
    pad_window = 1.5
    if scheme == 'WS':
        # Weighted Sum (WS) reward scheme
        if task == ModelTasks.CLASSIFICATION:
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold + pad)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold - pad)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)

    else:
        # Pareto Front reward scheme (PRTD or PRCD)
        if task == ModelTasks.CLASSIFICATION:
            active = ClippedScore(lower_x=0.2, upper_x=0.5)
            inactive = ClippedScore(lower_x=0.8, upper_x=0.5)
            window = ClippedScore(lower_x=0, upper_x=1)
        else:
            active = ClippedScore(lower_x=activity_threshold - pad, upper_x=activity_threshold)
            inactive = ClippedScore(lower_x=activity_threshold + pad, upper_x=activity_threshold)
            window = ClippedScore(lower_x=0 - pad_window, upper_x=0 + pad_window)

    return active, inactive, window
    

def CreateEnvironment(
        base_dir, 
        predictor,
        scheme, 
        active_targets=[], 
        inactive_targets=[], 
        window_targets=[],
        activity_threshold=6.5, 
        qed=False, 
        unique=False,
        sa_score=False,
        ra_score=False, 
        mw=False,
        mw_ths=[200,600],
        logP=False,
        logP_ths=[0,5],
        tpsa=False,
        tpsa_ths=[0,140],
        sim_smiles=None,
        sim_type='ECFP6',
        sim_th=0.6,
        sim_tw=[1.,1.],
        le = False,
        le_ths=[0,1],
        lipe = False,
        lipe_ths=[4,6],
        logger=None
        ):

    """
    Sets up the objectives of the desirability function.
    
    Arguments:
        base_dir (str)              : folder containing 'qspr' folder with saved environment-predictor models
        predictor (list)                  : environment-predictor algoritm
        task (str)                  : environment-predictor task: 'REG' or 'CLS'
        scheme (str)                : optimization scheme: 'WS' for weighted sum, 'PRTD' for Pareto ranking with Tanimoto distance or 'PRCD' for Pareto ranking with crowding distance.
        active_targets (lst), opt   : list of active target IDs
        inactive_targets (lst), opt : list of inactive target IDs
        window_targets (lst), opt   : list of target IDs for selectivity window
        activity_threshold (float), opt : activity threshold in case of 'CLS'
        qed (bool), opt             : if True, 'quantitative estimate of drug-likeness' included in the desirability function
        unique (bool), opt          : if Trye, molecule uniqueness in an epoch included in the desirability function
        ra_score (bool), opt        : if True, 'Retrosythesis Accessibility score' included in the desirability function
        mw (bool), opt              : if True, large molecules are penalized in the desirability function
        mw_ths (list), opt          : molecular weight thresholds to penalize large molecules
        logP (bool), opt            : if True, molecules with logP values are penalized in the desirability function
        logP_ths (list), opt        : logP thresholds to penalize large molecules
        tpsa (bool), opt            : if True, tpsa used in the desirability function
        tpsa_ths (list), opt        : tpsa thresholds
        sim_smiles (str), opt       : reference molecules used for similarity calculation
        sim_type (str), opt         : type of fingerprint or 'graph' used for similarity calculation
        sim_th (float), opt         : threshold for similarity desirability  
        sim_tw (list), opt          : tversky similarity weights
        le (bool), opt              : if True, ligand efficiency used instead of activity for active targets in the desirability function
        le_ths (list), opt          : ligand efficiency thresholds
        lipe (bool), opt            : if True, lipophilic efficiency used instead of activity for active targets in the desirability function
        lipe_ths (list), opt        : lipophilic efficiency thresholds
        log (str), opt              : log instance
    
    Returns:
        DrugExEnvironment (torch model) : environment-predictor model
        
    """
    logger = logger or log

    # TODO: update cli documentation/tutorials to reflect new scheme abbreviations
    schemes = {
        "PRTD" : ParetoTanimotoDistance(),
        "PRCD" : ParetoCrowdingDistance(),
        "WS" : WeightedSum()
    }
    objs = []
    ths = []
    targets = active_targets + inactive_targets + window_targets
    predictors = [QSPRModel.fromFile(os.path.join(base_dir, x)) for x in predictor]
    predictors = {x.name: x for x in predictors}
    assert len(predictors) == len(targets) and all([x in targets for x in predictors]), f'Predictors do not match targets: {predictors} != {targets}'
    for target in targets:
        model = predictors[target]
        task = model.task
        if task == ModelTasks.CLASSIFICATION and model.nClasses > 2:
            raise NotImplementedError('Classification models with more than 2 classes are not supported. Invalid model: {}'.format(target))

        active, inactive, window = getModifiers(task, scheme, activity_threshold)
        scorer = QSPRPredScorer(model)
        if target in active_targets:
            scorer.setModifier(active)
            if le or lipe:
                if task == ModelTasks.CLASSIFICATION:
                    raise NotImplementedError('Ligand efficiency and lipophilic efficiency are only available for regression tasks')
                if le:
                    objs.append(LigandEfficiency(qsar_scorer=scorer,
                                modifier=ClippedScore(lower_x=le_ths[0], upper_x=le_ths[1])))
                    ths.append(0.5)
                if lipe:
                    objs.append(LipophilicEfficiency(qsar_scorer=scorer,
                                modifier=ClippedScore(lower_x=lipe_ths[0], upper_x=lipe_ths[1])))
                    ths.append(0.5)
            else:
                objs.append(scorer)
                ths.append(0.5 if scheme == 'WS' else 0.99)
        elif target in inactive_targets:
            scorer.setModifier(inactive)
            objs.append(scorer)
            ths.append(0.5 if scheme == 'WS' else 0.99)
        elif target in window_targets:
            scorer.setModifier(window)
            objs.append(scorer)
            ths.append(0.5)
        else:
            raise ValueError('Target {} not found in active, inactive or window targets'.format(target))

    if qed:
        objs.append(Property('QED', modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.5)
    if unique:
        objs.append(Uniqueness(modifier=ClippedScore(lower_x=1.0, upper_x=0.0)))
        ths.append(0.0)
    if sa_score:
        objs.append(Property('SA', modifier=ClippedScore(lower_x=10, upper_x=1.0)))
        ths.append(0.5)
    if ra_score:
        from drugex.training.scorers.ra_scorer import \
            RetrosyntheticAccessibilityScorer
        objs.append(RetrosyntheticAccessibilityScorer(modifier=ClippedScore(lower_x=0, upper_x=1.0)))
        ths.append(0.0)
    if mw:
        objs.append(Property('MW', modifier=SmoothHump(lower_x=mw_ths[0], upper_x=mw_ths[1], sigma=100)))
        ths.append(0.99)
    if logP:
        objs.append(Property('logP', modifier=SmoothHump(lower_x=logP_ths[0], upper_x=logP_ths[1], sigma=1)))
        ths.append(0.99)
    if tpsa:
        objs.append(Property('TPSA', modifier=SmoothHump(lower_x=tpsa_ths[0], upper_x=tpsa_ths[1], sigma=10)))
        ths.append(0.99)
    if sim_smiles:
        if sim_type == 'fraggle':
            objs.append(FraggleSimilarity(sim_smiles))
        if sim_type == 'graph':
            objs.append(TverskyGraphSimilarity(sim_smiles, sim_tw[0], sim_tw[1]))
        else:
            objs.append(TverskyFingerprintSimilarity(sim_smiles, sim_type, sim_tw[0], sim_tw[1]))
        ths.append(sim_th)

    logger.info('DrugExEnvironment created using {} objectives: {}'.format(len(objs), [o.getKey() for o in objs]))

    return DrugExEnvironment(objs, ths, schemes[scheme])
    
if __name__ == "__main__":
    
    # Parse command line arguments
    args = GeneratorArgParser()

    # Create backup of files
    backup_msg = backUpFiles(args.base_dir, 'generators', (args.output_long,))
    
    # Create directory for generators
    if not os.path.exists(f'{args.base_dir}/generators'):
        os.mkdir(f'{args.base_dir}/generators')

    # Create log file
    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'generators'),
        args.output_long + '.log',
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )
    log = logSettings.log
    log.info(backup_msg)

    # Save arguments to json file
    with open(f'{args.output_file_base}.json', 'w') as f:
        json.dump(vars(args), f)

    # Train generator
    if args.training_mode == 'PT':
        Pretrain(args)()
    elif args.training_mode == 'FT':
        Finetune(args)()
    elif args.training_mode == 'RL':
        Reinforce(args)()
    else:
        raise ValueError(f"--mode should be either 'PT', 'FT' or 'RL', you gave {args.training_mode}")
