import os
import time

import pandas as pd
from rdkit import Chem

from drugex.corpus.corpus import SequenceCorpus
from drugex.datasets.processing import Standardization, MoleculeEncoder, FragmentEncoder, GraphFragDataCollector, \
    SmilesFragDataCollector, SmilesDataCollector
from drugex.logs.utils import enable_file_logger, commit_hash
from drugex.datasets.fragments import FragmentPairsSplitter, SequenceFragmentEncoder, \
    GraphFragmentEncoder
from drugex.molecules.converters.fragmenters import Fragmenter
from drugex.molecules.files.suppliers import SDFSupplier
from drugex.corpus.vocabulary import VocSmiles, VocGraph

import argparse
import json

def load_molecules(base_dir, input_file):
    """
    Loads raw SMILES from input file and transform to rdkit molecule
    Arguments:
        base_dir (str)            : base directory, needs to contain a folder data with input file
        input_file  (str)         : file containing SMILES, can be 'sdf.gz' or (compressed) 'tsv' or 'csv' file
    Returns:
        mols (lst)                : list of SMILES extracted from input_file
    """
    
    print('Loading molecules...')
    
    file_path = base_dir + '/data/' + input_file

    if input_file.endswith('.sdf.gz') or input_file.endswith('.sdf'):
        # TODO: could be parallel as well
        mols = SDFSupplier(file_path, hide_duplicates=True)
        mols = [x.smiles for x in mols.toList()]
    else:
        df = pd.read_csv(file_path, sep="\t", header=0, na_values=('nan', 'NA', 'NaN', '')).dropna(subset=[args.molecule_column])
        mols = df[args.molecule_column].tolist()
        
    return mols

# def graph_corpus(input, output, suffix='sdf'):
#     metals = {'Na', 'Zn', 'Li', 'K', 'Ca', 'Mg', 'Ag', 'Cs', 'Ra', 'Rb', 'Al', 'Sr', 'Ba', 'Bi'}
#     voc = VocGraph('data/voc_graph.txt')
#     inf = gzip.open(input)
#     if suffix == 'sdf':
#         mols = Chem.ForwardSDMolSupplier(inf)
#         total = 2e6
#     else:
#         mols = pd.read_table(input).drop_duplicates(subset=['Smiles']).dropna(subset=['Smiles'])
#         total = len(mols)
#         mols = mols.iterrows()
#     vals = {}
#     exps = {}
#     codes, ids = [], []
#     chooser = rdMolStandardize.LargestFragmentChooser()
#     disconnector = rdMolStandardize.MetalDisconnector()
#     normalizer = rdMolStandardize.Normalizer()
#     for i, mol in enumerate(tqdm(mols, total=total)):
#         if mol is None: continue
#         if suffix != 'sdf':
#             idx = mol[1]['Molecule ChEMBL ID']

#             mol = Chem.MolFromSmiles(mol[1].Smiles)
#         else:
#             idx = mol.GetPropsAsDict()
#             idx = idx['chembl_id']
#         try:
#             mol = disconnector.Disconnect(mol)
#             mol = normalizer.normalize(mol)
#             mol = chooser.choose(mol)
#             mol = disconnector.Disconnect(mol)
#             mol = normalizer.normalize(mol)
#         except:
#             print(idx)
#         symb = [a.GetSymbol() for a in mol.GetAtoms()]
#         # Nr. of the atoms
#         bonds = mol.GetBonds()
#         if len(bonds) < 4 or len(bonds) >= 63: continue
#         if {'C'}.isdisjoint(symb): continue
#         if not metals.isdisjoint(symb): continue

#         smile = Chem.MolToSmiles(mol)
#         try:
#             s0 = smile.replace('[O]', 'O').replace('[C]', 'C') \
#                  .replace('[N]', 'N').replace('[B]', 'B') \
#                  .replace('[2H]', '[H]').replace('[3H]', '[H]')
#             s0 = Chem.CanonSmiles(s0, 0)
#             code = voc.encode([smile])
#             s1 = voc.decode(code)[0]
#             assert s0 == s1
#             codes.append(code[0].reshape(-1).tolist())
#             ids.append(idx)
#         except Exception as ex:
#             print(ex)
#             print('Parse Error:', idx)
#     df = pd.DataFrame(codes, index=ids, columns=['C%d' % i for i in range(64*4)])
#     df.to_csv(output, sep='\t', index=True)
#     print(vals)
#     print(exps)

def graph_encode(base_dir, smiles, output_file):
    """
    Encodes fragments and molecules to graph-matrices.
    Arguments:
        df (pd.DataFrame)         : dataframe containing molecules
        file_base (str)           : base of output file
    """

    print('Encoding molecules to graph-matrices...')
    voc = VocGraph()

    # create columns for fragments
    col = ['C%d' % d for d in range(voc.max_len*5)]
    codes = []
    large = max(smiles, key=len)
    smiles.remove(large)
    mol = Chem.MolFromSmiles(large)
    total = mol.GetNumBonds()
    if total >= 75:
        raise ValueError("To create dataset largest smiles has to have less than 75 bonds'")

    for smile in smiles:
        output = voc.encode([large], [smile])
        f, s = voc.decode(output)
        assert large == s[0]
        code = output[0].reshape(-1).tolist()
        codes.append(code)

    codes = pd.DataFrame(codes, columns=col)
    codes.to_csv('%s/data/%s_graph.txt' % (base_dir, output_file), sep='\t', index=False)
    
def DatasetArgParser(txt=None):
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Base directory which contains a folder 'data' with input files")
    parser.add_argument('-k', '--keep_runid', action='store_true', help="If included, continue from last run")
    parser.add_argument('-p', '--pick_runid', type=int, default=None, help="Used to specify a specific run id")
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--input', type=str, default='LIGAND_RAW.tsv',
                        help="Input file containing raw data. tsv or sdf.gz format")   
    parser.add_argument('-o', '--output', type=str, default='ligand',
                        help="Prefix of output files")
    parser.add_argument('-mt', '--mol_type', type=str, default='smiles',
                        help="Type of molecular representation: 'graph' or 'smiles'")     
    parser.add_argument('-nof', '--no_frags', action='store_true',
                        help="If on, molecules are not split to fragments and a corpus is created")
    
    parser.add_argument('-fm', '--frag_method', type=str, default='brics',
                        help="Fragmentation method: 'brics' or 'recap'") 
    parser.add_argument('-nf', '--n_frags', type=int, default=4,
                        help="Number of largest leaf-fragments used per compound")
    parser.add_argument('-nc', '--n_combs', type=int, default=None,
                        help="Maximum number of leaf-fragments that are combined for each fragment-combinations. If None, default is {n_frags}")
    parser.add_argument('-np', '--n_proc', type=int, default=None,
                        help="Number of parallel processes to use for multi-core tasks. If not specified, this number is set to the number of available CPUs on the system.")
    parser.add_argument('-vf', '--voc_file', type=str, default='voc',
                        help="Name for voc file, used to save voc tokens")
    parser.add_argument('-mc', '--molecule_column', type=str, default='SMILES',
                        help="Name of the column in CSV files that contains molecules.")
    parser.add_argument('-sv', '--save_voc', action='store_true',
                        help="If on, save voc file (should only be done for the pretraining set). Currently only works is --mol_type is 'smiles'.")   
    parser.add_argument('-sif', '--save_intermediate_files', action='store_true',
                        help="If on, intermediate files")
    parser.add_argument('-nfs', '--no_fragment_split', action='store_true',
                        help="If on, split fragment data sets to training, test and unique sets.")
    parser.add_argument('-ng', '--no_git', action='store_true',
                        help="If on, git hash is not retrieved")
    
    if txt:
        args = parser.parse_args(txt)
    else:
        args = parser.parse_args()
        
    if args.n_combs is None:
        args.n_combs = args.n_frags
        
    return args

def save_encoded_data(collectors, file_base, mol_type, save_voc, voc_file, runid):
    for collector in collectors:
        collector.save()

    vocs = [x.getVoc() for x in collectors]
    voc = sum(vocs[1:], start=vocs[0])
    if save_voc:
        voc.toFile(os.path.join(file_base, f'{voc_file}_{mol_type}_{runid}.txt'))

def Dataset(args):
    """ 
    Prepare input files for DrugEx generators containing encoded molecules for three different cases:
    
    - SMILES w/o fragments: {output}_corpus.txt and [opt] {voc}_smiles.txt containing the SMILES-token-encoded molecules
                             and the token-vocabulary respectively
    - SMILES w/ fragments: {output}_{mf/sf}_{frag_method}_[train/test]_smi.txt and [opt] {voc}_smiles.txt containing
                             the SMILES-token-encoded fragment-molecule pairs for the train and test sets and 
                             the token-vocabulary respectively
    - Graph fragments: {output}_{mf/sf}_{frag_method}_[train/test]_graph.txt and [opt] {voc}_graph.txt containing the
                             encoded graph-matrices of fragement-molecule pairs for the train and test sets and
                             the token-vocabulary respectively   
    """
    
                        
    # load molecules
    tm_start = time.perf_counter()
    print('Dataset started. Loading molecules...')
    smiles = load_molecules(args.base_dir, args.input)

    print("Standardizing molecules...")
    standardizer = Standardization(n_proc=args.n_proc)
    smiles = standardizer.applyTo(smiles)

    file_base = os.path.join(args.base_dir, 'data')
    if args.no_frags:
        if args.mol_type == 'graph':
            graph_encode(args.base_dir, smiles, args.output)
            #raise ValueError("To apply --no_frags, --mol_type needs to be 'smiles'")
        # create corpus (only used in v2), vocab (only used in v2)  
        print('Creating the corpus...')

        encoder = MoleculeEncoder(
            SequenceCorpus,
            {
                'vocabulary': VocSmiles()
            },
            n_proc=args.n_proc
        )
        data_collector = SmilesDataCollector(os.path.join(file_base, f'{args.output}_corpus_{logSettings.runID}.txt'))
        encoder.applyTo(smiles, collector=data_collector)

        save_encoded_data([data_collector], file_base, args.mol_type, args.save_voc, args.voc_file, logSettings.runID)
    else:
        # create encoded fragment-molecule pair files for train and test set (only v3 models)
        file_name = f'%s_%d:%d_%s' % (args.output, args.n_frags, args.n_combs, args.frag_method)
        file_prefix = os.path.join(file_base, file_name)

        if args.n_combs > 1 :
            print('Breaking molecules to leaf fragments, making combinations and encoding...')
        else:
            print('Breaking molecules to leaf fragments and encoding...')

        # prepare splitter and collect intermediate files if required
        pair_collectors = dict()
        if args.save_intermediate_files:
            pair_collectors['train_collector'] = lambda x : x.to_csv(file_prefix + f'_train_{logSettings.runID}.txt', sep='\t', index=False)
            pair_collectors['test_collector'] = lambda x : x.to_csv(file_prefix + f'_test_{logSettings.runID}.txt', sep='\t', index=False)
            pair_collectors['unique_collector'] = lambda x : x.to_csv(file_prefix + f'_unique_{logSettings.runID}.txt', sep='\t', index=False)
        splitter = FragmentPairsSplitter(0.1, 1e4, **pair_collectors) if not args.no_fragment_split else None

        if args.mol_type == 'graph':
            encoder = FragmentEncoder(
                fragmenter=Fragmenter(args.n_frags, args.n_combs, args.frag_method),
                encoder=GraphFragmentEncoder(
                    VocGraph(n_frags=args.n_frags)
                ),
                pairs_splitter=splitter,
                n_proc=args.n_proc
            )

            data_collectors = [GraphFragDataCollector(file_prefix + f'_{split}' + '_graph_%s.txt' % logSettings.runID) for split in ('test', 'train', 'unique')] if splitter else [GraphFragDataCollector(file_prefix + f'_train' + '_graph_%s.txt' % logSettings.runID)]
            encoder.applyTo(smiles, encodingCollectors=data_collectors)

            save_encoded_data(data_collectors, file_base, args.mol_type, args.save_voc, args.voc_file, logSettings.runID)
        elif args.mol_type == 'smiles':
            data_collectors = [SmilesFragDataCollector(file_prefix + f'_{split}' + '_smi_%s.txt' % logSettings.runID) for split in ('test', 'train', 'unique')
            ] if splitter else [SmilesFragDataCollector(file_prefix + f'_train' + '_smi_%s.txt' % logSettings.runID)]
            encoder = FragmentEncoder(
                fragmenter=Fragmenter(args.n_frags, args.n_combs, args.frag_method),
                encoder=SequenceFragmentEncoder(
                    VocSmiles()
                ),
                pairs_splitter=splitter,
                n_proc=args.n_proc
            )
            encoder.applyTo(smiles, encodingCollectors=data_collectors)

            save_encoded_data(data_collectors, file_base, args.mol_type, args.save_voc, args.voc_file, logSettings.runID)
        else:
            raise ValueError("--mol_type should either 'smiles' or 'graph', you gave '{}' ".format(args.mol_type))

    tm_finish = time.perf_counter()

    print(f"Dataset finished. Execution time: {tm_finish - tm_start:0.4f} seconds")

if __name__ == '__main__':

    args = DatasetArgParser()

    # enable logger and get logSettings
    logSettings = enable_file_logger(
        os.path.join(args.base_dir,'logs'),
        'dataset.log',
        args.keep_runid,
        args.pick_runid,
        args.debug,
        __name__,
        commit_hash(os.path.dirname(os.path.realpath(__file__))) if not args.no_git else None,
        vars(args)
    )

    # Create json log file with used commandline arguments 
    print(json.dumps(vars(args), sort_keys=False, indent=2))
    with open('%s/logs/%s/data_args.json' % (args.base_dir, logSettings.runID), 'w') as f:
        json.dump(vars(args), f)

    Dataset(args)