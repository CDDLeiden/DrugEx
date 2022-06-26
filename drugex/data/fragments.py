import pandas as pd
from rdkit import Chem

from drugex.data.corpus.vocabulary import VocSmiles, VocGraph
from drugex.logs import logger
from drugex.data.interfaces import DataSplitter, FragmentPairEncoder
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.interfaces import AnnotationException, MolSupplier
from drugex.molecules.suppliers import DataFrameSupplier
from drugex.parallel.evaluator import ParallelSupplierEvaluator
from drugex.parallel.interfaces import ParallelProcessor

class SequenceFragmentEncoder(FragmentPairEncoder):

    def __init__(self, vocabulary=VocSmiles()):
        self.vocabulary = vocabulary

    def encodeMol(self, sequence):
        return self.vocabulary.addWordsFromSeq(sequence)

    def encodeFrag(self, mol, frag):
        return self.vocabulary.addWordsFromSeq(frag, ignoreConstraints=True)

    def getVoc(self):
        return self.vocabulary

class GraphFragmentEncoder(FragmentPairEncoder):

    def __init__(self, vocabulary=VocGraph()):
        self.vocabulary = vocabulary

    def encodeMol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        total = mol.GetNumBonds()
        if total >= 75:
            return None
        else:
            return smiles

    def encodeFrag(self, mol, frag):
        if mol == frag:
            return None
        try:
            output = self.vocabulary.encode([mol], [frag])
            f, s = self.vocabulary.decode(output)

            assert mol == s[0]
            #assert f == frag[0]
            code = output[0].reshape(-1).tolist()
            return code
        except Exception as exp:
            logger.warn(f'The following exception occured while encoding fragment {frag} for molecule {mol}: {exp}')
            return None

    def getVoc(self):
        return self.vocabulary

class FragmentPairsEncodedSupplier(DataFrameSupplier):

    class FragmentEncodingException(AnnotationException):
        pass

    class MoleculeEncodingException(ConversionException):
        pass

    def conversion(self, seq):
        encoded = self.encoder.encodeMol(seq)
        if not encoded:
            raise self.MoleculeEncodingException(f'Failed to encode molecule: {seq}')

        return {
            'mol' : seq,
            'mol_encoded' : encoded
        }

    def __init__(self, df_pairs, encoder, mol_col='Smiles', frags_col='Frags'):
        super().__init__(
            df_pairs,
            mol_col,
            extra_cols=(frags_col,),
            converter=self.conversion)
        self.encoder = encoder
        self.fragsCol = frags_col

    def __next__(self):
        ret = super().__next__()
        try:
            if len(ret) == 2:
                return ret
            return ret['mol_encoded'], ret['frag_encoded']
        except KeyError:
            logger.warning(f"Failed to encode fragment {ret['frag']} for molecule: {ret['mol']}")
            return next(self)

    def annotateMol(self, mol, key, value):
        if key == self.fragsCol:
            mol['frag'] = value
            encoded = self.encoder.encodeFrag(mol['mol'], value)
            if encoded:
                mol['frag_encoded'] = encoded
            else:
                raise self.FragmentEncodingException(f'Failed to encode fragment {value} from molecule: {mol["mol"]}')


class FragmentPairsSupplier(MolSupplier):

    def __init__(self, molecules, fragmenter):
        self.molecules = molecules if hasattr(molecules, "__next__") else iter(molecules)
        self.fragmenter = fragmenter

    def next(self):
        ret = self.fragmenter(next(self.molecules))
        if ret:
            return ret
        else:
            return None

    def convertMol(self, representation):
        return representation

    def annotateMol(self, mol, key, value):
        return mol


class FragmentCorpusEncoder(ParallelProcessor):

    def __init__(self, fragmenter, encoder, pairs_splitter=None, n_proc=None, chunk_size=None):
        super().__init__(n_proc, chunk_size)
        self.fragmenter = fragmenter
        self.encoder = encoder
        self.pairsSplitter = pairs_splitter if pairs_splitter else FragmentPairsSplitter()

    def getFragmentPairs(self, mols, collector):
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsSupplier,
            kwargs={
                "fragmenter" : self.fragmenter
            },
            return_unique=False,
            always_return=True,
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            result_collector=collector
        )
        results = []
        for result in evaluator.apply(mols):
            results.extend(result)
        return results

    def splitFragmentPairs(self, pairs):
        return self.pairsSplitter(pairs)

    def encodeFragments(self, pairs, collector):
        evaluator = ParallelSupplierEvaluator(
            FragmentPairsEncodedSupplier,
            kwargs={
                'encoder': self.encoder,
                'mol_col' : self.pairsSplitter.molCol,
                'frags_col': self.pairsSplitter.fragsCol
            },
            return_unique=False,
            return_suppliers=True,
            chunk_size=self.chunkSize,
            chunks=self.chunks,
            result_collector=collector
        )
        results = evaluator.apply(pairs)
        if results:
            voc = None
            data = []
            for result in results:
                data.extend(result[0])
                if not voc:
                    voc = result[1].encoder.getVoc()
                else:
                    voc += result[1].encoder.getVoc()

            return data, voc


    def apply(self, mols, fragmentCollector=None, encodingCollectors=None):
        pairs = self.getFragmentPairs(mols, fragmentCollector)
        ret = []
        ret_voc = None
        splits = self.splitFragmentPairs(pairs)
        if encodingCollectors and len(encodingCollectors) != len(splits):
            raise RuntimeError(f'The number of encoding collectors must match the number of splits: {len(encodingCollectors)} != {len(splits)}')
        for split_idx, split in enumerate(splits):
            result = self.encodeFragments(split, encodingCollectors[split_idx] if encodingCollectors else None)
            if result:
                result, voc = result
                ret.append(result)
                if not ret_voc:
                    ret_voc = voc
                else:
                    ret_voc += voc
        if ret or ret_voc:
            return ret, ret_voc

class FragmentPairsSplitter(DataSplitter):

    def __init__(self, ratio=0.2, max_test_samples=1e4, train_collector=None, test_collector=None, unique_collector=None, frags_col="Frags", mol_col="Smiles", unique_only=False, seed=None):
        self.fragsCol = frags_col
        self.molCol = mol_col
        self.ratio = ratio
        self.maxTestSamples = max_test_samples
        self.uniqueCollect = unique_collector
        self.trainCollect = train_collector
        self.testCollect = test_collector
        self.uniqueOnly = unique_only
        self.seed = seed

    def __call__(self, pairs):
        df = pd.DataFrame(pairs, columns=[self.fragsCol, self.molCol])
        frags = set(df.Frags)
        test_len = int(len(frags) * self.ratio)
        if self.seed:
            test_in = df.Frags.drop_duplicates().sort_values()
        else:
            test_in = df.Frags.drop_duplicates()
        if test_len > int(self.maxTestSamples):
            logger.warning(f'To speed up the training, the test set size was automatically capped at {self.maxTestSamples} fragments instead of the default 10% of original data, which would have been: {test_len}.')
            test_in = test_in.sample(int(self.maxTestSamples), random_state=self.seed)
        else:
            test_in = test_in.sample(test_len, random_state=self.seed)
        test = df[df.Frags.isin(test_in)]
        train = df[~df.Frags.isin(test_in)]
        unique = df.drop_duplicates(subset=self.fragsCol)

        if self.trainCollect:
            self.trainCollect(train)
        if self.testCollect:
            self.testCollect(test)
        if self.uniqueCollect:
            self.uniqueCollect(unique)

        if self.uniqueOnly:
            return test, unique

        return test, train, unique