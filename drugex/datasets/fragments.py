import pandas as pd
from rdkit import Chem

from drugex.corpus.vocabulary import VocSmiles, VocGraph
from drugex.logs import logger
from drugex.datasets.interfaces import TrainTestSplitter, FragmentPairEncoder
from drugex.molecules.converters.interfaces import ConversionException
from drugex.molecules.interfaces import AnnotationException
from drugex.molecules.suppliers import DataFrameSupplier


class FragmentPairsSplitter(TrainTestSplitter):

    def __init__(self, ratio=0.2, max_test_samples=1e4, train_collector=None, test_collector=None, unique_collector=None, frags_col="Frags", smiles_col="Smiles"):
        super().__init__(train_collector, test_collector)
        self.ratio = ratio
        self.maxTestSamples = max_test_samples
        self.uniqueCollect = unique_collector
        self.smilesCol = smiles_col
        self.fragsCol = frags_col

    def __call__(self, pairs):
        df = pd.DataFrame(pairs, columns=[self.fragsCol, self.smilesCol])
        frags = set(df.Frags)
        test_len = len(frags)//(100 * self.ratio)
        if test_len > int(self.maxTestSamples):
            logger.warning(f'To speed up the training, the test set size was automatically capped at {self.maxTestSamples} fragments instead of the default 10% of original data, which would have been: {test_len}.')
            test_in = df.Frags.drop_duplicates().sample(int(self.maxTestSamples))
        else:
            test_in = df.Frags.drop_duplicates().sample(len(frags) // 10)
        test = df[df.Frags.isin(test_in)]
        train = df[~df.Frags.isin(test_in)]
        unique = df.drop_duplicates(subset='Frags')

        if self.trainCollect:
            self.trainCollect(train)
        if self.testCollect:
            self.testCollect(test)
        if self.uniqueCollect:
            self.uniqueCollect(unique)

        return test, train, unique

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
            return ret['mol_encoded'], ret['frag_encoded']
        except KeyError:
            # logger.warning(f"Failed to encode fragment {ret['frag']} for molecule: {ret['mol']}")
            return next(self)

    def annotateMol(self, mol, key, value):
        if key == self.fragsCol:
            mol['frag'] = value
            encoded = self.encoder.encodeFrag(mol['mol'], value)
            if encoded:
                mol['frag_encoded'] = encoded
            else:
                raise self.FragmentEncodingException(f'Failed to encode fragment {value} from molecule: {mol["mol"]}')