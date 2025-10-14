from drugex.training.scorers.interfaces import Scorer
import sys
from openeye import oeomega

import pandas as pd
from rdkit import Chem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

import os
from openeye import oechem
from openeye import oefastrocs
from openeye import oeshape

from drugex.training.scorers.properties import Property
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance

from drugex.training.explorers import SequenceExplorer
import warnings
from drugex.training.generators import SequenceRNN
from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.monitors import FileMonitor



# Scorer
def Isomers(molecules, mol_ids, experiment_name):
    # read in dataset
    smiles_list = []
    # Assuming 'molecules' is your list of molecules and 'smiles_list' is your list to store results
    for mi, m in zip(mol_ids, molecules):
        if m is None:
            continue
          
        isomers = list(EnumerateStereoisomers(m))
        if len(isomers) > 4:
            continue
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(m)
        heavy_atoms = m.GetNumHeavyAtoms()
        
        if rot_bonds is not None and heavy_atoms is not None:
            if rot_bonds > 15 or heavy_atoms > 45:
                continue
    
        for si, smi in enumerate(sorted(Chem.MolToSmiles(x, isomericSmiles=True) for x in isomers)):
            msid = f"{mi}+{si}"
            smiles_list.append((m,smi,msid))
    # create dataframe with correction and new columns
    df_smile = pd.DataFrame(smiles_list, columns=['SMILES','Isomers','CID'])
    # take a sample and isolate Isomers and Index for conformer generation 
    isomers = f"./DrugEx_script/{experiment_name}_isomers_score.smi"
    df_smile[["Isomers", "CID"]].to_csv(isomers, sep='\t', index=False, header = False)
    
    return df_smile, isomers

def OMEGA(isomers, experiment_name):
    dbname = f"./DrugEx_script/{experiment_name}_conformers_score.sdf"
    argv=['-', '-in', isomers, '-out', dbname]
    
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetParameterVisibility(oechem.OEParamVisibility_Hidden)
    omegaOpts.SetParameterVisibility("-rms", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-ewindow", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-maxconfs", oechem.OEParamVisibility_Simple)
    omegaOpts.SetParameterVisibility("-useGPU", oechem.OEParamVisibility_Simple)

    opts = oechem.OESimpleAppOptions(omegaOpts, "Omega", oechem.OEFileStringType_Mol, oechem.OEFileStringType_Mol3D)
    if oechem.OEConfigureOpts(opts, argv, False) == oechem.OEOptsConfigureStatus_Help:
        raise Exception("Configuration Failed")

    omegaOpts.UpdateValues(opts)
    omega = oeomega.OEOmega(omegaOpts)

    ifs = oechem.oemolistream()
    if not ifs.open(opts.GetInFile()):
        oechem.OEThrow.Fatal("Unable to open %s for reading" % opts.GetInFile())

    ofs = oechem.oemolostream()
    if not ofs.open(opts.GetOutFile()):
        oechem.OEThrow.Fatal("Unable to open %s for writing" % opts.GetOutFile())

    for mol in ifs.GetOEMols():
        oechem.OEThrow.Info("Title: %s" % mol.GetTitle())
        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            oechem.OEWriteMolecule(ofs, mol)
        else:
            oechem.OEThrow.Warning("%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code)))

    return dbname

def ROCS(dbname, qfnames, experiment_name):
    
    if not oefastrocs.OEFastROCSIsGPUReady():
        oechem.OEThrow.Info("No supported GPU available!")

    
    if oechem.OEIsGZip(dbname):
        oechem.OEThrow.Fatal("%s is an unsupported database file format as it is gzipped.\n"
                            "Preferred formats are .oeb, .sdf or .oez" % dbname)

    # read in database
    ifs = oechem.oemolistream()
    if not ifs.open(dbname):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    print("Opening database file %s ..." % dbname)
    timer = oechem.OEWallTimer()
    dbase = oefastrocs.OEShapeDatabase()
    moldb = oechem.OEMolDatabase()
    if not moldb.Open(ifs):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    dots = oechem.OEThreadedDots(10000, 200, "conformers")  ## This shows progress of screening, database determines amount of conformers 
    if not dbase.Open(moldb, dots):
        oechem.OEThrow.Fatal("Unable to initialize OEShapeDatabase on '%s'" % dbname)

    dots.Total()
    print("%f seconds to load database" % timer.Elapsed())
    
    for qfname in qfnames: 

        ext = oechem.OEGetFileExtension(qfname)
        base = qfname[:-(len(ext) + 1)]

        if ext == 'sq':                         #adjust input and output extension
            query = oeshape.OEShapeQuery() 
            if not oeshape.OEReadShapeQuery(qfname, query):
                oechem.OEThrow.Fatal("Unable to open '%s'" % qfname)
            ext = 'csv'
        else:
            # read in query
            qfs = oechem.oemolistream()
            if not qfs.open(qfname):
                oechem.OEThrow.Fatal("Unable to open '%s'" % qfname)

            query = oechem.OEGraphMol()
            if not oechem.OEReadMolecule(qfs, query):
                oechem.OEThrow.Fatal("Unable to read query from '%s'" % qfname)

        # write out everthing to a similary named file
        ofs = oechem.oemolostream()
        ofname = f"./DrugEx_script/{experiment_name}_" + base + "_results_scorer." + ext
        if not ofs.open(ofname):
            oechem.OEThrow.Fatal("Unable to open '%s'" %ofname)

        print("Searching for %s" % qfname)
        numHits = moldb.NumMols()
        opts = oefastrocs.OEShapeDatabaseOptions()
        opts.SetLimit(numHits)
        for score in dbase.GetSortedScores(query, opts):
            dbmol = oechem.OEMol()
            molidx = score.GetMolIdx()
            if not moldb.GetMolecule(dbmol, molidx):
                print("Unable to retrieve molecule '%u' from the database" % molidx)
                continue

            mol = oechem.OEGraphMol(dbmol.GetConf(oechem.OEHasConfIdx(score.GetConfIdx())))

            oechem.OESetSDData(mol, "ShapeTanimoto", "%.4f" % score.GetShapeTanimoto())
            oechem.OESetSDData(mol, "ColorTanimoto", "%.4f" % score.GetColorTanimoto())
            oechem.OESetSDData(mol, "TanimotoCombo", "%.4f" % score.GetTanimotoCombo())
            score.Transform(mol)

            oechem.OEWriteMolecule(ofs, mol)
        print("Wrote results to %s" % ofname)

    return ofname

class ModelScorer(Scorer):
    
    def __init__(self, qfnames, score, experiment_name):
        super().__init__()
        self.qfnames = qfnames
        self.score = score
        self.experiment_name = experiment_name
    def getScores(self, mols, frags=None):
        if len(mols) == 0:
            warnings.warn(f"Empty molecule list presented to: {self}")
            return []
        mols = [Chem.MolFromSmiles(m) if isinstance(m, str) else m for m in mols]
        mol_ids = [f"molecule_{x}" for x in range(len(mols))]
        df_smiles, isomers = Isomers(mols, mol_ids, self.experiment_name)
        dbname = OMEGA(isomers, self.experiment_name)
        ofname = ROCS(dbname, self.qfnames, self.experiment_name)
        
        # try:
        data = pd.read_csv(ofname)
        # except Exception as e:
        #     print(f"Error reading data from ROCS output: {e}")
        #     data = pd.DataFrame()  # Initialize an empty DataFrame
        
            # rocs scores multiple isomers per conformer --> group based on Title (ID) and keep the highest score per group 
        
        # Splitting the 'CID' column and expanding into two separate columns
        data[['CID', 'ConformerNo']] = data['TITLE'].str.split('+', expand=True)
        ## compare Isomers second column with data CID 
        # put the missing ones in a new dataframe, add for every row in TanimotoCombo a 0 
        # concat with data
        missing_data = pd.DataFrame({"CID": sorted(set(mol_ids) - set(data.CID))})
        missing_data["ConformerNo"] = 0
        missing_data["TITLE"] = missing_data.CID.apply(lambda x: f"{x}+0")
        data = pd.concat([data, missing_data])
        data.fillna(0, inplace=True)
        data["MolIdx"] = data.CID.apply(lambda x: int(x.split("_")[1]))
        
        
        # split on +, store first value ---> group on stored value 
        grouped_max = data.groupby('CID', as_index=False).max()
        grouped_max.sort_values("MolIdx", inplace=True)

        return grouped_max[self.score].tolist()
        
    def getKey(self):
            """
            Unique Identifier among all the scoring functions used in a single environment.
            """
        
            return "ROCS"
       




