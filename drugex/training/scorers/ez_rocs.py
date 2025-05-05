# (C) 2022 Cadence Design Systems, Inc. (Cadence) 
# All rights reserved.
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of Cadence products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. Cadence claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable Cadence offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall Cadence be
# liable for any damages or liability in connection with the Sample Code
# or its use.



import numpy as np
import pandas as pd
import os
import tempfile
from openeye import oechem, oeomega
try:
    from openeye import oeshape, oefastrocs
    FASTROCS_AVAILABLE = True
except ImportError:
    FASTROCS_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer

def OMEGA(input_file, experiment_name):
    """
    Generate conformers using OMEGA.
    
    Parameters
    ----------
    input_file : str
        Path to the input file with molecules.
    experiment_name : str
        Name of the experiment for output file naming.
        
    Returns
    -------
    str
        Path to the generated conformer database.
    """
    dbname = f"./{experiment_name}_conformers.oeb.gz"
    
    # Set up OMEGA options
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(1)  # Only need 1 conformer for shape screening
    omegaOpts.SetStrictStereo(False)  # Don't enforce stereo constraints
    
    # Create omega
    omega = oeomega.OEOmega(omegaOpts)
    
    # Read molecules from input file
    ifs = oechem.oemolistream()
    if not ifs.open(input_file):
        raise ValueError(f"Cannot open input file: {input_file}")
        
    # Create output file
    ofs = oechem.oemolostream()
    if not ofs.open(dbname):
        raise ValueError(f"Cannot create output file: {dbname}")
    
    # Generate conformers - fixed to correctly handle OEMol objects for omega
    for mol_iter in ifs.GetOEMols():
        # Create a copy of the molecule for conformer generation
        mol = oechem.OEMol(mol_iter)
        if omega(mol):  # Correct calling pattern for omega
            oechem.OEWriteMolecule(ofs, mol)
    
    ifs.close()
    ofs.close()
    
    return dbname

def ROCS(dbname, query_files, experiment_name, use_gpu):
    """
    Run FastROCS shape comparison on a database of molecules.
    
    Parameters
    ----------
    dbname : str
        Path to the conformer database.
    query_files : list
        List of query file paths.
    experiment_name : str
        Name of the experiment for output file naming.
    use_gpu : bool
        Whether to use GPU acceleration if available.
        
    Returns
    -------
    str
        Path to the output CSV file with results.
    """
    outfname = f"./{experiment_name}_results.csv"
    
    # Create molecule database
    mdb = oechem.OEMolDatabase()
    if not mdb.Open(dbname):
        raise ValueError(f"Cannot open database: {dbname}")
    
    # Create output file
    ofs = oechem.oemolostream()
    if not ofs.open(outfname):
        raise ValueError(f"Cannot create output file: {outfname}")
    ofs.close()
    
    # Set up shape database
    db = oefastrocs.OEShapeDatabase()
    if use_gpu:
        db.SetNumOpenThreads(1)
        opts = oefastrocs.OEShapeDatabaseOptions()
        opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
    else:
        db.SetNumOpenThreads(max(1, os.cpu_count() or 2))
        opts = oefastrocs.OEShapeDatabaseOptions()
        opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
    
    # Open the database with the molecule database
    if not db.Open(mdb):
        raise ValueError(f"Failed to open shape database")
    
    # Process each query file
    for qfname in query_files:
        # Read query
        query = oeshape.OEShapeQuery()
        if not oeshape.OEReadShapeQuery(qfname, query):
            raise ValueError(f"Cannot read query file: {qfname}")
        
        # Get scores and write to file
        with open(outfname, 'w') as f:
            # Write header
            f.write("TITLE,TanimotoCombo,ShapeTanimoto,ColorTanimoto\n")
            
            # Process each score
            for score in db.GetSortedScores(query, opts):
                mol_idx = score.GetMolIdx()
                mol = oechem.OEGraphMol()
                
                if mdb.GetMolecule(mol, mol_idx):
                    title = mol.GetTitle()
                    tanimoto_combo = score.GetTanimotoCombo()
                    shape_tanimoto = score.GetShapeTanimoto()
                    color_tanimoto = score.GetColorTanimoto()
                    
                    # Write score to file
                    f.write(f"{title},{tanimoto_combo},{shape_tanimoto},{color_tanimoto}\n")
    
    return outfname

class RocsScorer(Scorer):
    """
    A minimal viable scorer for ROCS that computes shape similarity scores 
    for a list of molecules against a query molecule using OEFlipper for isomer enumeration.
    """
    
    def __init__(self, sq_model_path=None, query_file=None, experiment_name="rocs_experiment", 
                 score_type="TanimotoCombo", use_gpu=False, max_isomers=4, max_rot_bonds=10, 
                 max_heavy_atoms=30, cpu_processes=1):
        """
        Initialize the ROCS Scorer.

        Parameters
        ----------
        sq_model_path : str, optional
            Path to the ROCS query file (e.g., .sq file). Alternative to query_file.
        query_file : str, optional
            Path to the ROCS query file (e.g., .sq or molecule file). Alternative to sq_model_path.
        experiment_name : str, optional
            Name of the experiment for file naming (default: "rocs_experiment").
        score_type : str, optional
            Type of score to extract (e.g., "ShapeTanimoto", "ColorTanimoto", "TanimotoCombo").
        use_gpu : bool, optional
            Whether to use GPU acceleration if available (default: False).
        max_isomers : int, optional
            Maximum number of isomers to enumerate per molecule (default: 4).
        max_rot_bonds : int, optional
            Maximum number of rotatable bonds to consider (default: 10).
        max_heavy_atoms : int, optional
            Maximum number of heavy atoms to process (default: 30).
        cpu_processes : int, optional
            Number of CPU processes to use if not using GPU (default: 1).
        """
        super().__init__()
        # Handle both parameter options for the query file
        self.query_file = sq_model_path if sq_model_path is not None else query_file
        self.experiment_name = experiment_name
        self.score_type = score_type
        self.qfnames = [self.query_file]  # Single query for minimal implementation
        
        # Store additional parameters
        self.use_gpu = use_gpu and FASTROCS_AVAILABLE
        self.max_isomers = max_isomers
        self.max_rot_bonds = max_rot_bonds
        self.max_heavy_atoms = max_heavy_atoms
        self.cpu_processes = cpu_processes
        
        # Check that the query file exists
        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")

    def getScores(self, mols, frags=None):
        """
        Compute ROCS similarity scores for a list of molecules.

        Parameters
        ----------
        mols : List[str] or List[RDKit.Mol]
            A list of SMILES strings or RDKit molecule objects representing molecules.
        frags : List[str], optional
            A list of fragments (not used in this scorer).

        Returns
        -------
        scores : np.ndarray
            An array of similarity scores for the input molecules.
        """
        if not mols:
            return np.array([])

        # Check if input contains RDKit molecules and convert to SMILES if needed
        import_rdkit = False
        for mol in mols:
            if mol is not None and not isinstance(mol, str):
                import_rdkit = True
                break
                
        if import_rdkit:
            from rdkit import Chem
            smiles = []
            for mol in mols:
                if mol is None:
                    smiles.append(None)
                else:
                    try:
                        smi = Chem.MolToSmiles(mol)
                        smiles.append(smi)
                    except:
                        smiles.append(None)
            mols = smiles

        mol_ids = [f"molecule_{i}" for i in range(len(mols))]

        # Generate isomers using OEFlipper, adapted from the complex code
        isomers_list = []
        flipper_opts = oeomega.OEFlipperOptions()
        flipper_opts.SetMaxCenters(min(4, self.max_isomers))  # Use parameter for max centers
        
        for mi, smi in zip(mol_ids, mols):
            # Skip missing or empty entries
            if not smi:
                continue

            # Make sure it's a Python str, not numpy.str_ or bytes
            smi_str = str(smi)

            mol = oechem.OEMol()
            # Now pass in a real str, so the C++ wrapper can convert it
            if not oechem.OESmilesToMol(mol, smi_str):
                continue
                
            # Skip molecules that exceed the max heavy atom limit
            if mol.NumAtoms() > self.max_heavy_atoms:
                continue
                
            mol.SetTitle(mi)  # Set title for tracking
            isomer_count = 0
            for iso in oeomega.OEFlipper(mol, flipper_opts):
                if isomer_count >= self.max_isomers:  # Use parameter for max isomers
                    break
                iso_smi = oechem.OEMolToSmiles(iso)
                msid = f"{mi}+{isomer_count}"
                isomers_list.append((iso_smi, msid))
                isomer_count += 1

        if not isomers_list:
            return np.zeros(len(mols))

        # Create temporary files with unique names
        with tempfile.NamedTemporaryFile(suffix='.smi', delete=False) as tmp_file:
            isomers_file = tmp_file.name
            
        df_isomers = pd.DataFrame(isomers_list, columns=['Isomers', 'CID'])
        df_isomers.to_csv(isomers_file, sep='\t', index=False, header=False)

        try:
            # Generate conformers using OMEGA
            dbname = OMEGA(isomers_file, self.experiment_name)

            # Run ROCS
            ofname = ROCS(dbname, self.qfnames, self.experiment_name, self.use_gpu)

            # Process the output CSV
            data = pd.read_csv(ofname)
            data['MolID'] = data['TITLE'].apply(lambda x: x.split('+')[0])

            # Handle missing molecules
            all_mol_ids = set(mol_ids)
            present_mol_ids = set(data['MolID'])
            missing_mol_ids = all_mol_ids - present_mol_ids
            if missing_mol_ids:
                missing_data = pd.DataFrame({
                    "MolID": list(missing_mol_ids),
                    self.score_type: 0.0
                })
                data = pd.concat([data, missing_data])

            # Group by MolID and take max score
            grouped_max = data.groupby('MolID', as_index=False)[self.score_type].max()
            grouped_max['MolIdx'] = grouped_max['MolID'].apply(lambda x: int(x.split('_')[1]))
            grouped_max.sort_values('MolIdx', inplace=True)
            scores = grouped_max[self.score_type].tolist()
            
            # Clean up temporary files
            for file in [isomers_file, dbname, ofname]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
                        
            return np.array(scores)
            
        except Exception as e:
            # Clean up temporary files in case of error
            for file in [isomers_file]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
            # Return zeros if something failed
            print(f"Error in ROCS scoring: {e}")
            return np.zeros(len(mols))

    def getKey(self):
        """
        Return the identifier key for this scorer.

        Returns
        -------
        str
            The key identifier "ROCS".
        """
        return "ROCS"
