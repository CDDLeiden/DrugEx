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
    # Use temporary directory for output files
    temp_dir = tempfile.gettempdir()
    dbname = os.path.join(temp_dir, f"{experiment_name}_conformers.oeb.gz")
    
    # Set up OMEGA options
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(10)  # Increased from 1 for better conformational sampling
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
        try:
            if omega(mol):  # Correct calling pattern for omega
                oechem.OEWriteMolecule(ofs, mol)
        except oechem.OELicenseError as e:
            print(f"OpenEye license error in OMEGA: {e}")
            raise
        except Exception as e:
            print(f"Error generating conformers: {e}")
    
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
    # Use temporary directory for output files
    temp_dir = tempfile.gettempdir()
    outfname = os.path.join(temp_dir, f"{experiment_name}_results.csv")
    
    # Create molecule database
    mdb = oechem.OEMolDatabase()
    if not mdb.Open(dbname):
        raise ValueError(f"Cannot open database: {dbname}")
    
    # Set up shape database
    db = oefastrocs.OEShapeDatabase()
    if use_gpu:
        db.SetNumOpenThreads(1)
        opts = oefastrocs.OEShapeDatabaseOptions()
        opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_FastROCS)
    else:
        # Use multiple threads but keep some cores free for system
        db.SetNumOpenThreads(max(1, min(os.cpu_count() - 2, 4)))
        opts = oefastrocs.OEShapeDatabaseOptions()
        opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
    
    # Open the database with the molecule database
    try:
        if not db.Open(mdb):
            raise ValueError(f"Failed to open shape database")
    except oechem.OELicenseError as e:
        print(f"OpenEye license error: {e}")
        raise
    
    # Create a single output file for all queries
    with open(outfname, 'w') as f:
        # Write header
        f.write("TITLE,QueryFile,TanimotoCombo,ShapeTanimoto,ColorTanimoto\n")
    
    # Process each query file
    for qfname in query_files:
        # Read query
        query = oeshape.OEShapeQuery()
        if not oeshape.OEReadShapeQuery(qfname, query):
            raise ValueError(f"Cannot read query file: {qfname}")
        
        try:
            # Append to existing file instead of overwriting
            with open(outfname, 'a') as f:
                # Process each score
                for score in db.GetSortedScores(query, opts):
                    mol_idx = score.GetMolIdx()
                    mol = oechem.OEGraphMol()
                    
                    if mdb.GetMolecule(mol, mol_idx):
                        title = mol.GetTitle()
                        tanimoto_combo = score.GetTanimotoCombo()
                        shape_tanimoto = score.GetShapeTanimoto()
                        color_tanimoto = score.GetColorTanimoto()
                        
                        # Write score to file with query file name
                        query_name = os.path.basename(qfname)
                        f.write(f"{title},{query_name},{tanimoto_combo},{shape_tanimoto},{color_tanimoto}\n")
        except oechem.OELicenseError as e:
            print(f"OpenEye license error in ROCS scoring: {e}")
            raise
        except Exception as e:
            print(f"Error in ROCS scoring: {e}")
    
    return outfname

class RocsScorer(Scorer):
    """
    A minimal viable scorer for ROCS that computes shape similarity scores 
    for a list of molecules against a query molecule using OEFlipper for isomer enumeration.
    """
    
    def __init__(self, sq_model_path=None, query_file=None, experiment_name="rocs_experiment", 
                 score_type="TanimotoCombo", use_gpu=False, max_isomers=4, max_rot_bonds=10, 
                 max_heavy_atoms=30, cpu_processes=None):
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
            Number of CPU processes to use if not using GPU. If None, will automatically
            determine based on system resources.
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
        
        # Automatically determine CPU processes if not specified
        if cpu_processes is None:
            self.cpu_processes = max(1, min(os.cpu_count() - 2, 4))
        else:
            self.cpu_processes = cpu_processes
            
        # Set up cache directory
        self._cache_dir = os.path.join(tempfile.gettempdir(), "rocs_scorer_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Check that the query file exists and is valid
        if not os.path.exists(self.query_file):
            raise FileNotFoundError(f"Query file not found: {self.query_file}")
            
        # Set up OpenEye
        os.environ["OE_SILENT"] = "true"
        
        # Log configuration
        gpu_str = "GPU" if self.use_gpu else f"CPU ({self.cpu_processes} processes)"
        print(f"ROCS scorer initialized using {gpu_str} mode")
        
        # Validate the query file
        try:
            query = oeshape.OEShapeQuery()
            if not oeshape.OEReadShapeQuery(self.query_file, query):
                raise ValueError(f"Invalid shape query file: {self.query_file}")
        except oechem.OELicenseError as e:
            print(f"OpenEye license error: {e}")
            raise
        except Exception as e:
            print(f"Error validating query file: {e}")
            raise

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

        # Initial filtering to skip invalid molecules
        filtered_mols = []
        mol_ids = []
        
        for i, smi in enumerate(mols):
            # Skip missing or empty entries
            if not smi:
                continue
                
            # Basic filtering for problematic molecules
            if isinstance(smi, str) and len(smi) > 0:
                mol = oechem.OEMol()
                if oechem.OESmilesToMol(mol, str(smi)):
                    # Check heavy atom count
                    if oechem.OECount(mol, oechem.OEIsHeavy()) <= self.max_heavy_atoms:
                        filtered_mols.append((i, smi))
                        mol_ids.append(f"molecule_{i}")
        
        if not filtered_mols:
            return np.zeros(len(mols))

        # Generate isomers using OEFlipper
        isomers_list = []
        flipper_opts = oeomega.OEFlipperOptions()
        flipper_opts.SetMaxCenters(min(4, self.max_isomers))
        
        # Process molecules in batches for memory efficiency
        batch_size = 50
        for batch_start in range(0, len(filtered_mols), batch_size):
            batch_end = min(batch_start + batch_size, len(filtered_mols))
            batch = filtered_mols[batch_start:batch_end]
            
            for orig_idx, smi in batch:
                mol = oechem.OEMol()
                oechem.OESmilesToMol(mol, str(smi))
                mi = f"molecule_{orig_idx}"
                mol.SetTitle(mi)
                
                isomer_count = 0
                try:
                    for iso in oeomega.OEFlipper(mol, flipper_opts):
                        if isomer_count >= self.max_isomers:
                            break
                        iso_smi = oechem.OEMolToSmiles(iso)
                        msid = f"{mi}+{isomer_count}"
                        isomers_list.append((iso_smi, msid))
                        isomer_count += 1
                except Exception as e:
                    print(f"Error generating isomer for {smi}: {e}")

        if not isomers_list:
            return np.zeros(len(mols))

        # Create temporary directory for all files
        temp_dir = tempfile.mkdtemp(prefix="rocs_")
        
        try:
            # Create isomers file
            isomers_file = os.path.join(temp_dir, "isomers.smi")
            with open(isomers_file, 'w') as f:
                for smi, cid in isomers_list:
                    f.write(f"{smi}\t{cid}\n")

            # Generate conformers using OMEGA
            dbname = OMEGA(isomers_file, self.experiment_name)

            # Run ROCS
            ofname = ROCS(dbname, self.qfnames, self.experiment_name, self.use_gpu)

            # Process the output CSV more efficiently
            data = pd.read_csv(ofname)
            
            # Extract original molecule index from title
            data['MolID'] = data['TITLE'].apply(lambda x: x.split('+')[0])
            
            # Create result array with zeros
            result = np.zeros(len(mols))
            
            # More efficient groupby to get max scores
            if self.score_type in data.columns:
                max_scores = data.groupby('MolID')[self.score_type].max()
                
                # Update result array directly without concat
                for mol_id, score in max_scores.items():
                    if mol_id.startswith('molecule_'):
                        idx = int(mol_id.split('_')[1])
                        result[idx] = score
            
            return result
            
        except Exception as e:
            print(f"Error in ROCS scoring: {e}")
            return np.zeros(len(mols))
            
        finally:
            # Clean up all temporary files
            for file_path in [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")
            
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                print(f"Error removing directory {temp_dir}: {e}")
                
            # Force cleanup
            import gc
            gc.collect()

    def getKey(self):
        """
        Return the identifier key for this scorer.

        Returns
        -------
        str
            The key identifier "ROCS".
        """
        return "ROCS"
