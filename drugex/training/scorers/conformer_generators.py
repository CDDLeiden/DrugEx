import gc
import os
import logging
import warnings

try:
    from openeye import oechem, oemolprop, oeomega

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
    
import os
import subprocess
from typing import Callable, List

from drugex.training.scorers.interfaces import ConformerGenerator
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                               StereoEnumerationOptions)

try:
    import CDPL.Chem as CDPLChem
    import CDPL.ConfGen as CDPLConfGen
    import CDPL.Base as CDPLBase
    import CDPL.MolProp as CDPLMolProp
    from CDPL.Chem import StereoisomerGenerator
    CDPL_AVAILABLE = True
except ImportError:
    CDPL_AVAILABLE = False


logger = logging.getLogger(__name__)

class OmegaConformerGenerator(ConformerGenerator):
    """Conformer Generator using Omega
    
    Attributes:
        max_conformers (int): max number of conformers to generate
        max_centers (int): maximum number of stereocenters to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than
            max_rotatable_bonds
        use_gpu (bool): whether to use GPU for conformer generation
        show_progress (bool): whether to show progress during conformer generation
    """
    
    def __init__(
        self,
        max_conformers: int = 10,
        max_centers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        filter: oemolprop.OEFilter | int | str | None = None,
        use_gpu: bool = False,
        show_progress: bool = False,
    ):
        """Initialize the conformer generator
        
        Args:
            max_conformers (int): max number of conformers to generate
            max_centers (int): maximum number of stereocenters to enumerate
            max_heavy_atoms (int): drop molecules with more heavy atoms than 
                max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than
                max_rotatable_bonds
            filter (oemolprop.OEFilter | int | str | None): filter to apply
                Either a path to a OEFilter file or an OEFilter object, or 
                a int representing a OEFilter_Type (can be given like:
                    oemolprop.OEFilterType_BlockBuster)
            use_gpu (bool): whether to use GPU for conformer generation
            show_progress (bool): whether to show progress during conformer generation
        """
        if not OE_AVAILABLE:
            raise ImportError("OpenEye toolkits required")
        
        if max_conformers > 200:
            print(
                "Warning: max_conformers > 200 may cause memory issues "
                "Setting to 200."
            )
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers
        self.max_centers = max_centers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.filter = filter
        self.use_gpu = use_gpu
        self.show_progress = show_progress

    def _create_fresh_omega(self):
        """Create a fresh Omega instance with proven parameters"""
        opts = oeomega.OEOmegaOptions()
        # Use conservative conformer limits
        opts.SetMaxConfs(self.max_conformers)
        opts.SetStrictStereo(False)
        opts.SetFromCT(True)
        opts.SetFixRMS(True)
        opts.SetRMSThreshold(0.5)
        opts.SetEnumRing(True)
        opts.SetRotorOffset(False)

        # Force CPU mode to reduce memory pressure
        if self.use_gpu and oeomega.OEOmegaIsGPUReady():
            opts.GetTorDriveOptions().SetUseGPU(True)
            opts.SetSampleHydrogens(False)
        else:
            opts.GetTorDriveOptions().SetUseGPU(False)
            opts.SetSampleHydrogens(True)

        return oeomega.OEOmega(opts)
    
    def _filter_mol(self, smi, mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        # filter based on heavy atoms and rotatable bonds
        if oechem.OECount(mol, oechem.OEIsHeavy()) > self.max_heavy_atoms:
            oechem.OEThrow.Warning(
                f"Skipping {smi} with > {self.max_heavy_atoms} heavy atoms"
            )
            return True

        if oechem.OECount(mol, oechem.OEIsRotor()) > self.max_rotatable_bonds:
            oechem.OEThrow.Warning(
                f"Skipping {smi} with > {self.max_rotatable_bonds} rotatable bonds"
            )
            return True

        if self.filter is not None:
            filter_type = oechem.oeifstream(self.filter) if isinstance(self.filter, str) else self.filter
            filter = oemolprop.OEFilter(filter_type)
            filter.SetMMFFTypeCheck(True)
            if not filter(mol):
                oechem.OEThrow.Warning(f"Skipping {smi} due to: {filter.GetMessage(mol)}")
                return True
        return False

    def _get_isomers(self, mol):
        """Generate isomers for a molecule using OMEGA"""
        opts = oeomega.OEFlipperOptions()
        opts.SetMaxCenters(self.max_centers)
        for conf in oeomega.OEFlipper(mol, opts):
            iso = oechem.OEMol(conf)
            yield iso
            
    def genConformers(self, smiles_list, out_dir) -> str:
        """Generate conformers using Openeye Omega
        
        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for.
            out_dir (str): Path to the output directory for the generated conformers.

        Returns:
            str: Path to the output conformers file (oeb.gz)
        """
        tmp_outfile = os.path.join(out_dir, f"conformers.oeb.gz")
        ofs = oechem.oemolostream()
        if not ofs.open(tmp_outfile):
            oechem.OEThrow.Fatal(
                "Unable to open %s for writing conformers" % tmp_outfile
            )

        omega = self._create_fresh_omega()

        # Progress tracking for conformer generation
        dots = None
        if self.show_progress and len(smiles_list) > 50:
            print("Generating conformers...")
            dots = oechem.OEThreadedDots(100, 50, "molecules")

        for i, smi in enumerate(smiles_list):
            mol = oechem.OEMol()
            title = f"mol_{i}"
            mol.SetTitle(title)

            if smi is None or not oechem.OESmilesToMol(mol, smi):
                continue

            if self._filter_mol(smi, mol):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                iso.SetTitle(f"{title}+{j}")
                ret_code = omega.Build(iso)
                if ret_code == oeomega.OEOmegaReturnCode_Success:
                    oechem.OEWriteMolecule(ofs, iso)
                else:
                    oechem.OEThrow.Warning(
                        "%s: %s %s"
                        % (smi, iso.GetTitle(), oeomega.OEGetOmegaError(ret_code))
                    )

            if dots:
                dots.Update()

        if dots:
            dots.Total()

        ofs.close()
        omega = None
        gc.collect()
        return tmp_outfile
    
    
class SchrodingerConformerGenerator(ConformerGenerator):
    """Conformer generator using Schrodinger's software.
    
    Attributes:
        max_conformers: Maximum number of conformers to generate.
        max_isomers: Maximum number of isomers to generate.
        max_heavy_atoms: Maximum number of heavy atoms allowed.
        max_rotatable_bonds: Maximum number of rotatable bonds allowed.
        reactions: List of reaction functions to apply.
            Should take an rdkit molecule as input and return a modified molecule.
    """
    
    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        reactions: list[Callable] | None = None,
    ):
        if "SCHRODINGER" not in os.environ:
            raise RuntimeError("SCHRODINGER environment variable is not set")

        self.max_conformers = max_conformers
        self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.reactions = reactions

    def _filter_mol(self, mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""
        from rdkit import Chem

        if mol is None:
            return True

        # filter based on heavy atoms and rotatable bonds
        if rdMolDescriptors.CalcNumHeavyAtoms(mol) > self.max_heavy_atoms:
            print(
                f"Skipping {Chem.MolToSmiles(mol)} with > {self.max_heavy_atoms} heavy atoms"
            )
            return True

        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            print(
                f"Skipping {Chem.MolToSmiles(mol)} with > {self.max_rotatable_bonds} rotatable bonds"
            )
            return True

        return False

    def apply_reactions(self, mol):
        """Apply reactions to hydrolyze beta-lactams and DBO scaffolds"""
        if mol is None or self.reactions is None:
            return mol
        
        for reaction in self.reactions:
            mol = reaction(mol)

        return mol

    def genConformers(self, smiles_list, out_dir) -> str:
        """Generate conformers using Schrodinger confgenx

        Returns:
            str: Path to the output SDF file containing generated conformers.
        """
        currwd = os.getcwd()
        os.chdir(out_dir)

        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi is not None]

        # Filter out unwanted molecules
        mols = [mol for mol in mols if not self._filter_mol(mol)]

        # apply reactions to hydrolyze beta-lactams and DBO scaffolds
        mols = [self.apply_reactions(mol) for mol in mols if mol is not None]

        # save rdkit mols to temporary sd file
        tmp_infile = f"{out_dir}/input_mols.sdf"
        with Chem.SDWriter(tmp_infile) as w:
            for i, mol in enumerate(mols):
                mol.SetProp("_Name", f"mol_{i}")
                # get isomers
                opts = StereoEnumerationOptions(
                    tryEmbedding=False,
                    onlyUnassigned=False,
                    rand=0xF00D,
                    maxIsomers=self.max_isomers,
                )
                for j, iso in enumerate(EnumerateStereoisomers(mol, opts)):
                    iso.SetProp("_Name", f"mol_{i}+{j}")
                    if iso is not None:
                        iso = Chem.AddHs(iso)
                    if iso is not None:
                        w.write(iso)

        # run confgenx
        confgenx_cmd = [
            f"{os.environ['SCHRODINGER']}/confgenx",
            tmp_infile,
            # "-NSTRUCTS",
            # "50",  # Number of jobs to run in parallel
            # "-WAIT",
            "-NOJOBID",
            "-m",
            str(self.max_conformers),
        ]
        print("Running confgenx with command:", " ".join(confgenx_cmd))
        subprocess.run(confgenx_cmd, check=True)

        # multiple processed does not seem to work due to setting maxjobs localhost
        # wait for confgenx to finish
        # wait_cmd = [
        #     f"{os.environ['SCHRODINGER']}/jobcontrol",
        #     "-wait",
        #     "active",
        # ]
        # print("Waiting for confgenx to finish with command:", " ".join(wait_cmd))
        # subprocess.run(wait_cmd, check=True)

        # convert maegz file to sdf
        tmp_outfile = f"{out_dir}/output_conformers.sdf"
        sdconvert_cmd = [
            f"{os.environ['SCHRODINGER']}/utilities/sdconvert",
            "-imae",
            f"{os.path.basename(tmp_infile).removesuffix('.sdf')}-out.maegz",
            "-osd",
            tmp_outfile,
        ]
        subprocess.run(sdconvert_cmd, check=True)

        # read in sdf file, and add 0.01 to all coordinates
        # this prevents issues with ROCS if all coordinates of a dimension are 0
        # this happened with 8SKP of which the generated conformer is planar and had
        # all z-coordinates as 0
        suppl = Chem.SDMolSupplier(tmp_outfile, removeHs=False)
        corrected_mols = []
        for mol in suppl:
            if mol is not None:
                for atom in mol.GetAtoms():
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    new_pos = (pos.x, pos.y, pos.z + 0.01)
                    mol.GetConformer().SetAtomPosition(atom.GetIdx(), new_pos)
                corrected_mols.append(mol)

        # write modified sdf file
        with Chem.SDWriter(tmp_outfile) as w:
            for mol in corrected_mols:
                if mol is not None:
                    w.write(mol)

        os.chdir(currwd)
        return tmp_outfile


class RDKitConformerGenerator(ConformerGenerator):
    """Conformer Generator using RDKit ETKDGv3

    Attributes:
        max_conformers (int): max number of conformers to generate
        max_isomers (int): maximum number of stereoisomers to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than
            max_rotatable_bonds
        show_progress (bool): whether to show progress during conformer generation
    """
    
    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_centers: int | None = None,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        show_progress: bool = False,
    ):
        """Initialize the conformer generator

        Args:
            max_conformers (int): max number of conformers to generate
            max_isomers (int): maximum number of stereoisomers to enumerate
            max_centers (int, optional): deprecated alias for ``max_isomers``
            max_heavy_atoms (int): drop molecules with more heavy atoms than
                max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than
                max_rotatable_bonds
            show_progress (bool): whether to show progress during conformer generation
        """
        if max_conformers > 200:
            print(
                "Warning: max_conformers > 200 may cause memory issues "
                "Setting to 200."
            )
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers
        if max_centers is not None:
            warnings.warn(
                "max_centers is deprecated; use max_isomers instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.max_isomers = max_centers
        else:
            self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.show_progress = show_progress

    def _create_fresh_etkdg(self):
        """Create ETKDGv3 parameters for conformer generation"""
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xc0ffee
        params.numThreads = 0  # Use all available CPU threads
        params.pruneRmsThresh = 0.5
        return params
    
    def _filter_mol(self, smi, mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds"""

        # filter based on heavy atoms and rotatable bonds
        if rdMolDescriptors.CalcNumHeavyAtoms(mol) > self.max_heavy_atoms:
            message = f"Skipping {smi} with > {self.max_heavy_atoms} heavy atoms"
            if self.show_progress:
                logger.warning(message)
            else:
                logger.debug(message)
            return True

        if rdMolDescriptors.CalcNumRotatableBonds(mol) > self.max_rotatable_bonds:
            message = f"Skipping {smi} with > {self.max_rotatable_bonds} rotatable bonds"
            if self.show_progress:
                logger.warning(message)
            else:
                logger.debug(message)
            return True

        return False

    def _get_isomers(self, mol):
        """Generate isomers for a molecule using RDKit"""
        opts = StereoEnumerationOptions()
        opts.maxIsomers = self.max_isomers
        opts.onlyUnassigned = True
        opts.tryEmbedding = False
        opts.rand = 0xc0ffee
        for iso in EnumerateStereoisomers(mol, options=opts):
            yield iso
            
    def genConformers(self, smiles_list, out_dir) -> str:
        """Generate conformers using RDKit ETKDG
        
        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for.
            out_dir (str): Path to the output directory for the generated conformers.

        Returns:
            str: Path to the output conformers file (SDF)
        """
        tmp_outfile = os.path.join(out_dir, f"conformers.sdf")
        writer = Chem.SDWriter(tmp_outfile)

        etkdg = self._create_fresh_etkdg()

        if self.show_progress and len(smiles_list) > 50:
            logger.info("Generating conformers with RDKit ETKDG")

        for i, smi in enumerate(smiles_list):
            # Skip None and empty inputs
            if smi is None or not smi:
                continue
            
            mol = Chem.MolFromSmiles(smi)
            title = f"mol_{i}"

            if smi is None or mol is None:
                continue

            if self._filter_mol(smi, mol):
                continue

            for j, iso in enumerate(self._get_isomers(mol)):
                iso = Chem.AddHs(iso)
                iso.SetProp("_Name", f"{title}+{j}")
                
                try:
                    conf_ids = AllChem.EmbedMultipleConfs(
                        iso, 
                        numConfs=self.max_conformers, 
                        params=etkdg
                    )
                    
                    if len(conf_ids) > 0:
                        # Write all conformers
                        for conf_id in conf_ids:
                            writer.write(iso, confId=conf_id)
                    else:
                        message = f"{smi}: {iso.GetProp('_Name')} failed conformer generation"
                        if self.show_progress:
                            logger.warning(message)
                        else:
                            logger.debug(message)
                        
                except (RuntimeError, ValueError) as e:
                    message = (
                        f"{smi}: {iso.GetProp('_Name')} conformer generation error: {e}"
                    )
                    if self.show_progress:
                        logger.warning(message)
                    else:
                        logger.debug(message)

        writer.close()
        etkdg = None
        gc.collect()
        return tmp_outfile

    def write_conformers(self, mols: List, out_file: str) -> None:
        """Write conformers for RDKit molecules to an SDF file
        
        Args:
            mols: List of RDKit molecules
            out_file: Path to output SDF file
        """
        from rdkit import Chem
        
        # Convert molecules to SMILES
        smiles_list = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                smiles_list.append(smi)
            except Exception:
                continue
        
        if not smiles_list:
            # Create empty file
            with open(out_file, 'w') as f:
                pass
            return
        
        # Generate conformers to a temporary directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            sdf_file = self.genConformers(smiles_list, temp_dir)
            
            # Copy the generated SDF to the output location
            import shutil
            shutil.copy(sdf_file, out_file)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class CDPKitConformerGenerator(ConformerGenerator):
    """Conformer Generator using CDPKit CDPL.ConfGen

    CDPKit conformer generation following the official CDPKit examples.
    Based on gen_confs.py from CDPKit GitHub repository.
    
    This implementation uses StereoisomerGenerator for explicit stereoisomer enumeration,
    matching the behavior of RDKit and OpenEye implementations.

    Attributes:
        max_conformers (int): max number of conformers to generate per stereoisomer
        max_isomers (int): maximum number of stereoisomers to enumerate
        max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
        max_rotatable_bonds (int): drop molecules with more rotatable bonds than max_rotatable_bonds
        timeout (int): timeout for conformer generation in seconds
        min_rmsd (float): minimum RMSD between conformers
        energy_window (float): energy window for conformer selection in kcal/mol
        show_progress (bool): whether to show progress during conformer generation
    """
    
    def __init__(
        self,
        max_conformers: int = 10,
        max_isomers: int = 4,
        max_centers: int | None = None,
        max_heavy_atoms: int = 35,
        max_rotatable_bonds: int = 15,
        timeout: int = 3600,  # seconds (following CDPKit example)
        min_rmsd: float = 0.5,
        energy_window: float = 20.0,
        show_progress: bool = False,
    ):
        """Initialize the CDPKit conformer generator

        Args:
            max_conformers (int): max number of conformers to generate per stereoisomer
            max_isomers (int): maximum number of stereoisomers to enumerate
            max_centers (int, optional): deprecated alias for ``max_isomers``
            max_heavy_atoms (int): drop molecules with more heavy atoms than max_heavy_atoms
            max_rotatable_bonds (int): drop molecules with more rotatable bonds than max_rotatable_bonds
            timeout (int): timeout for conformer generation in seconds
            min_rmsd (float): minimum RMSD between conformers
            energy_window (float): energy window for conformer selection in kcal/mol
            show_progress (bool): whether to show progress during conformer generation
        """
        if not CDPL_AVAILABLE:
            raise ImportError("CDPKit not available. Install with: conda install -c conda-forge cdpkit")

        if max_conformers > 200:
            print(
                "Warning: max_conformers > 200 may cause memory issues. "
                "Setting to 200."
            )
            self.max_conformers = 200
        else:
            self.max_conformers = max_conformers

        if max_centers is not None:
            warnings.warn(
                "max_centers is deprecated; use max_isomers instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self.max_isomers = max_centers
        else:
            self.max_isomers = max_isomers
        self.max_heavy_atoms = max_heavy_atoms
        self.max_rotatable_bonds = max_rotatable_bonds
        self.timeout = timeout
        self.min_rmsd = min_rmsd
        self.energy_window = energy_window
        self.show_progress = show_progress
    
    def _create_conf_generator(self):
        """Create a CDPKit ConformerGenerator with optimal settings following CDPKit examples"""
        conf_gen = CDPLConfGen.ConformerGenerator()
        
        # Configure settings following CDPKit gen_confs.py example
        conf_gen.settings.timeout = self.timeout * 1000  # Convert to milliseconds
        conf_gen.settings.minRMSD = self.min_rmsd
        conf_gen.settings.energyWindow = self.energy_window
        conf_gen.settings.maxNumOutputConformers = self.max_conformers
        
        return conf_gen
    
    def _gen_conf_ensemble(self, mol, conf_gen):
        """Generate conformer ensemble following CDPKit gen_confs.py pattern
        
        Directly from CDPKit documentation:
        https://cdpkit.org/cdpl_python_cookbook/confgen/gen_ensemble.html
        
        Returns:
            tuple: (status, num_conformers)
        """
        # Prepare the molecule for conformer generation (from CDPKit example line 12)
        CDPLConfGen.prepareForConformerGeneration(mol)
        
        # Generate the conformer ensemble (from CDPKit example line 15)
        status = conf_gen.generate(mol)
        num_confs = conf_gen.getNumConformers()
        
        # If successful, set conformers to molecule (from CDPKit example line 20-21)
        if status == CDPLConfGen.ReturnCode.SUCCESS or status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
            conf_gen.setConformers(mol)
        else:
            num_confs = 0
            
        return status, num_confs
    
    def _smiles_to_cdpl_mol(self, smiles: str):
        """Convert SMILES to CDPKit molecule
        
        Minimal preparation - prepareForConformerGeneration() will handle
        all necessary molecular property calculations.
        """
        if not smiles:
            return None
            
        try:
            mol = CDPLChem.parseSMILES(smiles.strip())
            if mol is None or mol.getNumAtoms() == 0:
                return None
            
            return mol
        except Exception:
            return None
    
    def _filter_mol(self, smi: str, mol) -> bool:
        """Filter molecules based on heavy atoms and rotatable bonds

        Args:
            smi: SMILES string for error reporting
            mol: CDPKit molecule object

        Returns:
            True if molecule should be filtered out (rejected), False otherwise
        """
        if mol is None:
            return True

        try:
            heavy_atom_count = CDPLMolProp.getHeavyAtomCount(mol)

            if heavy_atom_count > self.max_heavy_atoms:
                message = f"Skipping {smi} with {heavy_atom_count} heavy atoms (max: {self.max_heavy_atoms})"
                if self.show_progress:
                    logger.warning(message)
                else:
                    logger.debug(message)
                return True

            # Try to filter by rotatable bonds to match RDKit/Omega behaviour
            try:
                rot_bonds = CDPLMolProp.getRotatableBondCount(mol)
                if rot_bonds > self.max_rotatable_bonds:
                    message = (
                        f"Skipping {smi} with {rot_bonds} rotatable bonds "
                        f"(max: {self.max_rotatable_bonds})"
                    )
                    if self.show_progress:
                        logger.warning(message)
                    else:
                        logger.debug(message)
                    return True
            except Exception:
                # If not available, don't block processing
                pass

            return False

        except Exception:
            return True

    def _get_isomers(self, mol):
        """Generate stereoisomers for a molecule using CDPKit StereoisomerGenerator
        
        This explicitly enumerates stereoisomers up to the ``max_isomers`` limit,
        matching the behavior of RDKit and OpenEye implementations.
        
        See: https://cdpkit.org/cdpl_api_doc/python_api_doc/classCDPL_1_1Chem_1_1StereoisomerGenerator.html
        
        Args:
            mol: CDPKit molecule object
            
        Yields:
            CDPKit molecule objects (stereoisomers)
        """
        # Create stereoisomer generator
        stereo_gen = StereoisomerGenerator()
        
        # Enable both atom and bond stereochemistry enumeration
        stereo_gen.enumerateAtomConfig(True)
        stereo_gen.enumerateBondConfig(True)
        
        # Don't include already specified centers (only enumerate unspecified)
        stereo_gen.includeSpecifiedCenters(False)
        
        # Set up the generator with the molecule
        stereo_gen.setup(mol)
        
        # Generate stereoisomers up to max_isomers limit
        count = 0
        while count < self.max_isomers:
            # Create a copy of the molecule for this stereoisomer
            iso_mol = CDPLChem.BasicMolecule(mol)
            
            # Generate next stereoisomer configuration
            if not stereo_gen.generate():
                break
                
            # Apply the stereochemistry descriptors to the molecule copy
            atom_descriptors = stereo_gen.getAtomDescriptors()
            bond_descriptors = stereo_gen.getBondDescriptors()
            
            # Set stereochemistry on the isomer molecule
            for i, desc in enumerate(atom_descriptors):
                if i < iso_mol.getNumAtoms():
                    CDPLChem.setStereoDescriptor(iso_mol.getAtom(i), desc)
            
            for i, desc in enumerate(bond_descriptors):
                if i < iso_mol.getNumBonds():
                    CDPLChem.setStereoDescriptor(iso_mol.getBond(i), desc)
            
            yield iso_mol
            count += 1
        
        # If no stereoisomers were generated, yield the original molecule
        if count == 0:
            yield mol
    
    def genConformers(self, smiles_list, out_dir) -> str:
        """Generate conformers for a list of SMILES and save to SDF
        
        For each input SMILES:
        1. Enumerate up to max_isomers stereoisomers using StereoisomerGenerator
        2. For each stereoisomer, generate up to max_conformers conformations
        3. Total output: up to (max_isomers × max_conformers) structures per molecule
        
        This matches the behavior of RDKit and OpenEye implementations.
        
        Args:
            smiles_list (list[str]): List of SMILES strings to generate conformers for
            out_dir (str): Output directory for the conformer SDF file
            
        Returns:
            str: Path to the generated SDF file with conformers
        """
        if not smiles_list:
            return ""
        
        os.makedirs(out_dir, exist_ok=True)
        tmp_outfile = os.path.join(out_dir, "conformers_cdpkit.sdf")
        
        if self.show_progress:
            logger.info(
                "Generating conformers for %d molecules using CDPKit",
                len(smiles_list),
            )
        
        # Create SDF writer (following CDPKit example)
        try:
            writer = CDPLChem.FileSDFMolecularGraphWriter(tmp_outfile)
        except Exception as e:
            if self.show_progress:
                logger.error("Error creating CDPKit SDF writer: %s", e)
            return ""
        
        # Create conformer generator once (following CDPKit example)
        conf_gen = self._create_conf_generator()
        
        total_conformers = 0
        valid_molecules = 0
        
        # Status code to string mapping (from CDPKit example)
        status_to_str = {
            CDPLConfGen.ReturnCode.UNINITIALIZED: 'uninitialized',
            CDPLConfGen.ReturnCode.TIMEOUT: 'max. processing time exceeded',
            CDPLConfGen.ReturnCode.ABORTED: 'aborted',
            CDPLConfGen.ReturnCode.FORCEFIELD_SETUP_FAILED: 'force field setup failed',
            CDPLConfGen.ReturnCode.FORCEFIELD_MINIMIZATION_FAILED: 'force field structure refinement failed',
            CDPLConfGen.ReturnCode.FRAGMENT_LIBRARY_NOT_SET: 'fragment library not available',
            CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_FAILED: 'fragment conformer generation failed',
            CDPLConfGen.ReturnCode.FRAGMENT_CONF_GEN_TIMEOUT: 'fragment conformer generation timeout',
            CDPLConfGen.ReturnCode.FRAGMENT_ALREADY_PROCESSED: 'fragment already processed',
            CDPLConfGen.ReturnCode.TORSION_DRIVING_FAILED: 'torsion driving failed',
            CDPLConfGen.ReturnCode.CONF_GEN_FAILED: 'conformer generation failed',
            CDPLConfGen.ReturnCode.NO_FIXED_SUBSTRUCT_COORDS: 'fixed substructure atoms do not provide 3D coordinates'
        }
        
        for i, smi in enumerate(smiles_list):
            if not smi or smi.strip() == "":
                continue
            
            # Convert SMILES to CDPL molecule
            mol = self._smiles_to_cdpl_mol(smi)
            if mol is None:
                if self.show_progress:
                    logger.warning("Failed to parse SMILES for CDPKit: %s", smi)
                continue
                
            # Filter molecules
            if self._filter_mol(smi, mol):
                continue

            # Generate stereoisomers and conformers for each
            for j, iso in enumerate(self._get_isomers(mol)):
                mol_name = f"mol_{i}+{j}"
                CDPLChem.setName(iso, mol_name)

                try:
                    # Generate conformer ensemble (following CDPKit example)
                    status, num_confs = self._gen_conf_ensemble(iso, conf_gen)

                    # Check for severe error reported by status code (from CDPKit example)
                    if status != CDPLConfGen.ReturnCode.SUCCESS and status != CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
                        if self.show_progress:
                            error_msg = status_to_str.get(status, f"unknown status {status}")
                            logger.warning(
                                "CDPKit conformer generation failed for %s: %s",
                                mol_name,
                                error_msg,
                            )
                        continue

                    # Output generated ensemble if available (from CDPKit example)
                    if num_confs > 0:
                        try:
                            writer.write(iso)
                            valid_molecules += 1
                            total_conformers += num_confs

                            if self.show_progress:
                                if status == CDPLConfGen.ReturnCode.TOO_MUCH_SYMMETRY:
                                    logger.info(
                                        "%s: generated %d conformers (too much symmetry)",
                                        mol_name,
                                        num_confs,
                                    )
                                else:
                                    logger.info(
                                        "%s: generated %d conformer(s)",
                                        mol_name,
                                        num_confs,
                                    )
                        except Exception as e:
                            if self.show_progress:
                                logger.warning(
                                    "Failed to write conformers for %s: %s",
                                    mol_name,
                                    e,
                                )
                    else:
                        if self.show_progress:
                            logger.warning("No CDPKit conformers generated for %s", mol_name)

                except Exception as e:
                    if self.show_progress:
                        logger.warning(
                            "CDPKit conformer generation error for %s: %s",
                            mol_name,
                            e,
                        )
        
        try:
            writer.close()
        except Exception:
            pass
        
        if self.show_progress:
            logger.info(
                "CDPKit conformer generation completed: %d molecules, %d conformers",
                valid_molecules,
                total_conformers,
            )
        
        # Clean up memory
        gc.collect()
        
        return tmp_outfile if total_conformers > 0 else ""

    def write_conformers(self, mols: List, out_file: str) -> None:
        """Write conformers for RDKit molecules to an SDF file
        
        Args:
            mols: List of RDKit molecules
            out_file: Path to output SDF file
        """
        from rdkit import Chem
        
        # Convert molecules to SMILES
        smiles_list = []
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
                smiles_list.append(smi)
            except Exception:
                continue
        
        if not smiles_list:
            # Create empty file
            with open(out_file, 'w') as f:
                pass
            return
        
        # Generate conformers to a temporary directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        try:
            sdf_file = self.genConformers(smiles_list, temp_dir)
            
            # Copy the generated SDF to the output location
            import shutil
            shutil.copy(sdf_file, out_file)
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
