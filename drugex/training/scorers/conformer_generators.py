import gc
import os

try:
    from openeye import oechem, oemolprop, oeomega

    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False
    
import os
import subprocess
from typing import Callable

from drugex.training.scorers.interfaces import ConformerGenerator
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
                                               StereoEnumerationOptions)


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

