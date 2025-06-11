#!/usr/bin/env python3
"""
Real ROCS API scorer using authentic OpenEye FastROCS shape-based scoring.
This implementation follows the color_opt pattern for proper ROCS API usage.
"""

import os
import gzip
import tempfile
from typing import Union, List

try:
    from openeye import oechem, oeshape, oefastrocs
    OE_AVAILABLE = True
except ImportError:
    OE_AVAILABLE = False

from drugex.training.scorers.interfaces import Scorer

class ROCSAPIError(Exception):
    """Custom exception for ROCS API-specific errors"""
    pass

class ROCSAPIScorer(Scorer):
    """Real ROCS API scorer using authentic shape-based scoring"""
    
    def getKey(self):
        return "ROCSAPIScorer (Real Shape-based Scoring)"
    
    def __init__(
        self,
        query_files: Union[str, List[str]],
        score_type: str = "TanimotoCombo",
        max_conformers: int = 200,
        shape_only: bool = False,
        optimize: bool = True,
        color_optimize: bool = True,
        color_force_field: str = "ImplicitMillsDean",
        output_file: str = None,
        use_gpu: bool = False
    ):
        super().__init__()
        if not OE_AVAILABLE:
            raise ImportError("OpenEye oechem toolkit is required")
        
        # Convert query files to absolute paths
        if isinstance(query_files, str):
            self.query_files = [os.path.abspath(query_files)]
        else:
            self.query_files = [os.path.abspath(qf) for qf in query_files]
        
        # Store parameters matching CLI scorer exactly
        self.score_type = score_type
        self.max_conformers = max_conformers
        self.shape_only = shape_only
        self.optimize = optimize
        self.color_optimize = color_optimize
        self.color_force_field = color_force_field
        self.output_file = output_file
        self.use_gpu = use_gpu
        
        # Validate environment and setup
        self._validate_environment()
        self._setup_rocs_options()
        
        # Initialize query
        self.query = None
        self._load_query()
    
    def _validate_environment(self):
        """Validate OpenEye environment and GPU availability"""
        # Check GPU availability if requested
        if self.use_gpu:
            if not oefastrocs.OEFastROCSIsGPUReady():
                print("Warning: GPU requested but not available, falling back to CPU")
                self.use_gpu = False
            else:
                print("✓ GPU acceleration available and enabled")
        else:
            print("Using CPU mode for ROCS scoring")
    
    def _setup_rocs_options(self):
        """Configure ROCS database options matching CLI parameters exactly"""
        self.opts = oefastrocs.OEShapeDatabaseOptions()
        
        # Map CLI parameters to API options:
        # CLI: -cutoff -1.0 (return all molecules) → API: Remove SetLimit restriction
        # Don't set any limit to match CLI "-cutoff -1.0" behavior
        # self.opts.SetLimit(500)  # REMOVED - this was limiting results
        
        # CLI: -optchem true/false → API: SetColorOptimization()
        # Only enable color optimization if not shape-only and color_optimize is True
        enable_color = self.color_optimize and not self.shape_only
        self.opts.SetColorOptimization(enable_color)
        
        # CLI: CPU/GPU mode → API: SetFastROCSMode
        if not self.use_gpu:
            self.opts.SetFastROCSMode(oefastrocs.OEFastROCSMode_ROCS)
        
        print(f"ROCS options configured: ColorOpt={enable_color}, GPU={self.use_gpu}, NoLimit=True")
    
    def _load_query(self):
        """Load query from .sq shape query file or molecule file"""
        query_file = self.query_files[0]
        print(f"Loading query from: {os.path.basename(query_file)}")
        
        # Try to read as shape query first (.sq format)
        if query_file.endswith('.sq'):
            try:
                self.query = oeshape.OEShapeQuery()
                if oeshape.OEReadShapeQuery(query_file, self.query):
                    print("✓ Successfully loaded shape query (.sq format)")
                    return
                else:
                    print("Failed to read .sq file, trying as molecule")
            except Exception as e:
                print(f"Error reading .sq file: {e}")
        
        # Try to read as molecule file
        try:
            qfs = oechem.oemolistream()
            if qfs.open(query_file):
                query_mol = oechem.OEGraphMol()
                if oechem.OEReadMolecule(qfs, query_mol):
                    self.query = query_mol
                    print("✓ Successfully loaded query molecule")
                    qfs.close()
                    return
                qfs.close()
        except Exception as e:
            print(f"Error reading molecule file: {e}")
        
        raise ROCSAPIError(f"Failed to load query from {query_file}")
    
    def _create_rocs_database(self, molecules_file):
        """Create OEShapeDatabase from molecules file following color_opt.py pattern"""
        print(f"Creating ROCS database from: {os.path.basename(molecules_file)}")
        
        # Handle gzipped files by creating temporary uncompressed version
        actual_file = molecules_file
        self.temp_file = None  # Store as instance variable to prevent cleanup during scoring
        
        if molecules_file.endswith('.gz'):
            print("Decompressing gzipped file...")
            self.temp_file = tempfile.NamedTemporaryFile(suffix='.oeb', delete=False)
            self.temp_file.close()
            
            with gzip.open(molecules_file, 'rb') as f_in:
                with open(self.temp_file.name, 'wb') as f_out:
                    f_out.write(f_in.read())
            actual_file = self.temp_file.name
        
        try:
            # Follow color_opt.py pattern lines 19-33
            ifs = oechem.oemolistream()
            if not ifs.open(actual_file):
                raise ROCSAPIError(f"Unable to open molecule file: {actual_file}")
            
            print("Initializing ROCS shape database...")
            timer = oechem.OEWallTimer()
            
            # Create database objects (this is where SIGABRT could occur)
            dbase = oefastrocs.OEShapeDatabase()
            moldb = oechem.OEMolDatabase()
            
            if not moldb.Open(ifs):
                raise ROCSAPIError(f"Unable to open molecule database: {actual_file}")
            
            # Use progress dots for large databases
            dots = oechem.OEThreadedDots(10000, 50, "conformers")
            if not dbase.Open(moldb, dots):
                raise ROCSAPIError(f"Unable to initialize OEShapeDatabase on: {actual_file}")
            
            dots.Total()
            print(f"Database created in {timer.Elapsed():.2f} seconds")
            
            # Keep file stream open by storing reference
            self.ifs = ifs
            
            return dbase, moldb
            
        except Exception as e:
            # Clean up and re-raise with context
            self._cleanup_temp_files()
            raise ROCSAPIError(f"Database creation failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_file') and self.temp_file and os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
            self.temp_file = None
    
    def _perform_rocs_scoring(self, query, dbase, moldb, file_label=0):
        """Execute real ROCS shape scoring following color_opt.py pattern"""
        print("Performing ROCS shape-based scoring...")
        results = []
        
        # Configure scoring options - no limit to match CLI "-cutoff -1.0" behavior
        numhits = moldb.NumMols()
        print(f"Processing all {numhits} molecules (no limit applied)")
        # self.opts.SetLimit(min(numhits, 500))  # REMOVED - process all molecules
        
        # Score molecules (follow color_opt.py pattern lines 58-82)
        score_count = 0
        for score in dbase.GetSortedScores(query, self.opts):
            dbmol = oechem.OEMol()
            molidx = score.GetMolIdx()
            
            if not moldb.GetMolecule(dbmol, molidx):
                print(f"Warning: Unable to retrieve molecule {molidx} from database")
                continue
            
            # Get the specific conformer that was scored
            mol = oechem.OEGraphMol(dbmol.GetConf(oechem.OEHasConfIdx(score.GetConfIdx())))
            
            # Extract conformer ID to match CLI naming format
            conf_idx = score.GetConfIdx()
            base_name = mol.GetTitle() if mol.GetTitle() else f"mol_{molidx}"
            
            # Include conformer ID in molecule name to match CLI format: "ZINC04617942_123"
            mol_name = f"{base_name}_{conf_idx}"
            
            # Extract authentic ROCS scores
            shape_tanimoto = score.GetShapeTanimoto()
            color_tanimoto = score.GetColorTanimoto()
            tanimoto_combo = score.GetTanimotoCombo()
            
            results.append({
                "Name": mol_name,
                "ShapeQuery": os.path.basename(self.query_files[0]),
                "TanimotoCombo": round(tanimoto_combo, 3),
                "ShapeTanimoto": round(shape_tanimoto, 3),
                "ColorTanimoto": round(color_tanimoto, 3),
                "Rank": 0,  # Will be set during sorting
                "Active": file_label  # Store the file label for proper Active column
            })
            
            score_count += 1
        
        # Sort by TanimotoCombo (descending) and assign ranks
        results.sort(key=lambda x: x['TanimotoCombo'], reverse=True)
        for i, result in enumerate(results):
            result['Rank'] = i + 1
        
        print(f"Scored {score_count} molecules successfully")
        return results
    
    def _format_tsv_output(self, results, output_file, append=False, file_label=0):
        """Generate TSV output matching CLI format exactly"""
        if not results:
            print("Warning: No results to write")
            return
        
        mode = 'a' if append else 'w'
        with open(output_file, mode) as f:
            # Write header only if not appending or file is empty
            if not append or not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                # Match CLI header format exactly with Active column
                header = "Name\tShapeQuery\tRank\tTanimotoCombo\tShapeTanimoto\tColorTanimoto\tActive\n"
                f.write(header)
            
            # Write results matching CLI format with Active column
            for result in results:
                line = f"{result['Name']}\t{result['ShapeQuery']}\t{result['Rank']}\t"
                line += f"{result['TanimotoCombo']:.3f}\t{result['ShapeTanimoto']:.3f}\t"
                # Use stored Active value if present, otherwise use file_label parameter
                active_val = result.get('Active', file_label)
                line += f"{result['ColorTanimoto']:.3f}\t{active_val}\n"
                f.write(line)
        
        print(f"Results written to: {output_file}")
    
    def getScores(self, mols_or_file, frags=None, output_file=None, append=False, file_label=0):
        """Score molecules using real ROCS API with authentic shape-based scoring"""
        use_output_file = output_file if output_file is not None else self.output_file
        
        if not isinstance(mols_or_file, str) or not os.path.exists(mols_or_file):
            raise NotImplementedError("API scorer currently only supports file input")
        
        molecules_file = mols_or_file
        
        try:
            # Create ROCS database from molecules file
            dbase, moldb = self._create_rocs_database(molecules_file)
            
            # Perform authentic ROCS scoring
            results = self._perform_rocs_scoring(self.query, dbase, moldb, file_label)
            
            # Save results if output file specified
            if use_output_file:
                self._format_tsv_output(results, use_output_file, append, file_label)
            
            return results
            
        except ROCSAPIError as e:
            print(f"ROCS API Error: {e}")
            print("Consider using CLI scorer as fallback")
            raise
        except Exception as e:
            print(f"Unexpected error during ROCS scoring: {e}")
            raise ROCSAPIError(f"Scoring failed: {e}")
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
