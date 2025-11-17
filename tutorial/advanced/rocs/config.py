"""
Configuration and Setup for DrugEx ROCS RL Tutorial

This module provides predefined settings and a setup helper function
to simplify the quickstart notebook.
"""

from pathlib import Path

from drugex.data.corpus.vocabulary import VocSmiles
from drugex.training.generators import SequenceRNN
from drugex.training.environment import DrugExEnvironment
from drugex.training.rewards import ParetoCrowdingDistance
from drugex.training.scorers.modifiers import SmoothClippedScore
from drugex.training.scorers.properties import Property
from drugex.training.scorers.conformer_generators import RDKitConformerGenerator
from drugex.training.scorers.rdkit_rocs import RDKitROCSScorer


# ============================================================================
# Default Paths
# ============================================================================

ROOT = Path.cwd()
CCR2_SDF = ROOT / 'rocs_rl_ccr/rdkit_cdpkit/CCR2_reference_ligands.sdf'
MODEL_DIR = ROOT / 'demo_out/models'
OUTPUT_DIR = ROOT / 'rl_runs_demo/rdkit_rl'
MODELS_PR_PATH = str(Path(__file__).parent.parent.parent / 'data/models/pretrained/smiles-rnn/Papyrus05.5_smiles_rnn_PT')

FINETUNE_BASE = MODEL_DIR / 'CCR2_finetuned'
FINETUNE_CHECKPOINT = Path(f"{FINETUNE_BASE}.pkg")
FINETUNE_VOCAB = Path(f"{FINETUNE_BASE}.vocab")


# ============================================================================
# RL Training Parameters
# ============================================================================

RL_EPOCHS = 50
RL_EPSILON = 0.2
RL_N_SAMPLES = 1000


# ============================================================================
# Conformer Generation Parameters (Aligned with OpenEye OMEGA)
# ============================================================================

MAX_CONFORMERS = 50          # Increased from 20 to match OMEGA output
MAX_ISOMERS = 4              # Increased from 2 to match OpenEye filter
MAX_HEAVY_ATOMS = 45         # Increased from 30 to match OpenEye filter
MAX_ROTATABLE_BONDS = 15     # NEW: Match OpenEye filter for flexibility


# ============================================================================
# Scoring Thresholds
# ============================================================================

# ROCS TanimotoCombo threshold
# Range: 0-2 (ShapeTanimoto 0-1 + ColorTanimoto 0-1)
# Molecules with score >= threshold are "Desired"
ROCS_THRESHOLD = 0.871 # computed using threshold_analysis.py

# SA Score threshold
# After SmoothClippedScore transformation: 0-1 range
# Molecules with score >= threshold are "Desired"
SA_THRESHOLD = 0.1  # synthetic accessibility

# Combined thresholds list for DrugExEnvironment
OBJECTIVE_THRESHOLDS = [ROCS_THRESHOLD, SA_THRESHOLD]


# ============================================================================
# Setup Functions
# ============================================================================

def create_rdkit_environment():
    """
    Create DrugEx environment with RDKit ROCS and SA scorers.

    Returns:
        DrugExEnvironment configured for CCR2 ligand design
    """
    # RDKit ROCS scorer
    rocs_scorer = RDKitROCSScorer(
        conformer_generator=RDKitConformerGenerator(
            max_conformers=MAX_CONFORMERS,
            max_isomers=MAX_ISOMERS,
            max_heavy_atoms=MAX_HEAVY_ATOMS,
            max_rotatable_bonds=MAX_ROTATABLE_BONDS,
            num_threads=1,  # Avoid CPU oversubscription with n_jobs=-1
            show_progress=False,
        ),
        references=str(CCR2_SDF),
        score_type='TanimotoCombo',
        use_colors=True,
        show_progress=False,
        n_jobs=-1,
    )

    # Synthetic accessibility scorer
    sa_scorer = Property('SA')
    sa_scorer.setModifier(SmoothClippedScore(lower_x=5, upper_x=3))

    # Create environment
    env = DrugExEnvironment(
        scorers=[rocs_scorer, sa_scorer],
        thresholds=OBJECTIVE_THRESHOLDS,
        reward_scheme=ParetoCrowdingDistance()
    )

    return env


def setup_rl_rdkit():
    """
    Complete setup for RDKit ROCS RL experiment.

    Loads vocabulary, creates models, and sets up environment.

    Returns:
        tuple: (agent, mutate, env, output_dir)
            - agent: SequenceRNN initialized from pretrained model
            - mutate: SequenceRNN initialized from fine-tuned model
            - env: DrugExEnvironment with ROCS and SA scorers
            - output_dir: Path for saving results

    Raises:
        FileNotFoundError: If fine-tuned model not found
    """
    # Verify fine-tuned model exists
    if not FINETUNE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Fine-tuned model not found at: {FINETUNE_CHECKPOINT}\n"
            "Please run: python prepare_models.py"
        )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load vocabulary
    voc = VocSmiles.fromFile(str(FINETUNE_VOCAB), encode_frags=False)
    print(f"Vocabulary loaded: {voc.size} tokens")

    # Create environment
    env = create_rdkit_environment()
    print("Environment created with RDKit ROCS + SA scorers")

    # Initialize agent (pretrained model)
    agent = SequenceRNN(voc, is_lstm=True)
    agent.loadStatesFromFile(str(Path(MODELS_PR_PATH) / 'Papyrus05.5_smiles_rnn_PT.pkg'))
    print("Agent loaded from pretrained model")

    # Initialize mutate network (fine-tuned model)
    mutate = SequenceRNN(voc, is_lstm=True)
    mutate.loadStatesFromFile(str(FINETUNE_CHECKPOINT))
    print("Mutate network loaded from fine-tuned model")

    print(f"\nSetup complete! Using device: {agent.device}")

    return agent, mutate, env, OUTPUT_DIR


def get_paths():
    """
    Get dictionary of all important paths.

    Returns:
        dict: Dictionary containing all paths
    """
    return {
        'root': ROOT,
        'ccr2_sdf': CCR2_SDF,
        'model_dir': MODEL_DIR,
        'output_dir': OUTPUT_DIR,
        'pretrained_path': MODELS_PR_PATH,
        'finetune_checkpoint': FINETUNE_CHECKPOINT,
        'finetune_vocab': FINETUNE_VOCAB,
    }


def get_config():
    """
    Get dictionary of all configuration parameters.

    Returns:
        dict: Dictionary containing all config parameters
    """
    return {
        'rl_epochs': RL_EPOCHS,
        'rl_epsilon': RL_EPSILON,
        'rl_n_samples': RL_N_SAMPLES,
        'max_conformers': MAX_CONFORMERS,
        'max_isomers': MAX_ISOMERS,
        'max_heavy_atoms': MAX_HEAVY_ATOMS,
        'rocs_threshold': ROCS_THRESHOLD,
        'sa_threshold': SA_THRESHOLD,
        'objective_thresholds': OBJECTIVE_THRESHOLDS,
    }
