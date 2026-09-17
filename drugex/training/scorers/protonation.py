"""Shared pH-based protonation primitive (Dimorphite-DL) — Task D.

ROCS *color* (pharmacophore) overlap depends on ionization state: a neutral carboxyl
carries no anion feature, a neutral amine no cation feature. The CCR2 references are
stored anionic, but generated molecules are scored neutral (review finding A4-1), which
under-measures color. This protonates a molecule to its dominant state at physiological
pH so query and reference color features are on a common basis. Apply it BEFORE conformer
generation. If Dimorphite-DL is unavailable or errors, the input is returned unchanged
(pipeline-safe).

Tool choice (see ccr2_gen/docs/color_validation/EXPERIMENT_PROTONATION_RESULTS.md):
Dimorphite-DL (protonation only) — NOT Gypsum-DL, whose 3D/tautomer/stereo enumeration
would duplicate the scorers' own conformer generators (Gypsum-DL calls Dimorphite-DL for
its protonation anyway).
"""
from typing import List, Union

try:
    import dimorphite_dl

    _DIMORPHITE_AVAILABLE = True
except ImportError:
    _DIMORPHITE_AVAILABLE = False

DEFAULT_PH = 7.4  # physiological


def _protonate_one(smi: str, ph: float) -> str:
    if not smi or not _DIMORPHITE_AVAILABLE:
        return smi
    try:
        # precision=0.0 -> the single DOMINANT protomer per site at this pH (n=1).
        # (Do NOT use max_variants=1: its truncation is not dominance-ordered and
        #  returns the neutral amine instead of the cation.)
        variants = dimorphite_dl.protonate_smiles(smi, ph_min=ph, ph_max=ph, precision=0.0)
        return variants[0] if variants else smi
    except Exception:
        return smi


def protonate_smiles(
    smiles: Union[str, List[str]], ph: float = DEFAULT_PH
) -> Union[str, List[str]]:
    """Return the dominant protomer at ``ph`` (default 7.4) via Dimorphite-DL.

    Accepts a single SMILES (returns ``str``) or a list (returns a list of the same
    length, order preserved). Acids → anion, bases → cation, neutral molecules unchanged.
    Invalid SMILES or a missing/erroring Dimorphite-DL fall back to the input unchanged.
    """
    if isinstance(smiles, str):
        return _protonate_one(smiles, ph)
    return [_protonate_one(s, ph) for s in smiles]
