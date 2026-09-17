"""Integration test: a single pathological molecule cannot stall scoring.

Root cause (2026-06-25, cdpkit/supermol/eps0.3/pt_pt cell 615): a molecule whose shape
alignment spins in an unbounded step wedged a Pool worker AND the serial fallback, hanging the
whole epoch for hours. The fix wraps each molecule's scoring in a per-molecule SIGALRM
(molecule_time_limit) so a hanging molecule is scored 0 (invalid -> 0 reward) and the worker
survives, plus a Pool-level get(timeout) backstop. This test injects an unbounded hang into the
alignment and asserts getScores returns in bounded time with that molecule scored 0.

Run: OE_LICENSE=... DRUGEX_SCORE_MOL_TIMEOUT=2 \
     ~/miniconda3/envs/drugex/bin/python -m pytest \
     drugex/training/scorers/tests/test_score_timeout_integration.py -q
"""
import time

import pytest

import drugex.training.scorers.rocs_cdpkit as rc

CDPL = getattr(rc, "CDPL_AVAILABLE", False)
REF = "tutorial/advanced/rocs/rocs_rl_ccr/rdkit_cdpkit/supermol_123.sdf"


@pytest.mark.skipif(not CDPL, reason="CDPKit not available")
def test_pathological_molecule_is_bounded_not_hanging(monkeypatch):
    import os
    from rdkit import Chem
    from drugex.training.scorers.conformer_generators import CDPKitConformerGenerator

    monkeypatch.setenv("DRUGEX_SCORE_MOL_TIMEOUT", "2")

    # Inject an unbounded hang into the per-conformer alignment (the cell-615 failure site).
    def _hang(*a, **k):
        time.sleep(30)
        return 0.0
    monkeypatch.setattr(rc, "_align_and_score_helper", _hang)

    sc = rc.CDPKitROCSScorer(
        conformer_generator=CDPKitConformerGenerator(
            max_conformers=5, max_isomers=1, show_progress=False),
        references={"CCR2": REF}, opt_mode="combo", show_progress=False,
        n_jobs=1,  # serial path runs in-process so the monkeypatched hang is exercised
    )
    mol = Chem.MolFromSmiles("O=C(CN1C(=O)COc2ccccc21)N1CCOCC1")

    t0 = time.time()
    out = sc.getScores([mol])
    elapsed = time.time() - t0

    assert elapsed < 10, f"scoring not bounded: {elapsed:.1f}s (per-molecule timeout failed)"
    # the hung molecule must be scored 0 (invalid), not crash the batch
    assert float(out[0][0]) == 0.0
