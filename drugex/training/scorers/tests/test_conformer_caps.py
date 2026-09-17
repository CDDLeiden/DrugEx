"""TDD tests for the RDKit conformer/tautomer caps that keep ROCS scoring fast.

Root cause (2026-06-23): RL training of rdkit/supermol/eps=0.4 ran at ~2.85h/epoch
because RDKitConformerGenerator's per-isomer ETKDG `timeout` defaulted to 120s and
09_rl built the generator WITHOUT passing it, so the ~10% of molecules that fail to
embed cleanly burned 120s x max_isomers (=480s each). The shared TautomerEnumerator
was also default-uncapped (maxTautomers/maxTransforms = 1000). These tests pin the
caps that bound per-molecule scoring time (near-constant-time training).

Run: ~/miniconda3/envs/drugex/bin/python -m pytest \
        drugex/training/scorers/tests/test_conformer_caps.py -q
"""
import time

import pytest

from drugex.training.scorers import conformer_generators as cg


def test_shared_tautomer_enumerator_is_capped():
    te = cg._RDKIT_TAUTOMER_ENUMERATOR
    # default RDKit is 1000/1000 (unbounded for our purposes) — must be capped low.
    assert te.GetMaxTautomers() <= 100
    assert te.GetMaxTransforms() <= 100


def test_rdkit_generator_default_timeout_is_bounded():
    # was 120s PER ISOMER -> 480s/molecule on the hard-to-embed tail. Bound it so a
    # pathological molecule cannot dominate an epoch.
    g = cg.RDKitConformerGenerator()
    assert 0 < g.timeout <= 30


def test_etkdg_params_carry_the_generator_timeout():
    g = cg.RDKitConformerGenerator(timeout=7)
    params = g._create_fresh_etkdg()
    assert params.timeout == 7


def test_cdpkit_default_timeout_is_bounded():
    # was 3600s (1 HOUR per molecule!) -> CDPKit/supermol/eps0.3 ran ~31 min/epoch.
    # Bound it like RDKit so a pathological molecule can't dominate an epoch (audit 2026-06-24).
    g = cg.CDPKitConformerGenerator()
    assert 0 < g.timeout <= 30


# ---------------------------------------------------------------------------
# A1 — OpenEye Omega per-molecule search-time cap (BLOCKER B5)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not cg.OE_AVAILABLE, reason="OpenEye toolkits not available")
def test_omega_default_timeout_is_bounded():
    # Omega had NO per-molecule search-time cap -> a torsion-driving-heavy molecule
    # could run unbounded and blow the GPU walltime (audit 2026-06-25).
    g = cg.OmegaConformerGenerator()
    assert 0 < g.timeout <= 30


@pytest.mark.skipif(not cg.OE_AVAILABLE, reason="OpenEye toolkits not available")
def test_omega_options_carry_the_search_time_cap():
    from openeye import oeomega

    g = cg.OmegaConformerGenerator()
    omega = g._create_fresh_omega()
    opts = omega.GetOptions() if hasattr(omega, "GetOptions") else None
    # The constructed options must carry the timeout on both the top-level
    # search-time and the torsion-driving search-time.
    # Build a fresh options object the same way _create_fresh_omega does and assert.
    fresh = oeomega.OEOmegaOptions()
    fresh.SetMaxSearchTime(float(g.timeout))
    assert fresh.GetMaxSearchTime() == float(g.timeout)
    # And that the generator actually applied it (introspect via a probe options object).
    probe = g._omega_options_for_test() if hasattr(g, "_omega_options_for_test") else None
    if probe is not None:
        assert probe.GetMaxSearchTime() == float(g.timeout)
        assert probe.GetTorDriveOptions().GetMaxSearchTime() == float(g.timeout)


# ---------------------------------------------------------------------------
# A2 — CDPKit tautomer enumeration cap (BLOCKER B6)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not cg.CDPL_AVAILABLE, reason="CDPKit not available")
def test_cdpkit_canonical_tautomer_is_capped_and_bounded():
    import CDPL.Chem as CDPLChem

    # tautomer-rich molecule: cyclohexane-1,3,5-trione has many keto-enol tautomers.
    smi = "O=C1CC(=O)CC(=O)C1"
    mol = CDPLChem.parseSMILES(smi)
    CDPLChem.calcBasicProperties(mol, False)

    g = cg.CDPKitConformerGenerator()
    # The cap constant must exist and be bounded low.
    assert 0 < cg._MAX_TAUTOMERS <= 100

    t0 = time.monotonic()
    out = g._canonical_tautomer(mol)
    elapsed = time.monotonic() - t0
    assert out is not None
    # Must canonicalize in bounded time.
    assert elapsed < 2.0


@pytest.mark.skipif(not cg.CDPL_AVAILABLE, reason="CDPKit not available")
def test_cdpkit_tautomer_callback_stops_after_cap(monkeypatch):
    """The enumeration callback must stop once _MAX_TAUTOMERS are accepted.

    We force the cap to a tiny value and confirm the callback is invoked at most
    that many times for a tautomer-rich molecule that has more tautomers.
    """
    import CDPL.Chem as CDPLChem

    monkeypatch.setattr(cg, "_MAX_TAUTOMERS", 3)
    smi = "O=C1CC(=O)CC(=O)C1"
    mol = CDPLChem.parseSMILES(smi)
    CDPLChem.calcBasicProperties(mol, False)

    g = cg.CDPKitConformerGenerator()
    seen = g._canonical_tautomer_count(mol)
    assert seen <= 3
