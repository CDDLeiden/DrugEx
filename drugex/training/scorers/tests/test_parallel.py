"""TDD tests for the scorer multiprocessing-context helper.

Root cause (2026-06-25): the cdpkit (and rdkit) ROCS scorers create a multiprocessing
``Pool`` with the default ``fork`` start method. In the RL loop the Pool is created *after*
``forward()``/``evolve()`` have spun up torch/OpenMP worker threads, so forking many workers
(n_jobs=62) hits the classic fork-after-threads deadlock — a child inherits a mutex locked by
a parent thread that does not exist in the child, ``pool.map`` blocks forever, and the epoch
never finishes (cdpkit/supermol/eps0.3/pt_pt, cell 615). ``forkserver`` forks workers from a
clean intermediary that never had those threads, avoiding the deadlock. This helper makes the
start method configurable (default ``fork`` — byte-for-byte for external users; the RL scripts
opt into ``forkserver``).

Run: ~/miniconda3/envs/drugex/bin/python -m pytest drugex/training/scorers/tests/test_parallel.py -q
"""
import multiprocessing as mp
import time

import pytest

from drugex.training.scorers.parallel import (
    pool_context, capped_n_jobs, score_timeout, mol_timeout,
    molecule_time_limit, ScoreTimeout,
)


def test_default_is_fork_when_env_unset(monkeypatch):
    monkeypatch.delenv("DRUGEX_MP_CONTEXT", raising=False)
    assert pool_context().get_start_method() == "fork"


def test_env_selects_forkserver(monkeypatch):
    monkeypatch.setenv("DRUGEX_MP_CONTEXT", "forkserver")
    assert pool_context().get_start_method() == "forkserver"


def test_env_selects_spawn(monkeypatch):
    monkeypatch.setenv("DRUGEX_MP_CONTEXT", "spawn")
    assert pool_context().get_start_method() == "spawn"


def test_bogus_method_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("DRUGEX_MP_CONTEXT", "not_a_method")
    assert pool_context().get_start_method() == "fork"


def test_explicit_default_arg_used_when_env_unset(monkeypatch):
    monkeypatch.delenv("DRUGEX_MP_CONTEXT", raising=False)
    assert pool_context(default="forkserver").get_start_method() == "forkserver"


def test_returns_a_real_context_usable_for_pool(monkeypatch):
    monkeypatch.setenv("DRUGEX_MP_CONTEXT", "forkserver")
    ctx = pool_context()
    # the returned object must expose Pool (i.e. it is a real multiprocessing context)
    assert hasattr(ctx, "Pool")
    assert ctx.get_start_method() in mp.get_all_start_methods()


# --- capped_n_jobs: bound scorer worker count (DRUGEX_SCORE_NJOBS) -------------------
def test_no_cap_when_env_unset(monkeypatch):
    monkeypatch.delenv("DRUGEX_SCORE_NJOBS", raising=False)
    assert capped_n_jobs(62) == 62


def test_cap_lowers_high_worker_count(monkeypatch):
    # high counts can exhaust memory / wedge the fork for molecule-heavy cdpkit batches
    monkeypatch.setenv("DRUGEX_SCORE_NJOBS", "16")
    assert capped_n_jobs(62) == 16


def test_cap_does_not_raise_already_low_count(monkeypatch):
    monkeypatch.setenv("DRUGEX_SCORE_NJOBS", "16")
    assert capped_n_jobs(8) == 8       # min(8, 16) -> 8, never increases workers


def test_cap_ignored_when_non_positive_or_garbage(monkeypatch):
    monkeypatch.setenv("DRUGEX_SCORE_NJOBS", "0")
    assert capped_n_jobs(62) == 62
    monkeypatch.setenv("DRUGEX_SCORE_NJOBS", "notanint")
    assert capped_n_jobs(62) == 62


def test_cap_preserves_serial(monkeypatch):
    monkeypatch.setenv("DRUGEX_SCORE_NJOBS", "16")
    assert capped_n_jobs(1) == 1       # serial stays serial


# --- score_timeout: per-batch scoring wall-clock cap (DRUGEX_SCORE_TIMEOUT) ----------
def test_score_timeout_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("DRUGEX_SCORE_TIMEOUT", raising=False)
    assert score_timeout() is None     # default: wait forever (current behaviour)


def test_score_timeout_reads_seconds(monkeypatch):
    monkeypatch.setenv("DRUGEX_SCORE_TIMEOUT", "300")
    assert score_timeout() == 300.0


def test_score_timeout_none_on_garbage_or_nonpositive(monkeypatch):
    monkeypatch.setenv("DRUGEX_SCORE_TIMEOUT", "0")
    assert score_timeout() is None
    monkeypatch.setenv("DRUGEX_SCORE_TIMEOUT", "-5")
    assert score_timeout() is None
    monkeypatch.setenv("DRUGEX_SCORE_TIMEOUT", "notanumber")
    assert score_timeout() is None


# --- mol_timeout + molecule_time_limit: per-molecule scoring cap --------------------
def test_mol_timeout_env(monkeypatch):
    monkeypatch.delenv("DRUGEX_SCORE_MOL_TIMEOUT", raising=False)
    assert mol_timeout() is None
    monkeypatch.setenv("DRUGEX_SCORE_MOL_TIMEOUT", "60")
    assert mol_timeout() == 60.0
    monkeypatch.setenv("DRUGEX_SCORE_MOL_TIMEOUT", "0")
    assert mol_timeout() is None


def test_molecule_time_limit_raises_when_exceeded():
    with pytest.raises(ScoreTimeout):
        with molecule_time_limit(0.3):
            time.sleep(2.0)            # exceeds -> SIGALRM -> ScoreTimeout


def test_molecule_time_limit_passes_fast_block():
    with molecule_time_limit(2.0):
        x = sum(range(1000))           # well under the limit
    assert x == 499500                 # no exception, alarm disarmed cleanly


def test_molecule_time_limit_noop_when_none():
    # None means "no cap" — must not arm an alarm or raise
    with molecule_time_limit(None):
        time.sleep(0.05)


def test_molecule_time_limit_noop_off_main_thread():
    # SIGALRM is main-thread-only; off-thread it must be a silent no-op (not raise ValueError)
    import threading
    err = []
    def run():
        try:
            with molecule_time_limit(0.1):
                time.sleep(0.3)        # would raise on main thread; off-thread -> no-op
        except Exception as e:         # noqa: BLE001
            err.append(e)
    t = threading.Thread(target=run); t.start(); t.join()
    assert err == []
