"""Multiprocessing start-method selection for the ROCS scorers.

The cdpkit and rdkit ROCS scorers parallelise per-molecule scoring with a ``multiprocessing``
``Pool``. The default ``fork`` start method is dangerous inside the RL loop: the Pool is
created *after* ``forward()``/``evolve()`` have spun up torch/OpenMP worker threads, and
forking many workers from a multi-threaded process can deadlock — a child inherits a mutex
(e.g. the allocator lock) locked by a parent thread that does not exist in the child, so a
worker wedges and ``pool.map`` blocks forever. This is probabilistic and worsens with worker
count; it hung cdpkit/supermol/eps0.3/pt_pt (cell 615) at n_jobs=62, epoch 8.

``forkserver`` forks workers from a clean intermediary process that never had those threads,
so it sidesteps the deadlock; unlike ``spawn`` it does not re-import ``__main__`` (the RL
training runs from a ``python << PYEOF`` heredoc, which has no importable main module).

``DRUGEX_MP_CONTEXT`` (``fork`` | ``forkserver`` | ``spawn``) selects the method. The default
is ``fork`` so external users see byte-for-byte unchanged behaviour; the CCR2 RL scripts set
``forkserver``.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
from contextlib import contextmanager


class ScoreTimeout(Exception):
    """Raised when a single molecule exceeds the per-molecule scoring time limit."""


def pool_context(default: str = "fork"):
    """Return the multiprocessing context for scorer Pools, honouring DRUGEX_MP_CONTEXT.
    An unknown/unavailable method falls back to ``default``."""
    method = os.environ.get("DRUGEX_MP_CONTEXT") or default
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context(default)


def capped_n_jobs(requested: int) -> int:
    """Cap the scorer worker count via ``DRUGEX_SCORE_NJOBS``.

    A very high worker count (NCPUS=62) can exhaust memory and/or wedge the fork for
    molecule-heavy batches: the cdpkit pt_pt cell 615 reliably hung in ``getRewards`` at
    n_jobs=62 (62 workers each loading a large per-batch conformer file), while leaner cdpkit
    cells scored fine. Fewer, fatter workers avoid it at a negligible throughput cost for the
    fast cdpkit/openeye backends. Unset / non-positive / non-integer ``DRUGEX_SCORE_NJOBS``
    means "no cap" (use ``requested``); the cap never *increases* the worker count.
    """
    cap = os.environ.get("DRUGEX_SCORE_NJOBS")
    if not cap:
        return requested
    try:
        c = int(cap)
    except ValueError:
        return requested
    if c <= 0:
        return requested
    return min(requested, c)


def score_timeout():
    """Per-batch scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_TIMEOUT``.

    Returns ``None`` (unset / non-positive / non-numeric) meaning "wait forever" — the
    current behaviour for external users. When set, the parallel scorer bounds ``pool.map``
    with it: on timeout it terminates the (wedged) workers and leaves the unscored molecules
    at score 0, so a single pathological molecule that hangs an unbounded scoring step
    (tautomer/protonation/alignment) can never stall a whole epoch (cdpkit cell 615).
    """
    v = os.environ.get("DRUGEX_SCORE_TIMEOUT")
    if not v:
        return None
    try:
        t = float(v)
    except ValueError:
        return None
    return t if t > 0 else None


def mol_timeout():
    """Per-MOLECULE scoring wall-clock cap in seconds, from ``DRUGEX_SCORE_MOL_TIMEOUT``.
    ``None`` (unset / non-positive / non-numeric) = no per-molecule cap (current behaviour)."""
    v = os.environ.get("DRUGEX_SCORE_MOL_TIMEOUT")
    if not v:
        return None
    try:
        t = float(v)
    except ValueError:
        return None
    return t if t > 0 else None


@contextmanager
def molecule_time_limit(seconds):
    """Best-effort per-molecule wall-clock cap via ``SIGALRM``; raises ``ScoreTimeout`` when
    exceeded. A no-op when ``seconds`` is falsy, when not on the main thread (SIGALRM is
    main-thread-only — Pool workers run their task on their own main thread, so it applies
    there and in the serial path), or on a platform without ``SIGALRM``.

    This bounds Python-interruptible hangs — including CDPKit's Python tautomer callback and
    the per-conformer alignment loop — so a single pathological molecule is scored 0 (invalid
    -> 0 reward) instead of wedging the whole epoch (cell 615). A pure-C single-call hang that
    never returns to Python is caught by the Pool-level ``score_timeout`` backstop instead.
    """
    if (not seconds
            or threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "SIGALRM")):
        yield
        return

    def _handler(signum, frame):
        raise ScoreTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
