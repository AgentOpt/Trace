"""
╔══════════════════════════════════════════════════════════════════════════╗
║         TraceMemoryDB – Trainer / Optimizer   End‑to‑End Test Suite      ║
╠════════╦══════════════════════════╦══════════════════════════════════════╣
║  Test  ║  Purpose                 ║  Key Assertions                      ║
╠════════╬══════════════════════════╬══════════════════════════════════════╣
║ 1      ║ Full step‑by‑step logging║ * TraceMemoryDB captures every       ║
║        ║ with minimal patching    ║   system/user prompt and LLM reply   ║
║        ║                          ║ * Works with in‑memory & Chroma      ║
╠════════╬══════════════════════════╬══════════════════════════════════════╣
║ 2      ║ Rollback on stagnation   ║ * Detect no‑progress window (N)      ║
║        ║ (single candidate)       ║ * Revert to best of last N _or_      ║
║        ║                          ║   tagged checkpoint                  ║
╠════════╬══════════════════════════╬══════════════════════════════════════╣
║ 3      ║ Log reuse for dynamic    ║ * All “reasoning” fields captured    ║
║        ║ prompt building (V2)     ║ * Can aggregate them into a follow‑  ║
║        ║                          ║   up reflection prompt               ║
╚════════╩══════════════════════════╩══════════════════════════════════════╝

*All tests run genuine `call_llm` calls; set the usual API keys (or local LLM
endpoint) beforehand.  A beefy CI job is recommended.*
# Example:
export OPENAI_API_KEY="your-key"
export TRACE_LITELLM_MODEL="gpt-4.1-nano"
"""

# ─────────────────────────────── Imports ──────────────────────────────── #
import json
import math
import os
import tempfile
from pathlib import Path

import pytest

# Project‑internal imports
from opto.trainer.trace_memory_db import TraceMemoryDB
from opto.trainer.trace_memory_db import UnifiedVectorDBConfig, UnifiedVectorDB  # chroma backend
from opto.trace.nodes import ParameterNode
from opto.optimizers import OptoPrime, OptoPrimeV2, OptoPrimeMulti, optimizer


# ────────────────────────── Fixtures & Helpers ────────────────────────── #
def _build_memory_db(backend: str) -> TraceMemoryDB:
    """
    backend  :=  "in_memory"   (pure RAM, no VectorDB)
                 "chroma"      (local Chroma instance under tmp dir)
    """
    if backend == "in_memory":
        return TraceMemoryDB()  # hot‑cache only
    if backend == "chroma":
        # Skip cleanly if chromadb is missing
        try:
            import chromadb  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("Chroma backend requested but chromadb not installed")
        tmp = tempfile.TemporaryDirectory()
        cfg = UnifiedVectorDBConfig(
            collection_name="tmdb_test",
            persist_directory=str(Path(tmp.name) / "vectordb"),
            reset_indices=True,
        )
        vdb = UnifiedVectorDB(cfg, check_db=False)
        return TraceMemoryDB(vector_db=vdb)
    raise ValueError(f"Unknown backend: {backend}")


@pytest.fixture(params=["in_memory", "chroma"], scope="module")
def memory_db(request):
    """Parametrised TraceMemoryDB – backed by RAM or Chroma."""
    return _build_memory_db(request.param)


# -- extremely tiny helper to create dummy feedback text
def _feedback_for_target(x: float, target: float = 10) -> str:
    out = 2 * x
    if math.isclose(out, target, abs_tol=1e-9):
        return "Correct"
    delta = abs(int(target - out))
    return f"Output is too {'low' if out < target else 'high'} by {delta}"


# ─────────────────────── ❶ Full Logging Demonstration ──────────────────── #
def test_full_logging_minimal_class_patch(memory_db):
    """
    Demonstrates minimal trainer monkey‑patch to pipe every
    prompt / reply into TraceMemoryDB without altering optimiser logic.
    """
    param = ParameterNode(0, name="x")
    optimiser = OptoPrimeV2(parameters=[param])

    # Monkey‑patch *only* the two attr that hold the rolling logs
    class _MemProxy:
        def __init__(self):
            self.step = 0

        def add(self, tpl):
            self.step += 1
            variables, feedback = tpl
            memory_db.log_data( problem_id="full_log", step_id=self.step, data={"variables": {k: v for k, v in variables.items()}, "feedback": feedback}, data_payload="feedback",)

        def __len__(self):
            # Return count of feedback records stored in TraceMemoryDB
            return len(memory_db.get_data(problem_id="full_log", data_payload="feedback"))

        def __iter__(self):
            # Return an iterator that yields (variables, feedback) tuples from TraceMemoryDB
            feedback_records = memory_db.get_data(problem_id="full_log", data_payload="feedback")
            for record in feedback_records:
                variables = record["data"]["variables"]
                feedback = record["data"]["feedback"]
                yield variables, feedback


    class _LogProxy:
        def __init__(self):
            self.step = 0

        def append(self, rec):
            self.step += 1
            memory_db.log_data(problem_id="full_log", step_id=self.step, data=rec, data_payload="candidate",)

        def __len__(self):
            # Return count of candidate records stored in TraceMemoryDB
            return len(memory_db.get_data(problem_id="full_log", data_payload="candidate"))

        def __iter__(self):
            # Return an iterator that yields candidate records from TraceMemoryDB
            candidate_records = memory_db.get_data(problem_id="full_log", data_payload="candidate")
            for record in candidate_records:
                yield record["data"]

    optimiser.memory = _MemProxy()
    optimiser.log = _LogProxy()
    optimiser.summary_log = []

    # --- lightweight optimisation loop (real LLM) ---
    max_steps = 2  # keep CI cost minimal
    for _ in range(max_steps):
        fb = _feedback_for_target(param._data)
        # optimiser.feedback = {param: fb}
        optimiser.zero_feedback()
        optimiser.backward(param, fb)
        optimiser.step()

    # Assert TraceMemoryDB populated
    cand_recs = memory_db.get_data(problem_id="full_log", data_payload="candidate")
    fb_recs = memory_db.get_data(problem_id="full_log", data_payload="feedback")
    assert cand_recs, "No candidate logs captured"
    assert fb_recs, "No feedback logs captured"


# ──────────────────────── ❷ Rollback / Patience ───────────────────────── #
@pytest.mark.parametrize("patience", [1, 2])
@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_rollback_on_stagnation(memory_db, patience, use_checkpoint):
    """
    Runs a few steps, injects scores, rolls back to best candidate
    of last *patience* or to last checkpoint tag.
    """
    param = ParameterNode(0, name="x")
    optimiser = OptoPrime(parameters=[param])

    # patch logs → memory_db
    class _LogProxy:
        def __init__(self):
            self.step = 0

        def append(self, rec):
            self.step += 1
            memory_db.log_data(
                problem_id="rollback",
                step_id=self.step,
                data=rec,
                data_payload="candidate",
            )

    optimiser.log = _LogProxy()
    optimiser.summary_log = []

    best_score = -float("inf")
    stagnation = 0

    for step in range(1, 5):
        # ----- run a step -----
        fb = _feedback_for_target(param._data)
        optimiser.zero_feedback()
        optimiser.backward(param, fb)
        patch = optimiser.step()
        if patch:
            optimiser.update(patch)

        # ----- score & log -----
        output = 2 * float(param._data)
        score = -abs(output - 10)  # zero is best
        memory_db.log_data( problem_id="rollback", step_id=step, data={"candidate": float(param._data)}, data_payload="candidate", scores={"score": score},)
        if use_checkpoint and score >= best_score:
            memory_db.log_data(
                problem_id="rollback", step_id=step, data={"note": "checkpoint"}, data_payload="checkpoint",)

        # ----- track patience -----
        if score > best_score + 1e-9:
            best_score = score
            stagnation = 0
        else:
            stagnation += 1
        if stagnation >= patience:
            break

    # ----- determine rollback target -----
    if use_checkpoint:
        ckpts = memory_db.get_data(problem_id="rollback", data_payload="checkpoint")
        if ckpts:
            target_step = ckpts[-1]["step_id"]
        else:
            target_step = None
    else:
        target_step = None

    if target_step is None:
        recent = memory_db.get_last_n(patience + 1, problem_id="rollback", data_payload="candidate")
        target_step = max(recent, key=lambda r: r["scores"]["score"])["step_id"]

    # rollback
    rec = memory_db.get_data(problem_id="rollback", step_id=target_step, data_payload="candidate")[0]

    param._data = rec["data"]["candidate"]
    rolled_score = -abs(2 * param._data - 10)
    assert rolled_score >= best_score - 1e-9

# ──────────────────── ❸ Dynamic Prompt Re‑use (V2) ────────────────────── #
def test_dynamic_prompt_reuse(memory_db):
    """
    Ensures we can extract all reasoning strings from OptoPrimeV2 logs
    and stitch them into a follow‑up reflection prompt.
    """
    param = ParameterNode(2, name="x")
    optimiser = OptoPrimeV2(parameters=[param], include_example=False)

    # patch logs → memory_db
    class _LogProxy:
        def __init__(self):
            self.step = 0

        def append(self, rec):
            self.step += 1
            memory_db.log_data( problem_id="reuse", step_id=self.step, data=rec, data_payload="suggestion",)

    optimiser.log = _LogProxy()
    optimiser.summary_log = []

    # --- run until LLM says stop (max 3 steps) ---
    for _ in range(3):
        fb = _feedback_for_target(param._data)
        optimiser.feedback = {param: fb}
        patch = optimiser.step()
        if not patch:
            break
        optimiser.update(patch)

    # --- collect reasoning fields ---
    reasonings = []
    recs = memory_db.get_data(problem_id="reuse", data_payload="suggestion")
    for r in recs:
        raw = r["data"].get("response", "")
        try:
            j = json.loads(raw) if isinstance(raw, str) else raw
            reason = j.get("reasoning", "")
            if reason:
                reasonings.append(reason.strip())
        except Exception:
            continue

    # Build combined prompt
    combined = "\n".join(reasonings)
    new_prompt = f"Earlier reasoning steps:\n{combined}\nNow, summarise the overall strategy."
    # At least one reasoning string should be present
    assert reasonings, "No reasoning logs found – adjust prompt for your LLM?"
    # Check each reasoning is contained
    for r in reasonings:
        assert r in new_prompt
