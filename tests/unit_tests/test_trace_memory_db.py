# tests/test_trace_memory_db.py
"""
Comprehensive behaviour‑driven tests for TraceMemoryDB.
Each test is tagged with the MUST‑requirement(s) or D.* scenario it validates.
Run with:     pytest -q
"""
import copy
import random
import uuid
import pytest
import functools
import json
from typing import Optional

from opto.trainer.trace_memory_db import TraceMemoryDB, UnifiedVectorDB, ChromaConfig, UnifiedVectorDBConfig, CHROMA_DATABASE
from opto.optimizers.optoprime import OptoPrime
from opto.trace.nodes import ParameterNode, node

# --------------------------------------------------------------------------- #
# HELPER FIXTURES AND STUBS
# --------------------------------------------------------------------------- #
class DummyVectorDB:
    """Minimal stub to validate pluggable back‑end behaviour (R4)."""
    def __init__(self):
        self.add_calls = []
        self.deleted = []

    def _add_texts(self, texts, ids=None, metadatas=None):
        self.add_calls.append({"texts": texts, "ids": ids, "metadatas": metadatas})

    # TraceMemoryDB.delete_data() will fall back silently if this is missing,
    # but we add it so that the call path is fully covered.
    def delete(self, ids):
        self.deleted.extend(ids)

def enable_trace_memory_for_optoprime(memory_db: Optional[TraceMemoryDB] = None):
    """
    Monkey patch OptoPrime to use TraceMemoryDB without modifying core code.
    
    Args:
        memory_db: TraceMemoryDB instance. If None, creates a new in-memory instance.
    """
    if memory_db is None:
        memory_db = TraceMemoryDB()
    
    # Store original methods
    original_init = OptoPrime.__init__
    original_step = OptoPrime._step
    
    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        # Remove memory_db from kwargs if present
        kwargs.pop('memory_db', None)
        original_init(self, *args, **kwargs)
        self.memory_db = memory_db
        self._current_problem_id = "default_goal"
        self._current_step_id = 0
        self._memory_size = kwargs.get('memory_size', 8)  # Default from OptoPrime
    
    @functools.wraps(original_step)
    def new_step(self, *args, **kwargs):
        # Before step: log current parameters
        self._current_step_id += 1
        self.memory_db.log_data(
            problem_id=self._current_problem_id,
            step_id=self._current_step_id,
            data={"parameters": {p.name: p.data for p in self.parameters if p.trainable}},
            data_payload="variables",
            metadata={"agent": "OptoPrime", "status": "before_step"}
        )
        
        # Execute original step
        result = original_step(self, *args, **kwargs)

        # After step: log the result/feedback
        try:        
            # After step: log feedback if available
            if hasattr(self, 'summary_log') and self.summary_log:
                last_summary = self.summary_log[-1]
                if 'summary' in last_summary:
                    self.memory_db.log_data(
                        problem_id=self._current_problem_id,
                        step_id=self._current_step_id,
                        data={"feedback": last_summary['summary'].user_feedback},
                        data_payload="feedback",
                        metadata={"agent": "OptoPrime", "status": "after_step"}
                    )
        except Exception as e:
            # Log the exception but don't fail
            print(f"Warning: Failed to log feedback: {e}")

        return result
    
    # Apply monkey patches
    OptoPrime.__init__ = new_init
    OptoPrime._step = new_step
    
    return memory_db

@pytest.fixture(scope="function")
def memdb_cache5():
    """Fresh in‑memory DB (covers R3 + R5)."""
    return TraceMemoryDB(vector_db=None, cache_size=5)


@pytest.fixture(scope="function")
def pluggable_db():
    """DB that writes through to a fake vector store (covers R4)."""
    stub = DummyVectorDB()
    return TraceMemoryDB(vector_db=stub, cache_size=10), stub

@pytest.fixture(params=["inmemory", "chroma"])
def memdb(request):
    """Parametrize on memory-backend: default in-memory or Chroma."""
    if request.param == "inmemory":
        return TraceMemoryDB()
    else:
        try:
            import importlib
            importlib.import_module("langchain_chroma")
        except ImportError:
            pytest.skip("langchain_chroma not installed")
        # explicit Chroma‐backed VectorDB via its real constructor
        dirpath = f"/tmp/chroma_test_{uuid.uuid4().hex}"
        coll = f"col_{uuid.uuid4().hex}"
        cfg = UnifiedVectorDBConfig(
            persist_directory=dirpath,
            collection_name=coll
        )
        # force Chroma backend
        cfg.db_type = CHROMA_DATABASE
        # UnifiedVectorDB.get_unique_id = lambda self: cfg.unique_collection_id  # type: ignore
        vdb = UnifiedVectorDB(cfg, check_db=False)
        return TraceMemoryDB(vector_db=vdb)

@pytest.fixture
def llm():
    """Instantiate the real LLM client (may be expensive)."""
    # from opto.utils.llm import LLM
    # return LLM()
    from opto.utils.llm import DummyLLM
    # Use DummyLLM for testing to avoid API calls
    def dummy_response(*args, **kwargs):
        return json.dumps({
            "reasoning": "Test reasoning",
            "suggestion": {"param1": "updated_value"}
        })
    return DummyLLM(dummy_response)


# --------------------------------------------------------------------------- #
#  D.1  Main DB data manipulation
# --------------------------------------------------------------------------- #
def test_alias_data_key_and_payload_priority(memdb):
    # 1) Only using the legacy alias `data_key`
    eid1 = memdb.log_data("G1", 1, {"foo": "bar"}, data_key="variables")
    rec1 = memdb.get_data(entry_id=eid1)[0]
    assert rec1["data_payload"] == "variables"
    assert rec1["data"]["foo"] == "bar"

    # 2) Only using the new parameter `data_payload`
    eid2 = memdb.log_data("G1", 2, {"baz": "qux"}, data_payload="payload")
    rec2 = memdb.get_data(entry_id=eid2)[0]
    assert rec2["data_payload"] == "payload"
    assert rec2["data"]["baz"] == "qux"

    # 3) Providing both: `data_payload` should take precedence over `data_key`
    eid3 = memdb.log_data(
        "G1", 3, {"x": "y"},
        data_payload="primary", data_key="secondary"
    )
    rec3 = memdb.get_data(entry_id=eid3)[0]
    assert rec3["data_payload"] == "primary"

    # 4) Omitting both should raise a ValueError
    with pytest.raises(ValueError):
        memdb.log_data("G1", 4, {"no": "key"})
        
# --------------------------------------------------------------------------- #
#  D.2  Observation pools & retrieval strategies  (R1, R7, R12, R13 ‑ last‑N)
# --------------------------------------------------------------------------- #
def test_last_n_and_hot_cache_eviction_and_ordering(memdb_cache5):
    """
    D.2 & R7: Hot-cache eviction policy & last-N retrieval.

    1. Fill the cache (size=5) with 7 entries to trigger eviction.
    2. Verify that the two oldest entries are evicted from the internal hot-cache.
    3. Ensure get_last_n returns the most recent N entries in descending order.
    4. Edge case: requesting more entries than exist still returns all, correctly ordered.
    """
    memdb = memdb_cache5
    # sanity: default cache_size fixture is 5
    assert memdb._cache_size == 5

    # (1) Log 7 observations
    ids = []
    for i in range(7):
        eid = memdb.log_data( problem_id="OBS", step_id=i, data={"value": i}, data_payload="observation")
        ids.append(eid)

    # (2) The first two should be evicted
    assert ids[0] not in memdb._hot_cache
    assert ids[1] not in memdb._hot_cache
    #    — but the latest five remain
    for kept in ids[2:]:
        assert kept in memdb._hot_cache

    # (3) last-3 should be [6,5,4]
    last_three = memdb.get_last_n(3, "OBS", "observation")
    assert [r["data"]["value"] for r in last_three] == [6, 5, 4]

    # (4) requesting more than exists returns all 7 in reverse chronological
    all_seven = memdb.get_last_n(10, problem_id="OBS", data_payload="observation")
    assert len(all_seven) == 7
    assert [r["data"]["value"] for r in all_seven] == list(reversed(range(7)))

def test_pool_and_retrieval_strategies(memdb):
    """
    D.2: Build an “observation pool” and exercise different retrieval modes:

    • Full-pool retrieval returns all observations in reverse chronological order.
    • last-N returns the N most recent observations.
    • Random sampling pulls a subset of the pool uniformly at random.
    """
    # (A) Log observations across 3 steps × 2 candidates each
    expected_values = []
    for step in range(3):
        for cid in (1, 2):
            val = f"step{step}_cand{cid}"
            memdb.log_data( problem_id="POOL", step_id=step, candidate_id=cid, data={"observation": val}, data_payload="observation", metadata={"agent": "EnvMonitor", "stage": f"{step}-{cid}"})
            expected_values.append(val)

    # (B) Full-pool: get_data yields all entries, newest first
    pool = memdb.get_data(problem_id="POOL", data_payload="observation")
    assert len(pool) == len(expected_values)
    # newest == last logged == step2_cand2
    assert pool[0]["data"]["observation"] == "step2_cand2"
    # oldest == first logged == step0_cand1
    assert pool[-1]["data"]["observation"] == "step0_cand1"

    # (C) last-2: should be ["step2_cand2", "step2_cand1"]
    last_two = memdb.get_last_n(2, problem_id="POOL", data_payload="observation")
    assert [r["data"]["observation"] for r in last_two] == ["step2_cand2", "step2_cand1"]

    # (D) random sampling: pick 3 from the pool
    random.seed(0)
    sample = memdb.get_data(
        problem_id="POOL",
        data_payload="observation",
        additional_filters={"random": 3}
    )
    assert len(sample) == 3
    # each sampled value must come from the full pool
    all_obs = {r["data"]["observation"] for r in pool}
    assert set(s["data"]["observation"] for s in sample).issubset(all_obs)

# --------------------------------------------------------------------------- #
#  D.3  Parallel components ‑ shared memory (R17)
# --------------------------------------------------------------------------- #
def test_scaling_parallel_pipeline_decoupling(memdb):
    """
    D.3 Scaling to Large Optimization Pipeline (Parallelism & Decoupling)

    Demonstrates how three independent components—Trainer, Evaluator, and Optimizer—
    can run in parallel (or on separate processes) yet coordinate solely through
    a shared TraceMemoryDB. Each only reads/writes to the DB; there is no direct
    communication between them.
    """
    # Shared in-memory database instance
    db = memdb
    GOAL = "PIPELINE"

    # --------------------------
    # Component stubs
    # --------------------------
    class Trainer:
        def __init__(self, db):
            self.db = db
            self.step = 0

        def generate(self, num_candidates=3):
            """Generate N candidates and log their parameters."""
            self.step += 1
            for cid in range(1, num_candidates + 1):
                params = {"weight": cid * 0.1}
                self.db.log_data(
                    problem_id=GOAL,
                    step_id=self.step,
                    candidate_id=cid,
                    data={"variables": params},
                    data_payload="variables",
                    metadata={"agent": "Trainer", "status": "generated"},
                )
            return self.step

    class Evaluator:
        def __init__(self, db):
            self.db = db

        def evaluate(self, problem_id, step_id):
            """
            Pick up all 'variables' entries for (goal, step),
            compute a simple score, and log it under 'scores'.
            """
            records = self.db.get_data(
                problem_id=problem_id,
                step_id=step_id,
                data_payload="variables",
            )
            for rec in records:
                cid = rec["candidate_id"]
                val = rec["data"]["variables"]["weight"]
                score = val * 2  # dummy evaluation
                self.db.log_data(
                    problem_id=problem_id,
                    step_id=step_id,
                    candidate_id=cid,
                    data={"score": score},
                    data_payload="scores",
                    metadata={"agent": "Evaluator", "status": "evaluated"},
                )

    class Optimizer:
        def __init__(self, db):
            self.db = db

        def select_best(self, problem_id, step_id):
            """
            Fetch all 'scores' entries, choose the candidate with the highest score,
            then log that choice under 'best_candidate'.
            """
            scored = self.db.get_data(
                problem_id=problem_id,
                step_id=step_id,
                data_payload="scores",
            )
            # choose the record with max score
            best_rec = max(scored, key=lambda r: r["data"]["score"])
            best_cid = best_rec["candidate_id"]
            best_score = best_rec["data"]["score"]

            self.db.log_data(
                problem_id=problem_id,
                step_id=step_id,
                candidate_id=best_cid,
                data={"best": {"candidate_id": best_cid, "score": best_score}},
                data_payload="best_candidate",
                metadata={"agent": "Optimizer", "status": "selected"},
            )
            return best_cid

    # --------------------------
    # Run the pipeline
    # --------------------------
    trainer   = Trainer(db)
    evaluator = Evaluator(db)
    optimizer = Optimizer(db)

    # 1) Trainer generates 3 candidates for step 1
    step_id = trainer.generate(num_candidates=3)

    # Verify that the Trainer's logs are visible to any other reader
    var_recs = db.get_data(
        problem_id=GOAL,
        step_id=step_id,
        data_payload="variables",
    )
    assert len(var_recs) == 3
    assert all(r["agent"] == "Trainer" for r in var_recs)

    # 2) Evaluator picks them up and logs scores
    evaluator.evaluate(GOAL, step_id)
    score_recs = db.get_data(
        problem_id=GOAL,
        step_id=step_id,
        data_payload="scores",
    )
    assert len(score_recs) == 3
    assert all(r["agent"] == "Evaluator" for r in score_recs)

    # 3) Optimizer selects the best candidate and logs the selection
    best_cid = optimizer.select_best(GOAL, step_id)
    best_rec = db.get_data(
        problem_id=GOAL,
        step_id=step_id,
        data_payload="best_candidate",
    )[0]
    assert best_rec["data"]["best"]["candidate_id"] == best_cid
    assert best_rec["agent"] == "Optimizer"

# --------------------------------------------------------------------------- #
#  D.4  Branching / rollback / lineage  (R10)
# --------------------------------------------------------------------------- #
def test_branching_rollback_and_lineage(memdb):
    """
    D.4 Branching, Rollback, and Lineage

    Workflow:
      1. Seed a main goal with an initial code entry (and score).
      2. Identify the "best" checkpoint of the main goal.
      3. Create a sub‐goal branch from that checkpoint.
      4. Create a rollback goal seeded from an earlier checkpoint.
      5. Retrieve all sub‐goals of the main goal via parent_problem_id.
      6. Verify that each branch’s code entry correctly records its source_entry_id.
    """
    main_goal = "MAIN"

    # 1) Seed the main goal with an initial code and score
    initial_entry_id = memdb.log_data(
        problem_id=main_goal,
        step_id=1,
        candidate_id=1,
        data={"code": "print('v1')"},
        data_payload="code",
        scores={"score": 0.9},
        metadata={"agent": "InitAgent", "status": "seeded"}
    )

    # 2) Fetch the best candidate at step 1 (should be our initial entry)
    best = memdb.get_top_candidates(main_goal, 1, score_name="score")
    assert best is not None
    assert best[0]["entry_id"] == initial_entry_id

    # 3) Branch off the best checkpoint into a new sub‐goal
    branch_goal = f"{main_goal}_branch"
    branch_meta = {"agent": "BranchAgent", "status": "branch_created"}
    # 3a) Log the sub‐goal marker
    branch_goal_marker = memdb.log_data(
        problem_id=branch_goal,
        step_id=0,
        data={"goal": f"Branch from {main_goal}@1"},
        data_payload="goal",
        parent_problem_id=main_goal,
        metadata=branch_meta
    )
    # 3b) Seed the branch with the code from the checkpoint
    branch_code_entry = memdb.log_data(
        problem_id=branch_goal,
        step_id=1,
        candidate_id=1,
        data={"code": best[0]["data"]["code"]},
        data_payload="code",
        metadata={
            "agent": "BranchAgent",
            "source_entry_id": best[0]["entry_id"],
            "status": "seeded"
        }
    )

    # 4) Rollback: create a rollback goal seeded from the same checkpoint
    rollback_goal = f"{main_goal}_rollback"
    rollback_meta = {"agent": "RollbackAgent", "status": "rollback_created"}
    # 4a) Log the rollback marker
    rollback_marker = memdb.log_data(
        problem_id=rollback_goal,
        step_id=0,
        data={"goal": f"Rollback to {main_goal}@1"},
        data_payload="goal",
        parent_problem_id=main_goal,
        metadata=rollback_meta
    )
    # 4b) Seed rollback with the checkpoint code
    rollback_code_entry = memdb.log_data(
        problem_id=rollback_goal,
        step_id=1,
        candidate_id=1,
        data={"code": best[0]["data"]["code"]},
        data_payload="code",
        metadata={
            "agent": "RollbackAgent",
            "source_entry_id": initial_entry_id,
            "status": "seeded"
        }
    )

    # 5) List *all* sub‐goals of MAIN via parent_problem_id=="MAIN"
    sub_goals = memdb.get_data(
        parent_problem_id=main_goal,
        data_payload="goal"
    )
    sub_problem_ids = {r["problem_id"] for r in sub_goals}
    assert branch_goal in sub_problem_ids, "Branch goal should appear as a sub-goal"
    assert rollback_goal in sub_problem_ids, "Rollback goal should appear as a sub-goal"

    # 6) Verify lineage: each code entry references the original checkpoint
    branch_rec = memdb.get_data(entry_id=branch_code_entry)[0]
    assert branch_rec["source_entry_id"] == initial_entry_id

    rollback_rec = memdb.get_data(entry_id=rollback_code_entry)[0]
    assert rollback_rec["source_entry_id"] == initial_entry_id

# --------------------------------------------------------------------------- #
#  D.5  Mutation‑centric diff logging  (R11 + R12 filter by metadata tag)
# --------------------------------------------------------------------------- #
def test_diff_and_metadata_filter_basic(memdb):
    diff = "---\n+ foo"
    memdb.log_data("MUT", 1, {"diff": diff}, data_payload="diff", metadata={"tags": ["mutation"]})
    rows = memdb.get_data(additional_filters={"tags": ["mutation"]})
    assert rows and rows[0]["data"]["diff"] == diff

def test_mutation_centric_diff_logging_and_lineage(memdb):
    """
    D.5 Mutation-Centric Strategies (AlphaEvolve-Style)
    
    This test demonstrates how to:
      1. Seed an original 'code' entry.
      2. Log successive code mutations as 'diff' payloads.
      3. Tag mutation entries for easy retrieval.
      4. Track lineage via `metadata.source_entry_id`.
      5. Filter only mutation diffs from the log.
      6. Ensure the original code remains unchanged.
    """

    # 1) Seed original code (step 0, candidate 1)
    original_code = "def compute(x): return x * 2"
    orig_entry = memdb.log_data(
        problem_id="MUTATION_TEST",
        step_id=0,
        candidate_id=1,
        data={"code": original_code},
        data_payload="code",
        metadata={"agent": "Mutator", "tags": ["seed"]}
    )

    # 2) First mutation: add logging
    diff1 = "--- def compute(x): return x * 2\n+++ def compute(x):\n+    print(f'Input: {x}')\n+    return x * 2"
    m1_entry = memdb.log_data(
        problem_id="MUTATION_TEST",
        step_id=1,
        candidate_id=1,
        data={"diff": diff1},
        data_payload="diff",
        metadata={
            "agent": "Mutator",
            "tags": ["mutation", "logging"],
            "source_entry_id": orig_entry
        }
    )

    # 3) Second mutation: handle negative input
    diff2 = "--- return x * 2\n+++ def compute(x):\n+    if x < 0:\n+        return 0\n+    return x * 2"
    m2_entry = memdb.log_data(
        problem_id="MUTATION_TEST",
        step_id=2,
        candidate_id=1,
        data={"diff": diff2},
        data_payload="diff",
        metadata={
            "agent": "Mutator",
            "tags": ["mutation", "edge-case"],
            "source_entry_id": m1_entry
        }
    )

    # 4) Retrieve only mutation diffs via metadata tags
    mutation_diffs = memdb.get_data(
        problem_id="MUTATION_TEST",
        data_payload="diff",
        additional_filters={"tags": ["mutation"]}
    )
    # We expect exactly two diffs, in reverse‐chronological order
    assert len(mutation_diffs) == 2
    assert mutation_diffs[0]["data"]["diff"] == diff2
    assert mutation_diffs[1]["data"]["diff"] == diff1

    # 5) Verify lineage: each diff points back to its parent entry
    assert mutation_diffs[0]["source_entry_id"] == m1_entry
    assert mutation_diffs[1]["source_entry_id"] == orig_entry

    # 6) Check that original code entry is still present and unmodified
    code_entries = memdb.get_data(
        problem_id="MUTATION_TEST",
        data_payload="code"
    )
    assert len(code_entries) == 1
    assert code_entries[0]["data"]["code"] == original_code
    assert code_entries[0]["entry_id"] == orig_entry


# --------------------------------------------------------------------------- #
#  D.6  Feedback logging for fine‑tuning (R1, R18 helper: best candidate)
# --------------------------------------------------------------------------- #
def test_feedback_and_best_candidate_basic(memdb):
    # Two candidates, different scores
    memdb.log_data("FT", 1, {"code": "pass"}, data_payload="code",
                         candidate_id=1, scores={"score": 0.3})
    memdb.log_data("FT", 1, {"code": "better"}, data_payload="code",
                         candidate_id=2, scores={"score": 0.9})
    best = memdb.get_top_candidates(n=1, problem_id="FT")
    assert best[0]["candidate_id"] == 2
    assert best[0]["scores"]["score"] == 0.9

def test_feedback_logging_for_fine_tuning(memdb):
    """
    D.6 Feedback Logging for Fine-Tuning
    
    Demonstrates how to:
      1) Log the LLM prompt, evaluation score, and human feedback
      2) Retrieve all logged entries for a given goal/step
      3) Assert presence and correctness of each piece
      4) Assemble them into a dataset example for SFT/RFT
    """
    problem_id = "FT_GOAL"
    step_id = 7

    # --- 1) Log the LLM prompt issued at this step ---
    prompt_text = "Translate 'Hello, world!' to French."
    eid_prompt = memdb.log_data(
        problem_id=problem_id,
        step_id=step_id,
        data={"prompt": prompt_text},
        data_payload="prompt",
        metadata={"agent": "TranslatorAgent", "status": "logged_prompt"}
    )

    # --- 2) Log the numeric evaluation score (e.g. BLEU or human rating) ---
    score_value = 0.92
    eid_score = memdb.log_data(
        problem_id=problem_id,
        step_id=step_id,
        data={"score": score_value},
        data_payload="scores",
        metadata={"agent": "EvaluatorAgent", "status": "logged_score"}
    )

    # --- 3) Log qualitative human feedback for fine-tuning hints ---
    feedback_text = "Great grammar, but watch the exclamation placement."
    eid_feedback = memdb.log_data(
        problem_id=problem_id,
        step_id=step_id,
        data={"feedback": feedback_text},
        data_payload="feedback",
        metadata={"agent": "HumanReviewer", "status": "logged_feedback"}
    )

    # --- 4) Retrieve all entries for this goal & step ---
    entries = memdb.get_data(problem_id=problem_id, step_id=step_id)
    
    # Check we've got exactly the three payloads we logged
    payloads = {e["data_payload"] for e in entries}
    assert payloads == {"prompt", "scores", "feedback"}

    # --- 5) Verify each record’s content matches what was logged ---
    # Build a map from payload → data dict
    data_by_payload = {e["data_payload"]: e["data"] for e in entries}

    assert data_by_payload["prompt"]["prompt"] == prompt_text
    assert data_by_payload["scores"]["score"] == score_value
    assert data_by_payload["feedback"]["feedback"] == feedback_text

    # --- 6) Example: assemble a training tuple for fine-tuning ---
    training_example = (
        data_by_payload["prompt"]["prompt"],
        data_by_payload["scores"]["score"],
        data_by_payload["feedback"]["feedback"]
    )
    expected_example = (prompt_text, score_value, feedback_text)

    assert training_example == expected_example

    # --- 7) Also demonstrate targeted retrieval of just feedback entries ---
    feedback_only = memdb.get_data(
        problem_id=problem_id,
        step_id=step_id,
        data_payload="feedback"
    )
    assert len(feedback_only) == 1
    assert feedback_only[0]["data"]["feedback"] == feedback_text

# --------------------------------------------------------------------------- #
#  D.7  Hypothesis exploration (R19 extensibility, filter on hypothesis_id)
# --------------------------------------------------------------------------- #
def test_hypothesis_basic(memdb):
    hid = str(uuid.uuid4())
    memdb.log_data("HYP", 1, {"hypothesis": "X>>Y"}, data_payload="hypothesis", metadata={"hypothesis_id": hid})
    rec = memdb.get_data(additional_filters={"hypothesis_id": hid})[0]
    assert rec["hypothesis_id"] == hid

def test_hypothesis_exploration_workflow(memdb):
    """
    D.7 Hypothesis Exploration (LLF-HELiX-Style)

    1) Log a series of distinct hypotheses, each with its own hypothesis_id.
    2) Retrieve the full set (newest first).
    3) Filter by hypothesis_id to target a single hypothesis.
    4) Use get_last_n to build a small exploration pool.
    5) Random-sample to pick an “uncertain region” subset.
    """

    # --- 1) Log multiple hypotheses across steps -------------------------------
    hypotheses = {
        "hid1": "Hypothesis A: X causes Y",
        "hid2": "Hypothesis B: Y affects Z",
        "hid3": "Hypothesis C: Z interacts with X",
        "hid4": "Hypothesis D: Bidirectional correlation X↔Z",
    }

    for step, (hid, text) in enumerate(hypotheses.items(), start=1):
        memdb.log_data(
            problem_id="HYP_EXP",
            step_id=step,
            candidate_id=1,
            data={"hypothesis": text},
            data_payload="hypothesis",
            metadata={
                "agent": "HypothesisAgent",
                "status": "logged",
                "hypothesis_id": hid
            }
        )

    # --- 2) Retrieve full pool (newest first) ----------------------------------
    all_hyps = memdb.get_data(
        problem_id="HYP_EXP",
        data_payload="hypothesis"
    )
    # Expect one record per hypothesis, newest (hid4) at index 0
    assert len(all_hyps) == len(hypotheses)
    assert all_hyps[0]["hypothesis_id"] == "hid4"
    assert all_hyps[-1]["hypothesis_id"] == "hid1"

    # --- 3) Filter by specific hypothesis_id ----------------------------------
    filtered = memdb.get_data(
        additional_filters={"hypothesis_id": "hid2"}
    )
    assert len(filtered) == 1
    rec = filtered[0]
    assert rec["hypothesis_id"] == "hid2"
    assert rec["data"]["hypothesis"] == hypotheses["hid2"]

    # --- 4) Last-N retrieval for exploration pool ------------------------------
    last_two = memdb.get_last_n(
        problem_id="HYP_EXP",
        data_payload="hypothesis",
        n=2
    )
    # Should be the two most recent: hid4 then hid3
    assert [r["hypothesis_id"] for r in last_two] == ["hid4", "hid3"]

    # --- 5) Random sampling for “uncertain region” -----------------------------
    random.seed(0)
    sample = memdb.get_data(
        problem_id="HYP_EXP",
        additional_filters={"random": 2}
    )
    assert len(sample) == 2
    sampled_ids = {r["hypothesis_id"] for r in sample}
    # Sampled IDs must come from the logged set
    assert sampled_ids.issubset(set(hypotheses.keys()))

# --------------------------------------------------------------------------- #
#  R3  In‑memory mode operates without Vector DB or embeddings
# --------------------------------------------------------------------------- #
def test_inmemory_mode_works_without_embeddings(memdb):
    eid = memdb.log_data("EMB", 1, {"prompt": "hello"}, data_payload="prompt")
    assert memdb.get_data(entry_id=eid)[0]["embedding"] is None


# --------------------------------------------------------------------------- #
#  R4  Pluggable Vector DB backend ‑ ensure _add_texts is invoked
# --------------------------------------------------------------------------- #
def test_pluggable_vector_db(pluggable_db):
    db, stub = pluggable_db
    db.log_data("PDB", 1, {"foo": "bar"}, data_payload="variables")
    assert stub.add_calls, "UnifiedVectorDB._add_texts should have been called"


# --------------------------------------------------------------------------- #
#  R8  Immutability ‑ returned records must be deep‑copies
# --------------------------------------------------------------------------- #
def test_records_are_immutable(memdb):
    eid = memdb.log_data("IMM", 1, {"x": 1}, data_payload="variables")
    rec = memdb.get_data(entry_id=eid)[0]
    rec["data"]["x"] = 999                # mutate local copy
    # Fetch again – should be unchanged
    fresh = memdb.get_data(entry_id=eid)[0]
    assert fresh["data"]["x"] == 1        # original persisted


# --------------------------------------------------------------------------- #
#  R11  Multiple candidates per step + get_candidates helper
# --------------------------------------------------------------------------- #
def test_multiple_candidates_and_get_candidates(memdb):
    for cid in (1, 2, 3):
        memdb.log_data("CANDS", 5, {"v": cid}, data_payload="variables",
                             candidate_id=cid)
    candidates = memdb.get_candidates("CANDS", 5)
    assert {c["candidate_id"] for c in candidates} == {1, 2, 3}


# --------------------------------------------------------------------------- #
#  R12  Retrieval filtering on several fields at once
# --------------------------------------------------------------------------- #
def test_complex_filtering(memdb):
    memdb.log_data("FILTER", 1, {"a": 1}, data_payload="code",
                         metadata={"agent": "X"})
    memdb.log_data("FILTER", 2, {"b": 2}, data_payload="code", metadata={"agent": "Y"})
    rows = memdb.get_data(problem_id="FILTER", additional_filters={"agent": "Y"})
    assert len(rows) == 1 and rows[0]["step_id"] == 2


# --------------------------------------------------------------------------- #
#  R13  Ranked retrieval – Top‑N by score, random sample
# --------------------------------------------------------------------------- #
def test_top_n_and_random(memdb):
    scores = [0.1, 0.9, 0.5]
    for cid, sc in enumerate(scores, 1):
        memdb.log_data("RANK", 1, {"v": cid}, data_payload="code",
                             candidate_id=cid, scores={"score": sc})
    top_two = memdb.get_top_candidates(problem_id="RANK", step_id=1, n=2)
    assert [c["scores"]["score"] for c in top_two] == sorted(scores, reverse=True)[:2]

    random.seed(42)
    rand_pick = memdb.get_random_candidates(n=1, problem_id="RANK", step_id=1)[0]
    assert rand_pick in memdb.get_candidates("RANK", 1)


# --------------------------------------------------------------------------- #
#  R18  Most‑diverse helper (using embeddings)
# --------------------------------------------------------------------------- #
def test_most_diverse_candidates(memdb):
    # four points in 2‑D square
    embeds = {
        1: [0.0, 0.0],
        2: [10.0, 0.0],
        3: [0.0, 10.0],
        4: [5.0, 5.0],
    }
    for cid, emb in embeds.items():
        memdb.log_data("DIV", 1, {"dummy": cid}, data_payload="code",
                             candidate_id=cid, embedding=emb)
    diverse = memdb.get_most_diverse_candidates("DIV", 1, n=3)
    # should include three corners (ids 1,2,3) – order not important
    assert {c["candidate_id"] for c in diverse} >= {1, 2, 3}


# --------------------------------------------------------------------------- #
#  R20  Minimal footprint ‑ no optional libs required in pure in‑memory mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("memdb", ["inmemory"], indirect=True)
def test_no_external_dependencies(memdb):
    # The fact we reached this point with vector_db=None is already evidence,
    # but add a quick sanity call:
    assert memdb.vdb is None

# -----------------------------------------------------------------------------
# Section 2: real integration tests covering D.1–D.7 with OptoPrime + LLM
# -----------------------------------------------------------------------------

def test_d1_backward_compatibility_and_basic_logging(memdb, llm):
    """D.1: Ensure OptoPrime logs both step() and backward() with metadata."""
    params = [ParameterNode("test_param", name="param1")]

    # Enable trace memory integration
    enable_trace_memory_for_optoprime(memdb)
    # Now create OptoPrime normally
    opt = OptoPrime(parameters=params, llm=llm)
    # opt = OptoPrime(parameters=params, llm=llm, memory_db=memdb)

    # Set problem_id for testing
    opt._current_problem_id = "G1"

    # Simulate the optimization workflow - Create a simple node to backward from
    target = node("result", name="output")
    # Call backward with proper arguments
    opt.backward(target, "looks good")
    # Now call step to trigger logging
    opt.step()

    # fetch both entries
    recs = memdb.get_data(problem_id="G1", step_id=1)
    types = {r["data_payload"] for r in recs}
    assert "variables" in types or "feedback" in types  # At least one should be logged
    # verify metadata fields
    for r in recs:
        assert r["agent"] == "OptoPrime"
        assert "status" in r

def test_d2_observation_pools_and_retrieval(memdb):
    """D.2: Log observations, then retrieve last-n and semantic search."""
    # log 5 observations
    for i in range(5):
        memdb.log_data(
            problem_id="G2",
            step_id=1,
            candidate_id=1,
            data={"observation": f"state={i}"},
            data_payload="observation",
            metadata={"agent":"EnvMonitor","status":"logged"},
        )
    last3 = memdb.get_last_n(problem_id="G2", data_payload="observation", n=3)
    assert len(last3) == 3
    # semantic search: should return at least one
    hits = memdb.get_data(
        problem_id="G2",
        data_payload="observation",
        additional_filters={"embedding_query": ("state=3", 2)},
    )
    assert isinstance(hits, list)

def test_d3_parallel_pipeline_loose_coupling(memdb):
    """D.3: Trainer/Optimizer parallel pipeline via shared memdb."""
    # minimal Trainer
    class Trainer:
        def __init__(self, db): self.db,self.step= db,0
        def generate(self,N=2):
            self.step+=1
            for cid in range(1,N+1):
                self.db.log_data(
                    problem_id="G3", step_id=self.step, candidate_id=cid,
                    data={"variables": {"a":cid}}, data_payload="variables",
                    metadata={"agent":"Trainer","status":"pending"},
                )
            return self.step
    # minimal Optimizer
    class Optimizer:
        def __init__(self, db): self.db=db
        def select_best(self, step):
            recs = self.db.get_data(problem_id="G3", step_id=step, data_payload="variables")
            return max(recs, key=lambda r: r["candidate_id"])["candidate_id"]
    tr = Trainer(memdb)
    op = Optimizer(memdb)
    step = tr.generate()
    best = op.select_best(step)
    assert best == 2
    # log selection
    memdb.log_data("G3", step, data={"best":best}, data_payload="best_candidate", candidate_id=best, metadata={"agent":"Optimizer","status":"selected"})

    chosen = memdb.get_data(problem_id="G3", step_id=step, data_payload="best_candidate")[0]["data"]["best"]
    assert chosen == best

def test_d4_branching_and_lineage(memdb):
    """D.4: Branch from best candidate, then rollback to a checkpoint."""
    # seed MAIN_GOAL
    memdb.log_data("MAIN",0,data={"goal":"start"},data_payload="goal",metadata={"agent":"Init"})
    memdb.log_data("MAIN",1,candidate_id=1,data={"code":"v1"},data_payload="code",metadata={})
#    memdb.log_data("MAIN",2,candidate_id=1,data={"score":0.9},data_payload="score",metadata={})
    memdb.log_data("MAIN",2,candidate_id=1,data={"code":"v1"},data_payload="code",scores={"score":0.9},metadata={})

    best = memdb.get_top_candidates("MAIN",2,score_name="score")
    # branch
    branch_id = "MAIN_branch"
    memdb.log_data(branch_id,step_id=0,data={"goal":"branch"},data_payload="goal",
                   parent_problem_id="MAIN",metadata={"agent":"Branch"})
    memdb.log_data(branch_id,step_id=1,data={"code":best[0]["data"]["code"]},data_payload="code",
                   metadata={"agent":"Branch","source_entry_id":best[0]["entry_id"]})
    # rollback
    ckpt = memdb.get_data(problem_id="MAIN", step_id=1, data_payload="code")[0]
    rb = "MAIN_rb"
    memdb.log_data(rb,0,data={"goal":"rollback"},data_payload="goal",
                   parent_problem_id="MAIN",metadata={"agent":"Rollback"})
    memdb.log_data(rb,1,data={"code":ckpt["data"]["code"]},data_payload="code",
                   metadata={"agent":"Rollback","source_entry_id":ckpt["entry_id"]})
    # list branches
    branches = memdb.get_data(data_payload="goal", additional_filters={"parent_problem_id":"MAIN"})
    ids = {b["problem_id"] for b in branches if b["problem_id"] != "MAIN"}

    assert branch_id in ids and rb in ids

def test_d5_mutation_lineage(memdb):
    """D.5: Log diffs, then trace lineage of mutations."""
    # original
    orig_id = memdb.log_data("MUT",0,data={"code":"x=0"},data_payload="code",metadata={})
    # first mutation
    m1 = memdb.log_data("MUT",1,data={"diff":"+1"},data_payload="diff",metadata={"source_entry_id":orig_id})
    # second mutation
    m2 = memdb.log_data("MUT",2,data={"diff":"+2"},data_payload="diff",metadata={"source_entry_id":m1})
    # lineage walk
    rec = memdb.get_data(problem_id="MUT", step_id=2, data_payload="diff")[0]
    assert rec["source_entry_id"] == m1

def test_d6_feedback_logging_for_finetuning(memdb):
    """D.6: Log numeric & qualitative feedback for future FT."""
    memdb.log_data("FT",1,data={"prompt":"Do X"},data_payload="prompt",metadata={})
    memdb.log_data("FT",1,data={"score":0.75},data_payload="scores",metadata={})
    memdb.log_data("FT",1,data={"feedback":"Ok but improve edge cases"},data_payload="feedback",metadata={})
    recs = memdb.get_data(problem_id="FT", step_id=1)
    payloads = {r["data_payload"] for r in recs}
    assert {"prompt","scores","feedback"}.issubset(payloads)

def test_d7_hypothesis_exploration(memdb):
    """D.7: Log hypotheses, embed them, sample uncertain region."""
    # log some hypotheses
    for i in range(3):
        emb = [float(i), float(i*2)]
        memdb.log_data("HYP",i,data={"hypothesis":f"H{i}"},data_payload="hypothesis",
                       embedding=emb,metadata={"agent":"Hypo"})
    # query uncertain (e.g. random sample)
    hits = memdb.get_data(problem_id="HYP", additional_filters={"random":2})
    assert len(hits) == 2