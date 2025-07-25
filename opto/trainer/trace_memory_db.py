# trace_memory_db.py
from __future__ import annotations

import json
import random
import math
import copy
import uuid
from datetime import datetime
import heapq
from typing import Dict, List, Optional, Any, Union

import socket

class TraceMemoryDB:
    """Structured, cache‑backed logging layer on top of UnifiedVectorDB."""

    # ─── Set of allowed data‐keys ─────────────────────────────
    CANONICAL_DATA_TYPES = {
        "objective", # default_objective or custom_prompt for the given optimization/problem (i.e. managed by Optimizers class instance)
        "variables","constraints","inputs","outputs","others","feedback","instruction","code","documentation","user_prompt", # typical data used by OptoPrime's prompt
        "reasoning","suggestion","answer", # typical data used by OptoPrime's response
        "candidates", # list of candidate answers each as dict {id:, candidate:, optional embedding:, scores:, feedback:} => each candidate should be stored separately (OptoPrimeMulti => will allow to simply backtrack and branch)
        "candidate",  # individual candidate string output selected
        "graph",
        "score", "scores", # Guide's score or a dict of scores / "feedback" is already listed in typical data from optimizer's prompt
        "diff_patch","error","validation_result", # if automatic validation/correction are used, also if we want to capture errors with context
        "hypothesis", # from reasoning analysis for future hypothesis exploration (could be stored as a list of tuples with embeddings or ID linkin to an external vector db for faster search into hypothesis?)
        "checkpoint", # to save state of the optimization problem instance to allow backtracking/branching
        "context",
        "process_state", # to store the state of the process for handing over to another process or for resuming later
    }

    # --------------------------------------------------------------------- #
    # Construction / hot‑cache                                              #
    # --------------------------------------------------------------------- #
    def __init__(self, *, vector_db: Optional[UnifiedVectorDB] = None,
                 cache_size: int = 1000, auto_vector_db: bool = False):
        """
        Args:
            vector_db: existing UnifiedVectorDB instance or *None* for in‑memory‑only operation (R3).
            cache_size: number of recent records held in RAM (hot cache).
        """
        # Minimal‑footprint default: keep vdb = None unless the caller
        # explicitly supplied one *or* set auto_vector_db=True.
        if vector_db is not None:
            self.vdb = vector_db
        elif auto_vector_db:
            cfg = UnifiedVectorDBConfig(reset_indices=False)
            self.vdb = UnifiedVectorDB(cfg, check_db=False)  # type: ignore
        else:
            self.vdb = None

        self._store: List[Dict[str, Any]] = []     # immutable append‑only log
        self._hot_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_heap: List[tuple[float, str]] = []
        self._cache_size = int(cache_size)

    def __len__(self) -> int:
        """Return the total number of records stored."""
        return len(self._store)
    
    def __iter__(self):
        """Iterate over all stored records."""
        for record in self._store:
            yield record

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Retrieve a specific record by index."""
        return self._store[index]

    # --------------------------------------------------------------------- #
    # Public logging / retrieval API                                        #
    # --------------------------------------------------------------------- #
    def log_data(
        self,
        problem_id: str,
        step_id: int,
        data: Dict[str, Any],
        *,
        data_payload: Optional[str] = None,
        candidate_id: int = 1,
        parent_problem_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None, # TODO: handle the possibility to directly pass an embedding vector
        scores: Optional[Dict[str, float]] = None,
        feedback: Optional[str] = None,
        process_state: Optional[str] = None,
        update_if_exists: bool = False,
        **legacy_aliases,
    ) -> str:
        """
        Append a new, immutable record.  Aliases:
            data_key → data_payload   (for backward‑compat D.1)
        """
        # ---- Backward compatibility (D.1) --------------------------------
        if data_payload is None and "data_key" in legacy_aliases:
            data_payload = legacy_aliases.pop("data_key")
        if data_payload is None:
            raise ValueError("`data_payload` (or legacy `data_key`) is required")
        # ── if this is a batch of candidates, unroll it ──
        if isinstance(data.get("candidates"), list):
            items = data.pop("candidates")
            return [
                self.log_data( problem_id=problem_id, step_id=step_id, data=data, data_payload=data_payload, candidate_id=cand.get("id",i),
                    parent_problem_id=parent_problem_id, metadata=metadata, embedding=cand.get("embedding"), scores=cand.get("scores"),
                    feedback=cand.get("feedback"), update_if_exists=update_if_exists,
                    **legacy_aliases,
                )
                for i, cand in enumerate(items)
            ]
        # ---- (Immutable) ID ------------------------------------------------
        entry_id = f"{problem_id}_{step_id}_{candidate_id}_{data_payload}"

        # ---- Build record dict --------------------------------------------
        record = {
            "entry_id": entry_id,
            "data": data,
            "embedding": embedding, # TODO: not sustainable, to be fixed
            "scores": scores,
            "feedback": feedback,
        }

        meta = {
            "problem_id": problem_id,
            "step_id": step_id,
            "candidate_id": candidate_id,
            "data_payload": data_payload,
            "process_state": process_state,
            'timestamp': datetime.utcnow().isoformat(),
            **(metadata or {}),
        }
        if parent_problem_id:
            meta["parent_problem_id"] = parent_problem_id
            
        # ---- Persist to Vector‑DB (if available) --------------------------
        if self.vdb is not None:
            self.vdb._add_texts([json.dumps(record)], ids=[entry_id], metadatas=[meta])

        # Merge meta into record for easy access
        record.update(meta)
        priority = None
        # Check for score in data
        if isinstance(data.get("score"), (int, float)):
            priority = -float(data["score"])  # Negate for max-heap behavior
        # Check for scores dict
        elif (isinstance(scores, dict) and len(scores) > 0) or (isinstance(data.get("scores"), dict) and len(data.get("scores", {})) > 0):
            # Use first available score
            priority = -float(next(iter((scores or data["scores"]).values())))
        # Fallback to timestamp (FIFO)
        priority = priority if priority is not None else datetime.fromisoformat(meta["timestamp"]).timestamp()
 
        # ---- Upsert in process memory -------------------------------------
        if update_if_exists:
            # locate first matching immutable record & overwrite (rare)
            for idx, r in enumerate(self._store):
                if (r.get("problem_id"), r.get("step_id"), r.get("candidate_id"), r.get("data_payload")) == (
                    problem_id, step_id, candidate_id, data_payload
                ):
                    self._store[idx] = record
                    break
            else:
                self._store.append(record)
        else:
            self._store.append(record)

        # ---- Hot cache maintenance ----------------------------------------
        self._hot_cache[entry_id] = record

        heapq.heappush(self._cache_heap, (priority, entry_id))
        
        # Evict lowest priority entries if cache exceeds size
        while len(self._hot_cache) > self._cache_size:
            # Pop entries until we find one still in cache
            while self._cache_heap:
                _, evict_id = heapq.heappop(self._cache_heap)
                if evict_id in self._hot_cache:
                    self._hot_cache.pop(evict_id)
                    break

        # Evict oldest entries if cache exceeds size
        while len(self._hot_cache) > self._cache_size:
            # Pop entries until we find one still in cache
            while self._cache_heap:
                _, oldest_id = heapq.heappop(self._cache_heap)
                if oldest_id in self._hot_cache:
                    self._hot_cache.pop(oldest_id)
                    break

        return entry_id

    # ..................................................................... #
    def get_data(
        self,
        *,
        problem_id: Optional[str] = None,
        step_id: Optional[int] = None,
        data_type: Optional[str] = None,
        data_payload: Optional[str] = None,  # alias for backward-compat
        candidate_id: Optional[int] = None,
        entry_id: Optional[str] = None,
        parent_problem_id: Optional[str] = None,
        last_n: Optional[int] = None,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve matching records.  You can filter by any top-level field, or by `data_type` (requires that key exist in rec["data"]).
        """
        if data_payload is not None: data_type = data_payload
        additional_filters = dict(additional_filters or {})  # mutable copy
        # Semantic vector query (ignored in pure in‑memory mode)
        additional_filters.pop("embedding_query", None)
        # Random sampling request
        rand_n = additional_filters.pop("random", None)

        def _match(rec: Dict[str, Any]) -> bool:
            if problem_id is not None and rec.get("problem_id") != problem_id:
                return False
            if step_id is not None and rec.get("step_id") != step_id:
                return False
            if data_type is not None and rec.get("data_payload") != data_type:
                return False
            if candidate_id is not None and rec.get("candidate_id") != candidate_id:
                return False
            if parent_problem_id is not None and rec.get("parent_problem_id") != parent_problem_id:
                return False
            # nested path look‑ups, e.g. "metadata.agent"
            for key, expected in additional_filters.items():
                # Special case: if filtering by parent_problem_id in metadata but record has it as top-level
                if key == "metadata.parent_problem_id" and rec.get("parent_problem_id") == expected:
                    continue  # This filter matches, continue to check other filters
                path = key.split(".")
                cur = rec
                for p in path:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        return False
                # Enhanced filtering logic for list membership
                if isinstance(expected, list) and isinstance(cur, list):
                    # Check if expected is a subset of cur (all items in expected are in cur)
                    if not all(item in cur for item in expected):
                        return False
                elif cur != expected:
                    return False
            return True

        hits = [copy.deepcopy(r) for r in reversed(self._store) if _match(r)]
        # honour random‑sampling after initial filtering
        if rand_n is not None and hits:
            hits = random.sample(hits, min(int(rand_n), len(hits)))

        if entry_id is not None:
            # shortcut: single record by id
            if entry_id in self._hot_cache:
                return [copy.deepcopy(self._hot_cache[entry_id])]
            return [copy.deepcopy(r) for r in self._store if r["entry_id"] == entry_id]
        return hits[: last_n] if last_n else hits

    # ..................................................................... #
    def get_last_n(self, n: int, problem_id: str = None, data_type: str = None, data_payload: str = None) -> List[Dict]:
        """Return newest N records for that goal / payload."""
        return self.get_data(problem_id=problem_id, data_type=data_type, data_payload=data_payload, last_n=n)

    def get_candidates(self, problem_id: str, step_id: int) -> List[Dict]:
        """All candidate entries at this (goal, step)."""
        return self.get_data(problem_id=problem_id, step_id=step_id)

    # ------------------------------------------------------------------ #
    #  Ranked / stochastic / diversity helpers  (R13 & R18)
    # ------------------------------------------------------------------ #
    def get_top_candidates(
        self, problem_id: str, step_id: int = None, score_name: str = None, score_fn = None, n: int = 3
    ) -> List[Dict]:
        """Return Top‑N candidates by descending <score_name> (R13)."""
        # fast‐path: if no score_name and no custom score_fn, use cache heap directly
        if score_name is None and score_fn is None:
            # cache heap stores (priority, entry_id) where lower priority == higher score
            top = heapq.nsmallest(n, self._cache_heap)
            return [
                copy.deepcopy(self._hot_cache[eid])
                for _, eid in top
                if eid in self._hot_cache
            ]
        cands = self.get_candidates(problem_id, step_id)
        # choose ranking key: custom fn > named score > direct .score
        if score_fn is not None:
            key_fn = lambda c: score_fn(c["data"])
        elif score_name:
            key_fn = lambda c: c["data"].get("scores", c.get("scores", {})).get(score_name, c["data"].get("score"))
        else:
            key_fn = lambda c: c["data"].get("score")
        # filter out non-numeric scores
        valid = [c for c in cands if isinstance(key_fn(c), (int, float))]
        # fast top-N selection
        top_n = heapq.nlargest(n, valid, key=key_fn)
        return [copy.deepcopy(c) for c in top_n]

    def get_random_candidates(self, problem_id: str, step_id: int, n: int = 1) -> List[Dict]:
        """Return a random sample of candidates (R13)."""
        cands = self.get_candidates(problem_id, step_id)
        if not cands:
            return []
        return random.sample(cands, min(n, len(cands)))

    def get_most_diverse_candidates(
        self, problem_id: str, step_id: int, n: int = 3
    ) -> List[Dict]:
        """
        Simple farthest‑first traversal using candidate embeddings.
        Falls back to random if embeddings are unavailable.  (R18)
        """
        candidates = self.get_candidates(problem_id, step_id)
        # Keep only those with an embedding vector
        emb_cands = [(c, c.get("embedding")) for c in candidates if isinstance(c.get("embedding"), list)]
        if len(emb_cands) < n:
            return self.get_random_candidates(problem_id, step_id, n)

        def _dist(a, b):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

        # ---- farthest‑first ----------------------------
        selected = []
        # start with the vector of max norm
        norms = [sum(x * x for x in e) for _, e in emb_cands]
        first_idx = norms.index(max(norms))
        selected.append(first_idx)

        while len(selected) < n:
            best, best_idx = -1.0, None
            for idx, (_, emb) in enumerate(emb_cands):
                if idx in selected:
                    continue
                min_d = min(_dist(emb, emb_cands[s][1]) for s in selected)
                if min_d > best:
                    best, best_idx = min_d, idx
            if best_idx is not None:
                selected.append(best_idx)
            else:
                break  # No more candidates to select

        return [copy.deepcopy(emb_cands[i][0]) for i in selected]

    # ..................................................................... #
    def delete_data(self, entry_id: str) -> bool:
        """Remove from hot‑cache, in‑memory store and vector‑db (best‑effort)."""
        self._hot_cache.pop(entry_id, None)
        self._store = [r for r in self._store if r["entry_id"] != entry_id]
        if self.vdb is not None:
            try:
                self.vdb.delete([entry_id])
            except Exception:  # pragma: no cover
                pass
        return True

    # ..................................................................... #
    #  Check‑point helpers –  tag + serialise
    # ..................................................................... #
    def save_checkpoint(
        self, problem_id: str, step_id: int, *, file_path: str, label: str | None = None
    ):
        """
        Serialise all records for (goal, step) – *including* leaf parameters –
        and additionally **log** a 'checkpoint' entry so Trace’s downstream
        optimisers can find it later.
        """
        rows = self.vdb.query_records(problem_id=problem_id, step_id=step_id)
        with open(file_path, "w") as fh:
            json.dump(rows, fh, indent=2)

        self.log_data(
            problem_id=problem_id,
            step_id=step_id,
            data={"file": file_path},
            data_payload="checkpoint",
            metadata={"label": label or f"ckpt@{step_id}"}
        )

    def load_checkpoint(self, file_path: str, *, new_problem_id: str | None = None):
        rows = json.load(open(file_path))
        for r in rows:
            r["problem_id"] = new_problem_id or r["problem_id"]
            self.vdb.insert_record(r)

# -------------------------------------------------------
# Raw VectorDB interface
# -------------------------------------------------------

if 'ELASTIC_DATABASE' not in globals(): ELASTIC_DATABASE="elasticsearch"
if 'CHROMA_DATABASE' not in globals(): CHROMA_DATABASE="chroma"

import logging
import re
import os
import requests
import time
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.util.retry import Retry  # type: ignore

#––– Utility to produce a “unique collection id” if none is given –––#
def _default_unique_collection_id() -> str:
    """
    Exactly the same fallback used previously in llm_utils.py:
    f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    """
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    return f"{hostname}_{timestamp}"
class UnifiedVectorDBConfig:
    """Configuration for the vector database."""
    common_vectordb_embedding_function = None

    def __init__(
        self,
        embedding_function: Optional[Any] = None,
        collection_name: Optional[str] = "human_llm_logs",
        persist_directory: Optional[str] = "human_llm_vectordb",
        reset_indices: bool = False,
        unique_collection_id: Optional[str] = None
    ):
        """Initialize VectorDBConfig."""
        self.embedding_function = embedding_function
        self.collection_name = collection_name.lower()
        self.persist_directory = persist_directory
        self.reset_indices = reset_indices

        # If the caller gave a unique_collection_id, use it; otherwise call our fallback.
        self.unique_collection_id = unique_collection_id or _default_unique_collection_id()

        # Put a default “db_type” (will be overridden at runtime if needed):
        self.db_type: str = CHROMA_DATABASE

        # Build out an OpenAI or HuggingFace embedding function exactly as llm_utils did:
        self.openai_embedding_function_name = "text-embedding-ada-002"
        self.set_common_vectordb_embedding_function()

        self.es_config = ElasticSearchDB_Config() if self.db_type == ELASTIC_DATABASE else None
        
    def set_common_vectordb_embedding_function(self):
        """Set the embedding function for the vector database."""
        if self.__class__.common_vectordb_embedding_function is not None:
            return
        # only import the embeddings when we need them
        from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

        if isinstance(self.embedding_function, str):
            if self.embedding_function in ["OpenAIEmbeddings", "text-embedding-ada-002"]:
                emb = OpenAIEmbeddings(
                    model=self.embedding_function,
                    deployment=self.openai_embedding_function_name
                )
            elif self.embedding_function == "HuggingFaceEmbeddings":
                emb = HuggingFaceEmbeddings(
                    model_name="intfloat/e5-base-v2",
                    encode_kwargs={"normalize_embeddings": True}
                )
            else:
                emb = HuggingFaceEmbeddings(
                    model_name=self.embedding_function,
                    encode_kwargs={"normalize_embeddings": True},
                    model_kwargs={"trust_remote_code": True}
                )
        else:
            emb = self.embedding_function

        # If no embedding function is provided, create a simple one
        if emb is None:
            # Create a simple embedding function that returns a fixed-size vector
            class SimpleEmbedding:
                def embed_query(self, text):
                    # Simple hash-based embedding for testing
                    import hashlib
                    h = hashlib.md5(text.encode()).hexdigest()
                    # Convert to a vector of floats
                    return [float(int(h[i:i+2], 16))/255.0 for i in range(0, 32, 2)]
                    
                def embed_documents(self, texts):
                    return [self.embed_query(text) for text in texts]
            
            emb = SimpleEmbedding()

        self.__class__.common_vectordb_embedding_function = emb

    def set_unique_collection_id(self, unique_id):
        self.unique_collection_id = unique_id

class ElasticSearchDB_Config:
    def __init__(self):
        try: import config as cfg
        except: cfg = None
        self.es_url: str = 'http://127.0.0.1:9200' if not hasattr(cfg, 'elastic_url_port') else cfg.elastic_url_port
        self.es_user: Optional[str] = None if not hasattr(cfg, 'elastic_user') else cfg.elastic_user
        self.es_password: Optional[str] = None if not hasattr(cfg, 'elastic_password') else cfg.elastic_password

class ChromaConfig:          # expected only by the test-suite
    def __init__(self, persist_directory: str = "chroma_persist", collection: str = "default"):
        """
        :param persist_directory: local directory to store Chroma files.
        :param collection:        name of the Chroma collection.
        """
        self.persist_directory = persist_directory
        self.collection = collection

class UnifiedVectorDB:
    """Unified interface for vector databases (Elasticsearch or Chroma)."""
    db_connection_check_done = False

    def __init__(self, config: Optional[UnifiedVectorDBConfig]=None, check_db:bool=False):
        """Initialize UnifiedVectorDB."""
        # Copy config reference
        self.config: UnifiedVectorDBConfig = config

        # If the caller changed db_type to ELASTIC_DATABASE after config was built, ensure es_config exists
        if self.config.db_type == ELASTIC_DATABASE and self.config.es_config is None:
            self.config.es_config = ElasticSearchDB_Config()

        def friendly_collectionname_string(s):
            # Constraint 1: Truncate or pad the string to ensure it's between 3-63 characters
            s = s[:63].ljust(3, 'a')
            # Constraint 2: Ensure it starts and ends with an alphanumeric character
            if not s[0].isalnum():
                s = 'a' + s[1:]
            if not s[-1].isalnum():
                s = s[:-1] + 'a'
            # Constraint 3: Replace invalid characters with underscores
            s = re.sub(r'[^a-zA-Z0-9_-]', '_', s)
            # Constraint 4: Replace two consecutive periods with underscores
            s = s.replace('..', '__')
            # Constraint 5: Ensure it's not a valid IPv4 address
            if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', s):
                s = 'a' + s[1:]
            return s[:63]
        self.logger = logging.getLogger(__name__)
        
        self.elastic_client: Optional[Any] = None
        self.db: Optional[Any] = None
        self._collection: Optional[Any] = None
        
        if check_db:
            self.check_db()
        self.get_unique_id()

        if self.config.unique_collection_id is not None:
            self.config.collection_name = f"{self.config.unique_collection_id}_{self.config.collection_name}".lower()
        self.config.collection_name = friendly_collectionname_string(self.config.collection_name)

        if self.config.db_type == CHROMA_DATABASE:
            # lazy-load Chroma only now
            try:
                from langchain_chroma import Chroma
            except ImportError:
                raise ImportError("Chroma vector store selected but 'chromadb' or langchain community support is not installed.")
        elif self.config.db_type == ELASTIC_DATABASE:
            # lazy-load Elasticsearch and its store only when needed
            try:
                from elasticsearch import Elasticsearch
                from langchain_community.vectorstores import ElasticsearchStore
            except ImportError:
                raise ImportError("Elasticsearch vector store selected but 'elasticsearch' library or LangChain ES support is not installed.")
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

        if self.config.db_type == CHROMA_DATABASE:
            self.config.persist_directory = friendly_collectionname_string(self.config.persist_directory)
            self.db = Chroma(
                collection_name=self.config.collection_name,
                embedding_function=self.config.common_vectordb_embedding_function,
                persist_directory=self.config.persist_directory
            )
            self._collection = self.db._collection
        elif self.config.db_type == ELASTIC_DATABASE:
            self.elastic_client = Elasticsearch(
                self.config.es_config.es_url,
                http_auth=(
                    self.config.es_config.es_user,
                    self.config.es_config.es_password
                ) if (self.config.es_config.es_user not in [False, "", None]) else None,
                verify_certs=True,
                ssl_show_warn=False
            )
            self.db = ElasticsearchStore(
                index_name=self.config.collection_name,
                embedding=self.config.common_vectordb_embedding_function,
                es_connection=self.elastic_client,
                distance_strategy="COSINE"
            )
            self._collection = self.db
            embedding_test = self.config.common_vectordb_embedding_function.embed_query("test")
            embedding_size = len(embedding_test)
            if self.config.reset_indices:
                self.db.client.indices.delete(
                    index=self.config.collection_name,
                    ignore=[400, 404]
                )
            self.db._create_index_if_not_exists(
                index_name=self.config.collection_name,
                dims_length=embedding_size
            )
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

    def get_unique_id(self):
        """Generate or retrieve a unique ID for the collection."""
        if self.config.unique_collection_id is None:
            self.config.unique_collection_id = os.environ.get('unique_id', f"{socket.gethostname()}_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        return self.config.unique_collection_id

    def check_db(self):
        """Check the database connection."""
        if self.__class__.db_connection_check_done:
            return
        if self.config.db_type == ELASTIC_DATABASE:
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=1)
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            auth = (
                HTTPBasicAuth(self.config.es_config.es_user, self.config.es_config.es_password)
                if self.config.es_config.es_user else None
            )
            try:
                response = session.get(self.config.es_config.es_url, auth=auth, timeout=5, verify=False)
                response.raise_for_status()
                self.logger.info(f"Elasticsearch response: {response.text}")
                self.__class__.db_connection_check_done = True
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error: {e}\nURL: {self.config.es_config.es_url}\nCheck Elasticsearch and credentials.")
                exit(1)
        elif self.config.db_type == CHROMA_DATABASE:
            self.logger.info("Chroma DB check is not yet implemented")
            self.db_connection_check_done = True
        else:
            raise ValueError(f"Unsupported DB type: {self.config.db_type}")

    def _add_texts(self, texts, ids=None, metadatas=None):
        """Add texts to the database."""
        try:
            if isinstance(metadatas, dict):
                metadatas = {k: v for k, v in metadatas.items() if v is not None}
            elif isinstance(metadatas, list):
                metadatas = [{k: v for k, v in (m or {}).items() if v is not None} for m in metadatas]
            else:
                metadatas = None
            if self.config.db_type == CHROMA_DATABASE or self.config.db_type == ELASTIC_DATABASE:
                if isinstance(texts, list):
                    for i in range(len(texts)):
                        if not isinstance(texts[i], str):
                            texts[i] = str(texts[i])
                if metadatas is None:
                    return self.db.add_texts(texts=texts, ids=ids)
                return self.db.add_texts(texts=texts, ids=ids, metadatas=metadatas)
            else:
                self.logger.error(f"Unsupported DB type: {self.config.db_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error adding texts to database: {str(e)} / Texts: {texts} / IDs: {ids} / Metadatas: {metadatas}")
            return None

    def delete(self, ids):
        """Delete entries from the database by IDs."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db.delete(ids=ids)
        elif self.config.db_type == ELASTIC_DATABASE:
            return self.db.delete(ids=ids)

    def _similarity_search_with_score(self, query, k=1):
        """Perform a similarity search with scores."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db.similarity_search_with_score(query, k=k)
        elif self.config.db_type == ELASTIC_DATABASE:
            return self.db.similarity_search_with_score(query, k=(k if k <= 50 else 50))

    def _query(self, query_text="", k=1, metadata_filter=None, metadata_filter_or=False,
              custom_filter_chrome=None, custom_filter_es=None, sort_order=None):
        """Query the database with filters and sorting."""
        if self.config.db_type == CHROMA_DATABASE:
            filter_chroma = None
            if metadata_filter and custom_filter_chrome is None:
                conditions = []
                for key, value in metadata_filter.items():
                    sign = '$eq' if (isinstance(value, str) or isinstance(value, bool)) else '$in'
                    if sign == '$in' and not isinstance(value, (list, tuple)):
                        value = [value]
                    conditions.append({key: {sign: value}})
                # If only one condition, use it directly; otherwise wrap in $and or $or.
                if len(conditions) == 1:
                    filter_chroma = conditions[0]
                else:
                    filter_chroma = {('$or' if metadata_filter_or else '$and'): conditions}
                if sort_order in ['asc', 'desc']:
                    self.logger.warning("WARNING: sort not implemented for Chroma DB; performing in-memory sort")
            # Query the database using the filter (if any)
            # Check db size if the underlying store supports it; otherwise skip
            try:
                size = self.db._collection.count()
                if size == 0:
                    self.logger.warning("WARNING: Chroma DB is empty, returning empty results")
                    return []
                elif size < k:
                    k = size
            except Exception:
                pass
            try:
                results = self.db.similarity_search(query_text, k=k, filter=filter_chroma)
            except Exception as e:
                self.logger.error(f"Error querying Chroma DB: {str(e)}")
                return []
            # If a sort order is provided, sort the results in memory.
            if sort_order in ['asc', 'desc']:
                # Each item may be either Document or (Document, score) - Extract .metadata['time'] from whichever form it is.
                results = sorted( results, key=lambda x: ( x[0].metadata.get('time', "") if isinstance(x, tuple) else x.metadata.get('time', "") ), reverse=(sort_order == 'desc'))
            return results
        elif self.config.db_type == ELASTIC_DATABASE:
            if metadata_filter and "_id" in metadata_filter and not metadata_filter_or:
                # metadata_filter["_id"] might be a single string or a list
                id_values = (metadata_filter["_id"] if isinstance(metadata_filter["_id"], list) else [metadata_filter["_id"]])
                # Build a Document for each requested ID, with that ID in metadata and .id
                from langchain.schema import Document
                results = [Document(page_content="", metadata={"_id": doc_id}, id=doc_id) for doc_id in id_values]
                return results
            if metadata_filter and custom_filter_es is None:
                custom_filter_es = []
                for key, value in metadata_filter.items():
                    if key == '_id':
                        if isinstance(value, list):
                            custom_filter_es.append({"ids": {"values": value}})
                        else:
                            custom_filter_es.append({"ids": {"values": [value]}})
                    elif isinstance(value, dict) and any(k in value for k in ['gte', 'lte', 'gt', 'lt']):
                        custom_filter_es.append({"range": {f"metadata.{key}": value}})
                    elif isinstance(value, list):
                        custom_filter_es.append({"terms": {f"metadata.{key}": value}})
                    else:
                        custom_filter_es.append({"match": {f"metadata.{key}": value}})
                if metadata_filter_or:
                    custom_filter_es = {"bool": {"should": custom_filter_es}}
            if sort_order in ['asc', 'desc']:
                def custom_query(query_body: dict, query: str):
                    return {"query": {"bool": {"must": custom_filter_es}},
                            "sort": [{"metadata.time": {"order": sort_order}}]}
                results = self.db.similarity_search(query_text, k=(k if k <= 50 else 50), custom_query=custom_query)
            else:
                results = self.db.similarity_search(query_text, k=(k if k <= 50 else 50), filter=custom_filter_es)
            # Minimal propagation of _id into Document.id
            for doc in (item[0] if isinstance(item, tuple) else item for item in results):
                if not doc.id and "_id" in doc.metadata: doc.id = doc.metadata["_id"]
            return results

    def count(self):
        """Count the number of entries in the database."""
        if self.config.db_type == CHROMA_DATABASE:
            return self.db._collection.count()
        elif self.config.db_type == ELASTIC_DATABASE:
            response = self.db.client.count(index=self.config.collection_name, body={"query": {"match_all": {}}})
            return response['count']

    def clear(self):
        """Clear the database."""
        if self.config.db_type == CHROMA_DATABASE:
            self.db._collection.clear()
        if self.config.db_type == ELASTIC_DATABASE:
            response = self.db.client.delete_by_query(index=self.config.collection_name, body={"query": {"match_all": {}}})
            self.logger.info(f"Deleted {response['deleted']} documents from index {self.config.collection_name}")
            time.sleep(2)

    # ───────────────────────── Few‑shot helpers (moved from HumanLLMConfig) ──
    def populate_few_shot_tags(self, prompt: str) -> str:
        """
        Expand every `few_shots:{…}` tag found in *prompt* by calling
        :py:meth:`get_few_shot_examples`.  The tag (including its JSON payload)
        is replaced by the generated examples.
        """
        if not prompt:
            return ""

        pattern = r"few_shots:\s*\{"
        for m in reversed(list(re.finditer(pattern, prompt, re.DOTALL))):
            # Find matching closing brace (manual balance → works for nested {})
            depth, i = 1, m.end()
            while depth and i < len(prompt):
                if prompt[i] == "{":
                    depth += 1
                elif prompt[i] == "}":
                    depth -= 1
                i += 1
            try:
                criteria_json = json.loads("{" + prompt[m.end():i - 1] + "}")
            except json.JSONDecodeError:
                continue  # leave tag untouched if JSON is malformed
            replacement = self.get_few_shot_examples([criteria_json])
            prompt = prompt[:m.start()] + replacement + prompt[i:]
        return prompt

    # ---- public: get a formatted block of examples --------------------------------
    def get_few_shot_examples(self, criteria_list) -> str:
        if not criteria_list:
            return ""

        blocks = []
        for spec in self._normalize_criteria(criteria_list):
            src      = spec["sources"]
            num      = spec["num"]
            q_text   = spec.get("query_text", "*")
            m_filter = spec.get("metadata_filter", {})

            docs = self._query(query_text=q_text,
                               k=num,
                               metadata_filter=m_filter,
                               sort_order=spec.get("sort_order"))
            examples = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
            blocks.append(self.format_examples(examples, spec, spec.get("separators")))
        return "\n".join(blocks)

    # ---- helpers few shots -----------------------------------------------------------------
    def _normalize_criteria(self, lst):
        """Ensure required keys & defaults."""
        out = []
        for c in lst:
            out.append({
                "sources":  c.get("sources", "learnt"),
                "num":      c.get("num", 5),
                "format":   c.get("format", "Json"),
                "query_text":        c.get("query_text", "*"),
                "metadata_filter":   c.get("metadata_filter", {}),
                "sort_order":        c.get("sort_order"),
                "similarity_search": c.get("similarity_search", False),
                "template":          c.get("template"),
                "separators":        c.get("separators"),
            })
        return out

    def format_examples(self,
                        examples,
                        criteria,
                        separators=None) -> str:
        """Return examples in **Json**, **Markdown** or **Jinja2** format."""
        if not examples:
            return ""

        fmt       = criteria.get("format", "Json").lower()
        template  = criteria.get("template")
        sep = separators or {
            "global_prefix": "\n<<",
            "global_suffix": ">>\n",
            "item_prefix":   "\n|",
            "item_suffix":   "|"
        }

        rendered = []
        for ex in examples:
            try:
                data = json.loads(ex["content"])
            except Exception:
                data = {"text": ex["content"]}

            if fmt == "json":
                rendered.append(json.dumps(data, indent=2))
            elif fmt == "markdown":
                rendered.append(self._to_markdown(data))
            # elif fmt == "jinja2" and template:
            #     rendered.append(Template(template).render(**data))
            else:
                rendered.append(ex["content"])

        body = "".join(f"{sep['item_prefix']}{t}{sep['item_suffix']}" for t in rendered)
        return f"{sep['global_prefix']}{body}{sep['global_suffix']}"

    @staticmethod
    def _to_markdown(obj):
        return "\n".join(f"**{k}**: {v}" for k, v in obj.items())