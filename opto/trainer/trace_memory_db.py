# trace_memory_db.py
from __future__ import annotations

import json
import random
import math
import copy
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

import socket

class TraceMemoryDB:
    """Structured, cache‑backed logging layer on top of UnifiedVectorDB."""

    # ─── Set of allowed data‐keys ─────────────────────────────
    CANONICAL_DATA_TYPES = {
        "goal","prompt","code","diff_patch","graph_nodes","score","scores",
        "feedback","error","validation_result","hypothesis","observation",
        "checkpoint","context","variables"
    }

    # --------------------------------------------------------------------- #
    # Construction / hot‑cache                                              #
    # --------------------------------------------------------------------- #
    def __init__(self, *, vector_db: Optional[UnifiedVectorDB] = None,
                 cache_size: int = 1000, auto_vector_db: bool = False):
        """
        Args:
            vector_db: existing UnifiedVectorDB instance or *None* for
                       in‑memory‑only operation (R3).
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
        self._cache_order: List[str] = []
        self._cache_size = int(cache_size)

    # --------------------------------------------------------------------- #
    # Public logging / retrieval API                                        #
    # --------------------------------------------------------------------- #
    def log_data(
        self,
        goal_id: str,
        step_id: int,
        data: Dict[str, Any],
        *,
        data_payload: Optional[str] = None,
        candidate_id: int = 1,
        parent_goal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
        scores: Optional[Dict[str, float]] = None,
        feedback: Optional[str] = None,
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

        # ---- (Immutable) ID ------------------------------------------------
        entry_id = f"{goal_id}_{step_id}_{candidate_id}_{data_payload}_{uuid.uuid4().hex[:6]}"

        # ---- Build record dict --------------------------------------------
        record = {
            "entry_id": entry_id,
            "goal_id": goal_id,
            "step_id": int(step_id),
            "candidate_id": int(candidate_id),
            "parent_goal_id": parent_goal_id,
            "data_payload": data_payload,
            "data": data,
            "metadata": metadata or {},
            "embedding": embedding,
            "scores": scores,
            "feedback": feedback,
        }
        # timestamp in metadata
        record["metadata"].setdefault("timestamp", datetime.utcnow().isoformat())

        # ---- Upsert in process memory -------------------------------------
        if update_if_exists:
            # locate first matching immutable record & overwrite (rare)
            for idx, r in enumerate(self._store):
                if (r["goal_id"], r["step_id"], r["candidate_id"], r["data_payload"]) == (
                    goal_id, step_id, candidate_id, data_payload
                ):
                    self._store[idx] = record
                    break
            else:
                self._store.append(record)
        else:
            self._store.append(record)

        # ---- Hot cache maintenance ----------------------------------------
        self._hot_cache[entry_id] = record
        self._cache_order.append(entry_id)
        if len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            self._hot_cache.pop(oldest, None)

        # ---- Persist to Vector‑DB (if available) --------------------------
        if self.vdb is not None:
            meta = {
                "goal_id": goal_id,
                "step_id": step_id,
                "candidate_id": candidate_id,
                "data_payload": data_payload,
                **(metadata or {}),
            }
            self.vdb._add_texts([json.dumps(record)], ids=[entry_id], metadatas=[meta])

        return entry_id

    # ..................................................................... #
    def get_data(
        self,
        *,
        goal_id: Optional[str] = None,
        step_id: Optional[int] = None,
        data_type: Optional[str] = None,
        data_payload: Optional[str] = None,  # alias for backward-compat
        candidate_id: Optional[int] = None,
        entry_id: Optional[str] = None,
        parent_goal_id: Optional[str] = None,
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
            if goal_id is not None and rec["goal_id"] != goal_id:
                return False
            if step_id is not None and rec["step_id"] != step_id:
                return False
            if data_type is not None and rec.get("data_payload") != data_type:
                return False
            if candidate_id is not None and rec["candidate_id"] != candidate_id:
                return False
            if parent_goal_id is not None and rec["parent_goal_id"] != parent_goal_id:
                return False
            # nested path look‑ups, e.g. "metadata.agent"
            for key, expected in additional_filters.items():
                # Special case: if filtering by parent_goal_id in metadata but record has it as top-level
                if key == "metadata.parent_goal_id" and rec.get("parent_goal_id") == expected:
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
    def get_last_n(self, goal_id: str, data_type: str, n: int) -> List[Dict]:
        """Return newest N records for that goal / payload."""
        return self.get_data(goal_id=goal_id, data_type=data_type, last_n=n)

    def get_candidates(self, goal_id: str, step_id: int) -> List[Dict]:
        """All candidate entries at this (goal, step)."""
        return self.get_data(goal_id=goal_id, step_id=step_id)

    def get_best_candidate(self, goal_id: str, step_id: int, score_name: str = "score") -> Optional[Dict]:
        """Return candidate with highest <score_name> inside .scores."""
        candidates = self.get_candidates(goal_id, step_id)
        scored: List[tuple[float,dict]] = []
        for c in candidates:
            # look for either data["scores"] or data["score"]
            d = c["data"]
            val = None
            if isinstance(d.get("scores"), dict):
                val = d["scores"].get(score_name)
            elif isinstance(d.get("score"), (int, float)):
                val = d["score"]
            if isinstance(val, (int, float)):
                scored.append((val, c))
        if not scored:
            return None
        _, best_rec = max(scored, key=lambda x: x[0])
        return best_rec

    # ------------------------------------------------------------------ #
    #  Ranked / stochastic / diversity helpers  (R13 & R18)
    # ------------------------------------------------------------------ #
    def get_top_candidates(
        self, goal_id: str, step_id: int, score_name: str = "score", n: int = 3
    ) -> List[Dict]:
        """Return Top‑N candidates by descending <score_name> (R13)."""
        cands = self.get_candidates(goal_id, step_id)
        scored: List[tuple[float,dict]] = []
        for c in cands:
            d = c["data"]
            val = None
            if isinstance(d.get("scores"), dict):
                val = d["scores"].get(score_name)
            elif isinstance(d.get("score"), (int, float)):
                val = d["score"]
            if isinstance(val, (int, float)):
                scored.append((val, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [copy.deepcopy(c) for _, c in scored[:n]]

    def get_random_candidates(self, goal_id: str, step_id: int, n: int = 1) -> List[Dict]:
        """Return a random sample of candidates (R13)."""
        cands = self.get_candidates(goal_id, step_id)
        if not cands:
            return []
        return random.sample(cands, min(n, len(cands)))

    def get_most_diverse_candidates(
        self, goal_id: str, step_id: int, n: int = 3
    ) -> List[Dict]:
        """
        Simple farthest‑first traversal using candidate embeddings.
        Falls back to random if embeddings are unavailable.  (R18)
        """
        candidates = self.get_candidates(goal_id, step_id)
        # Keep only those with an embedding vector
        emb_cands = [(c, c.get("embedding")) for c in candidates if isinstance(c.get("embedding"), list)]
        if len(emb_cands) < n:
            return self.get_random_candidates(goal_id, step_id, n)

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
            selected.append(best_idx)

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

    # ─── New method: log_agent_data ────────────────────────────────────────────────
    def log_agent_data(
        self,
        agent_name: str,
        data_key: str,
        data_value: Any,
        function_name: Optional[str] = None,
        task_id: bool = False,
        before_after: Optional[str] = None,
        user_id: Optional[str] = None,
        step_id: Optional[int] = None,
        task_type: Optional[str] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store agent‐specific data as a new document in the vector DB.  Serializes data_value to JSON,
        builds a metadata dict (including agent_name, data_key, and any provided tags), and then calls add_texts().
        """
        # (1) Serialize the payload
        if isinstance(data_value, dict):
            serialized_data = json.dumps(data_value)
        else:
            # wrap primitive or list under a key of data_key
            serialized_data = json.dumps({data_key: data_value})

        # (2) Optionally generate an task_id UUID
        task_id = str(uuid.uuid4()) if task_id else None

        # (3) Build metadata tags
        tags: Dict[str, Any] = metadata.copy() if isinstance(metadata, dict) else {}
        params = {"agent_name": agent_name, "data_key": data_key, "function_name":function_name, "task_id": task_id, "before_after": before_after, "user_id": user_id, "step_id": step_id, "task_type": task_type, "score": score}
        # filter out any that are None
        for key, val in params.items():
            if val is not None:
                tags[key] = val
        # Always tag with a timestamp
        tags["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # (4) Delegate to add_texts()
        # Note: Chroma / Elasticsearch both accept `metadatas=[{…}]`.
        self._add_texts(texts=[serialized_data], metadatas=[tags])


    # ─── New method: get_agent_data ────────────────────────────────────────────────
    def get_agent_data(
        self,
        agent_name: Optional[str] = None,
        data_key: Optional[str] = None,
        task_id: Optional[str] = None,
        function_name: Optional[str] = None,
        before_after: Optional[str] = None,
        user_id: Optional[str] = None,
        step_id: Optional[int] = None,
        task_type: Optional[str] = None,
        score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        sort_order: Optional[str] = None,
        k: int = 5,
        start_index: int = 0,
        end_index: Optional[int] = None,
        query_text: str = "*",
        new_storage: bool = True
    ):
        """
        Retrieve agent‐specific data from the vector DB.  Builds a metadata dict from all provided parameters,
        runs a `query(...)` under the hood, then paginates and parses results (JSON‐decoding payloads).

        Returns:
            parsed_list: list of deserialized payloads (dict or primitive)  
            raw_list:   list of full `Document`‐like objects returned by the underlying vector store
        """
        # (1) Build metadata dict to filter by
        filters: Dict[str, Any] = {}
        # params = ["agent_name", "data_key", "function_name", "task_id", "before_after", "user_id", "step_id", "task_type", "score"]
        # filters.update({k: locals()[k] for k in params if (k in locals()) and (locals()[k] is not None)})
        params = {"agent_name": agent_name, "data_key": data_key, "function_name":function_name, "task_id": task_id, "before_after": before_after, "user_id": user_id, "step_id": step_id, "task_type": task_type, "score": score}
        # filter out any that are None
        for key, val in params.items():
            if val is not None:
                filters[key] = val

        # Merge in any explicit metadata_filter
        if metadata_filter:
            filters.update(metadata_filter)

        # (2) Determine how many to fetch
        max_k = end_index if (end_index is not None) else k

        # (3) Query the DB
        results = self._query( query_text=query_text, k=max_k, metadata_filter=filters, sort_order=sort_order)

        # (4) Apply pagination
        paginated = (results[start_index:end_index] if end_index is not None else results[start_index:])

        # (5) Parse/deserialize each Document.page_content
        parsed_list: List[Any] = []
        for item in paginated:
            try:
                # `item.page_content` may be a JSON string
                temp = json.loads(item.page_content)
            except Exception:
                # fallback: return the raw page_content
                temp = item.page_content
            if new_storage:
                parsed_list.append(temp)
            else:
                # If caller wants the “old-style” raw data_key value:
                if isinstance(temp, dict) and (data_key in temp): #if isinstance(temp, dict) and (data_key in temp) and (len(temp) == 1):
                    parsed_list.append(temp[data_key])
                else:
                    parsed_list.append(None)

        return parsed_list, results

