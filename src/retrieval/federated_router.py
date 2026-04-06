"""
File: src/retrieval/federated_router.py

Purpose:
Routes queries to domain-specific retrieval indices using LLM-based detection.
Implements federated retrieval with lazy loading and domain awareness.

Dependencies:
from typing import Dict, List, Any, Optional
from src.import_map import IRetriever, RetrievalResult, DocumentSource, RetrievalStrategy, DocumentChunk
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.embedding_model import EmbeddingModel

Implements Interface:
IRetriever (federated router acts as a unified retriever)

Notes:
- LLM-based domain detection (qwen2.5-7b-instruct)
- Lazy loading of per-domain FAISS indices
- Caching of loaded retrievers
- Falls back to keyword matching if LLM fails
- Minimal modifications to existing interfaces
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from src.import_map import DocumentSource, IRetriever, RetrievalResult, RetrievalStrategy
from src.config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, ROUTING_ENABLE_LLM_FALLBACK


logger = logging.getLogger(__name__)


class FederatedRouter(IRetriever):
    """
    Routes queries to domain-specific retrieval indices using LLM-based detection.
    
    Implements:
    - LLM-based domain detection from query text
    - Lazy loading of per-domain retrievers
    - Federated retrieval across multiple sources
    - Compatible with IRetriever interface
    """

    def __init__(
        self,
        embedding_model,
        index_dir: str = "data/vector_store",
        domain_detection_model: str = "qwen/qwen2.5-7b-instruct",
    ) -> None:
        """
        Initialize federated router with LLM domain detection.
        
        Args:
            embedding_model: EmbeddingModel instance for per-domain retrievers
            index_dir: Base directory for per-domain indices
            domain_detection_model: OpenRouter model for domain classification
        """
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.domain_detection_model = domain_detection_model
        
        # Cache for loaded retrievers (lazy loading)
        self.retrievers: Dict[DocumentSource, IRetriever] = {}
        
        # Routing cache to avoid repeated LLM calls for similar queries
        self.routing_cache: Dict[int, List[DocumentSource]] = {}
        
        # Domain keywords for fallback detection
        self.domain_keywords = {
            DocumentSource.PERSONAL_TAX: ["income tax", "itr", "salary", "individual", "personal", "income", "ppf", "80c", "deduction", "hra"],
            DocumentSource.CORPORATE_TAX: ["corporate tax", "company", "business", "corporation", "corporate"],
            DocumentSource.GST: ["gst", "goods", "services", "supply"],
        }
        
        self.logger = logging.getLogger(__name__)
        self.available_sources = self.get_available_sources()

        if ROUTING_ENABLE_LLM_FALLBACK:
            self._init_llm_router()
        else:
            self.domain_llm = None
            self.logger.info("LLM routing fallback disabled; keyword-first routing enabled")

        self.logger.info(f"Active routing sources: {[s.value for s in self.available_sources]}")

    def _init_llm_router(self) -> None:
        """Initialize LLM for domain detection."""
        try:
            self.domain_llm = ChatOpenAI(
                model="mistralai/mistral-7b-instruct",
                openai_api_key=OPENROUTER_API_KEY,
                openai_api_base=OPENROUTER_BASE_URL,
                temperature=0,  # Deterministic
                timeout=30,
            )
            self.logger.info(f"✅ Domain detection LLM initialized: mistralai/mistral-7b-instruct")
        except Exception as e:
            self.logger.warning(f"⚠️  Domain detection LLM initialization failed: {e}. Will use keyword fallback.")
            self.domain_llm = None

    def detect_domains(self, query: str) -> List[DocumentSource]:
        """
        Detect relevant domains from query using LLM.
        
        Args:
            query: User query text
        
        Returns:
            List of detected DocumentSource domains
        """
        if not query:
            return self.available_sources or list(self.domain_keywords.keys())

        # Try LLM-based detection first
        if self.domain_llm:
            try:
                system_prompt = """You are a financial query classifier.
Analyze the query and return ONLY a JSON list of relevant tax domains.
Domains: ["personal_tax", "corporate_tax", "gst"]
Return EXACTLY this format: ["personal_tax"] or ["corporate_tax", "gst"]
Do NOT include explanations."""

                prompt = f"Query: {query}\n\nReturn JSON domains list:"

                response = self.domain_llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ])

                parsed = self._parse_domain_response(response.content)
                if parsed:
                    filtered = [domain for domain in parsed if domain in self.available_sources]
                    if filtered:
                        self.logger.debug(f"✅ LLM detected domains: {filtered}")
                        return filtered
            except Exception as e:
                self.logger.debug(f"⚠️  LLM detection failed: {e}. Falling back to keywords.")

        # Fallback: keyword-based detection
        return self._detect_domains_keyword_fallback(query)

    def _parse_domain_response(self, response_text: str) -> Optional[List[DocumentSource]]:
        """
        Safely parse LLM response to domain list.
        
        Args:
            response_text: Raw LLM response
        
        Returns:
            List of DocumentSource or None if parsing fails
        """
        try:
            # Extract JSON from response (may contain extra text)
            import re
            match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if not match:
                return None

            json_str = match.group(0)
            domains_list = json.loads(json_str)

            if not isinstance(domains_list, list):
                return None

            # Map string values to DocumentSource enum
            result = []
            for domain_str in domains_list:
                domain_str = domain_str.strip().lower().replace(" ", "_")

                try:
                    source = DocumentSource(domain_str)
                    result.append(source)
                except ValueError:
                    # Unknown domain, skip
                    pass

            return result if result else None
        except Exception as e:
            self.logger.debug(f"Failed to parse domain response: {e}")
            return None

    def _detect_domains_keyword_fallback(self, query: str) -> List[DocumentSource]:
        """
        Fallback domain detection using keyword matching.
        
        Args:
            query: User query text
        
        Returns:
            List of detected DocumentSource domains
        """
        query_lower = query.lower()
        detected = set()

        for domain, keywords in self.domain_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.add(domain)

        # Default to active domains if nothing detected
        if detected:
            return [domain for domain in detected if domain in self.available_sources]
        return self.available_sources or list(self.domain_keywords.keys())

    def route_hybrid(self, query: str) -> tuple[List[DocumentSource], bool]:
        """
        Hybrid routing strategy: keyword-first with LLM fallback.
        
        Uses binary confidence scoring (keyword_count >= 1 = confident).
        Implements caching to avoid repeated LLM calls for same query types.
        
        Args:
            query: User query text
        
        Returns:
            Tuple of (domains, is_confident) where is_confident indicates
            if routing was done via keywords (True) or LLM fallback (False)
        """
        # Check cache first (hash of query)
        query_hash = hash(query)
        if query_hash in self.routing_cache:
            cached_domains = self.routing_cache[query_hash]
            self.logger.debug(f"🔄 Routing cache hit: {cached_domains}")
            return cached_domains, True
        
        # Try keyword-based routing first (binary confidence: >= 1 match)
        query_lower = query.lower()
        keyword_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            if domain not in self.available_sources:
                continue
            score = sum(1 for kw in keywords if kw in query_lower)
            keyword_scores[domain] = score
        
        # Binary confidence: if any domain has >= 1 keyword match, use it
        confident_domains = [domain for domain, score in keyword_scores.items() if score >= 1]
        
        if confident_domains:
            self.logger.info(f"✅ Keyword routing (confident): {confident_domains}")
            # Cache result
            self.routing_cache[query_hash] = confident_domains
            return confident_domains, True
        
        if not ROUTING_ENABLE_LLM_FALLBACK or not self.domain_llm:
            fallback_domains = self.available_sources or list(self.domain_keywords.keys())
            self.routing_cache[query_hash] = fallback_domains
            return fallback_domains, False

        # Fallback to LLM routing if no keyword matches
        self.logger.info(f"⚠️  No keyword matches, using LLM routing...")
        llm_domains = self.detect_domains(query)
        llm_domains = [domain for domain in llm_domains if domain in self.available_sources]
        if not llm_domains:
            llm_domains = self.available_sources or list(self.domain_keywords.keys())
        
        # Cache result
        self.routing_cache[query_hash] = llm_domains
        
        return llm_domains, False

    def preload_all_retrievers(self) -> None:
        """
        Force-load all domain retrievers at startup.
        
        Triggers lazy loading for all FAISS indices to remove cold-start
        latency from first query.
        """
        self.logger.info("🚀 Preloading all FAISS indices...")
        
        for source in self.available_sources:
            try:
                self._load_retriever(source)
                self.logger.info(f"  ✓ {source.value} preloaded")
            except Exception as e:
                self.logger.warning(f"  ✗ {source.value} preload failed: {e}")
        
        self.logger.info(f"✅ Preload complete: {len(self.retrievers)}/{len(self.available_sources)} domains ready")

    def _load_retriever(self, source: DocumentSource) -> Optional[IRetriever]:
        """
        Lazy-load per-domain retriever from disk.
        Caches after first load.
        
        Args:
            source: DocumentSource to load
        
        Returns:
            HybridRetriever instance or None if load fails
        """
        # Check cache
        if source in self.retrievers:
            return self.retrievers[source]

        domain_name = source.value
        domain_path = os.path.join(self.index_dir, domain_name)
        faiss_path = os.path.join(domain_path, "index.faiss")
        metadata_path = os.path.join(domain_path, "metadata.json")

        # Check if indices exist
        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            self.logger.warning(f"⚠️  Indices not found for domain {domain_name}")
            return None

        try:
            # Load FAISS index (now 384-dim for MiniLM)
            from src.retrieval.vector_index import VectorIndex
            vector_index = VectorIndex(dimension=384)  # Updated: MiniLM embedding dimension
            vector_index.load_index(faiss_path)

            # Load metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            vector_index.metadata = metadata

            # Reconstruct DocumentChunk objects from metadata
            from src.import_map import DocumentChunk
            chunks = []
            for idx, (chunk_id, chunk_meta) in enumerate(metadata.items()):
                # Handle nested metadata structure from VectorIndex
                # VectorIndex wraps metadata in "metadata" key
                if isinstance(chunk_meta, dict) and "metadata" in chunk_meta:
                    meta_dict = chunk_meta["metadata"]
                else:
                    meta_dict = chunk_meta
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=meta_dict.get("text", ""),
                    source=source,
                    document_name=meta_dict.get("document", ""),
                    chunk_index=meta_dict.get("chunk_index", idx),
                    page_number=meta_dict.get("page", 0),
                    metadata=meta_dict,
                )
                chunks.append(chunk)

            # Initialize HybridRetriever for this domain
            from src.retrieval.hybrid_retriever import HybridRetriever
            retriever = HybridRetriever(
                embedding_model=self.embedding_model,
                vector_index=vector_index,
                chunks=chunks,
            )

            # Cache it
            self.retrievers[source] = retriever
            self.logger.info(f"✅ Loaded {domain_name} retriever ({len(chunks)} chunks)")
            return retriever

        except Exception as e:
            self.logger.error(f"❌ Failed to load {domain_name} retriever: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ============================================================================
    # IRetriever Interface Implementation
    # ============================================================================

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """
        Main search entry point: hybrid route → load retrievers → search.
        Uses keyword-first routing (binary confidence) with LLM fallback.
        Implements IRetriever.search()
        
        Args:
            query: Search query
            k: Number of results
            filters: Optional filters (unused for federated)
        
        Returns:
            Combined RetrievalResult from all detected domains
        """
        if not query:
            raise ValueError("Query cannot be empty")

        # Use hybrid routing (keyword-first with LLM fallback)
        detected_domains, is_confident = self.route_hybrid(query)
        routing_method = "keyword" if is_confident else "LLM"
        self.logger.info(f"🔍 Query: '{query}' → {routing_method} routing: {[d.value for d in detected_domains]}")

        # Load retrievers and search
        results = []
        for source in detected_domains:
            retriever = self._load_retriever(source)
            if not retriever:
                self.logger.debug(f"  ⊗ {source.value}: not available")
                continue

            try:
                result = retriever.search(query, k=k)
                results.append(result)
                self.logger.debug(f"  ✓ {source.value}: {len(result.chunks)} results")
            except Exception as e:
                self.logger.error(f"  ❌ {source.value}: {e}")

        # Combine results
        if not results:
            return RetrievalResult(
                chunks=[],
                strategy_used=RetrievalStrategy.HYBRID,
                scores=[],
                query_used=query,
            )

        return self._combine_results(results, k=k, query=query)

    def search_multi_hop(
        self,
        initial_query: str,
        num_hops: int = 2,
        k_per_hop: int = 5,
    ) -> List[RetrievalResult]:
        """
        Multi-hop retrieval. Implements IRetriever.search_multi_hop()
        
        Args:
            initial_query: Starting query
            num_hops: Number of hops
            k_per_hop: Results per hop
        
        Returns:
            List of RetrievalResult for each hop
        """
        results = []
        current_query = initial_query

        for hop in range(num_hops):
            result = self.search(current_query, k=k_per_hop)
            results.append(result)

            if result.chunks:
                current_query = " ".join(
                    [chunk.text[:100] for chunk in result.chunks[:2]]
                )

        return results

    def is_indexed(self) -> bool:
        """
        Check if any domain index is available. Implements IRetriever.is_indexed()
        
        Returns:
            True if at least one domain index exists
        """
        return len(self.available_sources) > 0

    def index_documents(self, chunks: List) -> None:
        """
        Not used for federated router (indices created by run_pipeline.py).
        Implements IRetriever.index_documents()
        
        Args:
            chunks: Unused
        """
        self.logger.warning("index_documents() not supported on FederatedRouter")
        pass

    # ============================================================================
    # Federated Utility Methods
    # ============================================================================

    def _combine_results(
        self,
        results: List[RetrievalResult],
        k: int = 5,
        query: str = "",
    ) -> RetrievalResult:
        """
        Combine results from multiple domains.
        
        Args:
            results: List of RetrievalResult
            k: Top-k to return
            query: Original query
        
        Returns:
            Combined RetrievalResult
        """
        if not results:
            return RetrievalResult(
                chunks=[],
                strategy_used=RetrievalStrategy.HYBRID,
                scores=[],
                query_used=query,
            )

        all_chunks = []
        all_scores = []

        for result in results:
            all_chunks.extend(result.chunks)
            all_scores.extend(result.scores)

        # Sort by score descending
        combined = sorted(
            zip(all_chunks, all_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        top_chunks = [chunk for chunk, _ in combined[:k]]
        top_scores = [score for _, score in combined[:k]]

        return RetrievalResult(
            chunks=top_chunks,
            strategy_used=RetrievalStrategy.HYBRID,
            scores=top_scores,
            query_used=query,
        )

    def get_source_distribution(
        self,
        results: List[RetrievalResult],
    ) -> Dict[str, int]:
        """
        Get distribution of results by source.
        
        Args:
            results: List of RetrievalResult
        
        Returns:
            Dict mapping source to count
        """
        distribution = {}

        for result in results:
            for chunk in result.chunks:
                source = chunk.source.value
                distribution[source] = distribution.get(source, 0) + 1

        return distribution

    def is_source_available(self, source: DocumentSource) -> bool:
        """
        Check if source index is available.
        
        Args:
            source: DocumentSource to check
        
        Returns:
            True if source index exists on disk
        """
        domain_path = os.path.join(self.index_dir, source.value)
        faiss_path = os.path.join(domain_path, "index.faiss")
        return os.path.exists(faiss_path)

    def get_available_sources(self) -> List[DocumentSource]:
        """
        Get list of available domain sources.
        
        Returns:
            List of available DocumentSource
        """
        available = []
        for source in DocumentSource:
            if self.is_source_available(source):
                available.append(source)
        return available
