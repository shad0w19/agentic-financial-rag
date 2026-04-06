"""
Phase D: Parallel Retrieval Component

Purpose:
Enable parallel querying of domain-specific retrievers to reduce retrieval latency from
0.5-1s (sequential) to 0.2-0.3s (parallel).

Strategy:
1. Accept query + domain classification
2. Query all relevant domain retrievers in parallel (ThreadPoolExecutor)
3. Merge results with deduplication
4. Rerank merged results
5. Return top-k

Expected Impact: -200-400ms per query (30-50% retrieval speedup)

Usage:
    parallel_retriever = ParallelRetriever(
        domain_classifier=domain_classifier,
        personal_tax_retriever=personal_retriever,
        corporate_tax_retriever=corporate_retriever,
        gst_retriever=gst_retriever,
        reranker=reranker,
    )
    
    result = parallel_retriever.search(
        query="How to claim 80C deduction?",
        k=5,
        domain_hint="personal_tax"  # Optional
    )
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.import_map import IRetriever, RetrievalResult, Domain, DomainClassifier


logger = logging.getLogger(__name__)


@dataclass
class ParallelRetrievalMetrics:
    """Metrics for parallel retrieval operation."""
    sequential_equivalent_time: float  # What single-domain retrieval would take
    parallel_actual_time: float        # Actual parallel time
    speedup_pct: float                 # (sequential - actual) / sequential * 100
    domains_queried: int
    total_results_before_merge: int
    total_results_after_dedup: int
    total_results_after_rerank: int


class ParallelRetriever:
    """
    Wrapper that enables parallel domain-specific retrieval.
    
    Queries multiple domain retrievers in parallel and merges results,
    providing significant latency improvement for multi-domain queries.
    """
    
    RETRIEVAL_TIMEOUT = 5.0  # Max time per domain retriever
    MAX_WORKERS = 3          # Max parallel threads (3 domains)
    
    def __init__(
        self,
        domain_classifier: DomainClassifier,
        personal_tax_retriever: IRetriever,
        corporate_tax_retriever: IRetriever,
        gst_retriever: IRetriever,
        reranker=None,  # Optional reranker for result merging
    ):
        """
        Initialize parallel retriever with domain-specific backends.
        
        Args:
            domain_classifier: DomainClassifier for query routing
            personal_tax_retriever: Retriever for personal tax domain
            corporate_tax_retriever: Retriever for corporate tax domain
            gst_retriever: Retriever for GST domain
            reranker: Optional reranker for merged results
        """
        self.domain_classifier = domain_classifier
        self.retrievers = {
            Domain.PERSONAL_TAX: personal_tax_retriever,
            Domain.CORPORATE_TAX: corporate_tax_retriever,
            Domain.GST: gst_retriever,
        }
        self.reranker = reranker
        self.logger = logging.getLogger(__name__)
    
    def search(
        self,
        query: str,
        k: int = 5,
        domain_hint: Optional[str] = None,
        force_parallel: bool = False,
    ) -> RetrievalResult:
        """
        Search across domain-specific retrievers in parallel.
        
        Args:
            query: Query text
            k: Number of results to return
            domain_hint: Optional domain hint ("personal_tax", "corporate_tax", "gst")
            force_parallel: If True, always use parallel retrieval even for single domain
            
        Returns:
            RetrievalResult with merged and reranked results
        """
        start_time = time.time()
        
        # Step 1: Classify query domain
        domain_classification = self.domain_classifier.classify(query)
        domains_to_query = self._determine_domains(
            domain_classification,
            domain_hint,
            force_parallel,
        )
        
        # Step 2: Decide: parallel or sequential?
        if len(domains_to_query) == 1:
            # Single domain: use sequential retrieval
            logger.debug(f"Single domain detected: {domains_to_query[0].value}. Using sequential retrieval.")
            result = self._sequential_search(query, domains_to_query, k)
        else:
            # Multi-domain: use parallel retrieval
            logger.debug(f"Multi-domain query detected: {[d.value for d in domains_to_query]}. Using parallel retrieval.")
            result = self._parallel_search(query, domains_to_query, k)
        
        # Step 3: Measure and log
        total_time = (time.time() - start_time) * 1000  # ms
        self.logger.info(
            f"Parallel retrieval completed in {total_time:.0f}ms | "
            f"Domains: {len(domains_to_query)} | Results: {len(result.chunks)}"
        )
        
        return result
    
    def _determine_domains(
        self,
        domain_classification,
        domain_hint: Optional[str],
        force_parallel: bool,
    ) -> List[Domain]:
        """Determine which domains to query."""
        domains = domain_classification.domains_detected
        
        if domain_hint and not force_parallel:
            # Respect domain hint if provided
            hint_domain = self._parse_domain_hint(domain_hint)
            if hint_domain:
                return [hint_domain]
        
        if domain_classification.is_multi_domain or force_parallel:
            # Multi-domain or forced: return all detected domains
            return domains or [domain_classification.primary_domain]
        
        # Single domain: return primary only
        return [domain_classification.primary_domain]
    
    def _parse_domain_hint(self, hint: str) -> Optional[Domain]:
        """Parse domain hint string to Domain enum."""
        hint_lower = hint.lower()
        if "personal" in hint_lower:
            return Domain.PERSONAL_TAX
        elif "corporate" in hint_lower or "company" in hint_lower:
            return Domain.CORPORATE_TAX
        elif "gst" in hint_lower:
            return Domain.GST
        return None
    
    def _sequential_search(
        self,
        query: str,
        domains: List[Domain],
        k: int,
    ) -> RetrievalResult:
        """Execute sequential search for single domain."""
        if not domains:
            # Fallback: empty result
            from datetime import datetime
            return RetrievalResult(
                chunks=[],
                query_used=query,
                strategy_used="parallel_retrieval_fallback",
                scores=[],
                timestamp=datetime.utcnow(),
            )
        
        domain = domains[0]
        retriever = self.retrievers.get(domain)
        
        if not retriever:
            logger.warning(f"No retriever for domain {domain.value}")
            from datetime import datetime
            return RetrievalResult(
                chunks=[],
                query_used=query,
                strategy_used="parallel_retrieval_fallback",
                scores=[],
                timestamp=datetime.utcnow(),
            )
        
        try:
            result = retriever.search(query, k=k)
            return result
        except Exception as e:
            logger.exception(f"Sequential search failed: {e}")
            from datetime import datetime
            return RetrievalResult(
                chunks=[],
                query_used=query,
                strategy_used="parallel_retrieval_failed",
                scores=[],
                timestamp=datetime.utcnow(),
            )
    
    def _parallel_search(
        self,
        query: str,
        domains: List[Domain],
        k: int,
    ) -> RetrievalResult:
        """
        Execute parallel search across multiple domains.
        
        Returns merged, deduplicated, and reranked results.
        """
        # Step 1: Query all domains in parallel
        all_chunks = []
        seen_chunk_ids = set()
        
        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = {}
            
            for domain in domains:
                retriever = self.retrievers.get(domain)
                if not retriever:
                    logger.warning(f"No retriever for domain {domain.value}")
                    continue
                
                future = executor.submit(
                    self._safe_retrieve,
                    retriever,
                    query,
                    k,
                    domain,
                )
                futures[domain] = future
            
            # Step 2: Collect results with timeout
            for domain, future in futures.items():
                try:
                    result = future.result(timeout=self.RETRIEVAL_TIMEOUT)
                    
                    if result and result.chunks:
                        for chunk in result.chunks:
                            if chunk.chunk_id not in seen_chunk_ids:
                                all_chunks.append(chunk)
                                seen_chunk_ids.add(chunk.chunk_id)
                    
                    logger.debug(
                        f"Domain {domain.value}: {len(result.chunks) if result else 0} results"
                    )
                
                except TimeoutError:
                    logger.warning(f"Retrieval timeout for domain {domain.value}")
                except Exception as e:
                    logger.exception(f"Retrieval failed for domain {domain.value}: {e}")
        
        # Step 3: Rerank merged results if reranker available
        if self.reranker and all_chunks:
            all_chunks = self._rerank_results(query, all_chunks)
        
        # Step 4: Select top-k
        top_chunks = all_chunks[:k]
        
        # Step 5: Create result
        from datetime import datetime
        scores = [1.0 / (i + 1) for i in range(len(top_chunks))]  # Decay scores for ranking
        result = RetrievalResult(
            chunks=top_chunks,
            query_used=query,
            strategy_used="parallel_retrieval",
            scores=scores,
            timestamp=datetime.utcnow(),
        )
        
        return result
    
    def _safe_retrieve(self, retriever: IRetriever, query: str, k: int, domain) -> Optional:
        """Safely retrieve with exception handling."""
        try:
            return retriever.search(query, k=k)
        except Exception as e:
            logger.warning(f"Retrieval error for domain {domain.value}: {e}")
            return None
    
    def _rerank_results(self, query: str, chunks: List):
        """Rerank chunks using CrossEncoder if available."""
        if not self.reranker:
            return chunks
        
        try:
            # Reranker scores chunks
            reranked = self.reranker.rerank(query, chunks)
            # Sort by relevance score descending
            reranked_sorted = sorted(
                reranked,
                key=lambda x: x.get("score", 0),
                reverse=True,
            )
            # Extract chunks in sorted order
            return [chunk for chunk, score in reranked_sorted]
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original order.")
            return chunks


class ParallelRetrievalBenchmark:
    """Utility for benchmarking parallel vs. sequential retrieval."""
    
    def __init__(self, parallel_retriever: ParallelRetriever):
        self.parallel_retriever = parallel_retriever
        self.logger = logging.getLogger(__name__)
    
    def benchmark(
        self,
        test_queries: List[str],
        k: int = 5,
        num_runs: int = 3,
    ) -> Dict:
        """
        Benchmark parallel retrieval against baseline.
        
        Args:
            test_queries: List of queries to test
            k: Number of results
            num_runs: Number of runs for averaging
            
        Returns:
            Benchmark results with latency comparison
        """
        results = {
            "queries_tested": len(test_queries),
            "runs": num_runs,
            "each_run_results": [],
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
        
        timings = []
        
        for run_idx in range(num_runs):
            run_timings = []
            
            for query in test_queries:
                start = time.time()
                result = self.parallel_retriever.search(query, k=k, force_parallel=True)
                latency_ms = (time.time() - start) * 1000
                run_timings.append(latency_ms)
            
            avg_run = sum(run_timings) / len(run_timings)
            results["each_run_results"].append({
                "run": run_idx + 1,
                "avg_latency_ms": avg_run,
                "min_latency_ms": min(run_timings),
                "max_latency_ms": max(run_timings),
            })
            
            timings.extend(run_timings)
        
        results["avg_latency_ms"] = sum(timings) / len(timings)
        results["p50_latency_ms"] = sorted(timings)[len(timings) // 2]
        results["p95_latency_ms"] = sorted(timings)[int(len(timings) * 0.95)]
        
        self.logger.info(
            f"Benchmark complete: Avg={results['avg_latency_ms']:.0f}ms, "
            f"P50={results['p50_latency_ms']:.0f}ms, P95={results['p95_latency_ms']:.0f}ms"
        )
        
        return results
