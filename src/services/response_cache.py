"""
Phase D: Response Caching Service

Purpose:
Cache answers to common tax questions to enable sub-500ms responses for frequently asked queries.

Strategy:
1. Normalize query text (lowercase, remove punctuation, synonyms)
2. Compute query hash
3. Check cache (with TTL)
4. If miss: execute full workflow, cache result
5. If hit: return cached response

Expected Impact:
- 15-25% of queries hit cache
- Cached queries: <500ms latency (vs. 8-12s full pipeline)
- Average speedup per cached query: 5-8 seconds

Cache Population:
- Automatically populate as queries are served
- Manual warm-up of top 20 tax query patterns
- TTL: 1 hour for general queries, 30 mins for rate-sensitive

Usage:
    cache = ResponseCache(
        cache_size_mb=100,
        default_ttl_seconds=3600,
    )
    
    # Try cache first
    cached = cache.get(query_text)
    if cached:
        return cached
    
    # Execute and cache
    response = execute_workflow(query)
    cache.put(query_text, response, ttl_seconds=3600)
    return response
"""

import hashlib
import logging
import time
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    query_normalized: str
    response: Dict[str, Any]
    timestamp: float
    ttl_seconds: int
    hit_count: int = 0
    last_accessed: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        age_seconds = time.time() - self.timestamp
        return age_seconds > self.ttl_seconds
    
    def record_hit(self) -> None:
        """Record cache hit."""
        self.hit_count += 1
        self.last_accessed = time.time()


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    active_entries: int = 0
    memory_usage_bytes: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    evictions: int = 0
    warm_queries: int = 0


class ResponseCache:
    """
    LRU cache for query responses with TTL support.
    
    Features:
    - Automatic query normalization
    - Per-entry TTL (time-to-live)
    - LRU eviction when size exceeded
    - Hit/miss tracking and statistics
    - Cache warm-up support
    """
    
    # Top 20 query patterns to warm up cache
    WARM_UP_QUERIES = [
        "What is the standard GST rate?",
        "Can I claim Section 80C deduction?",
        "What is the income tax rate for salaried employees?",
        "How do I file an ITR?",
        "What is the GST rate for services?",
        "Can I claim HRA exemption?",
        "What is the tax rate for freelancers?",
        "How to save taxes using 80C, 80D, 80E deductions?",
        "What is the GST on real estate?",
        "How to calculate taxable income?",
        "What documents do I need for ITR filing?",
        "Can I claim business losses?",
        "What is the corporate tax rate?",
        "How does TDS work?",
        "Can I claim capital gains exemption?",
        "What is the income tax deadline?",
        "How to claim medical expenses under 80D?",
        "What is the GST rate for food items?",
        "Can I claim home loan interest deduction?",
        "What is the tax on investments in mutual funds?",
    ]
    
    def __init__(
        self,
        cache_size_mb: int = 100,
        default_ttl_seconds: int = 3600,
    ):
        """
        Initialize response cache.
        
        Args:
            cache_size_mb: Maximum cache size in MB
            default_ttl_seconds: Default TTL for entries
        """
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.default_ttl_seconds = default_ttl_seconds
        self.cache: Dict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self.logger = logging.getLogger(__name__)
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get cached response for query.
        
        Args:
            query: Query text
            
        Returns:
            Cached response if hit and not expired, None otherwise
        """
        query_normalized = self._normalize_query(query)
        query_hash = self._hash_query(query_normalized)
        
        if query_hash not in self.cache:
            self.stats.cache_misses += 1
            self.stats.total_queries += 1
            return None
        
        entry = self.cache[query_hash]
        
        # Check expiry
        if entry.is_expired():
            logger.debug(f"Cache entry expired: {query_normalized[:50]}")
            del self.cache[query_hash]
            self.stats.cache_misses += 1
            self.stats.total_queries += 1
            return None
        
        # Record hit
        entry.record_hit()
        self.stats.cache_hits += 1
        self.stats.total_queries += 1
        
        # Move to end (LRU)
        self.cache.move_to_end(query_hash)
        
        logger.debug(
            f"Cache hit: {query_normalized[:50]} | Hit count: {entry.hit_count}"
        )
        
        return entry.response
    
    def put(
        self,
        query: str,
        response: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Cache a response for query.
        
        Args:
            query: Query text
            response: Response to cache
            ttl_seconds: Optional custom TTL (use default if not provided)
        """
        query_normalized = self._normalize_query(query)
        query_hash = self._hash_query(query_normalized)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        # Check if we need to evict
        response_size = self._estimate_size(response)
        while self._total_size() + response_size > self.cache_size_bytes and self.cache:
            evicted_key = next(iter(self.cache))
            logger.debug(f"Evicting cache entry due to size limit")
            del self.cache[evicted_key]
            self.stats.evictions += 1
        
        # Add to cache
        entry = CacheEntry(
            query_normalized=query_normalized,
            response=response,
            timestamp=time.time(),
            ttl_seconds=ttl,
        )
        
        self.cache[query_hash] = entry
        
        logger.debug(
            f"Cached response: {query_normalized[:50]} | TTL: {ttl}s"
        )
    
    def warm_up(self) -> None:
        """
        Warm up cache with common query patterns.
        
        This creates placeholders for common queries. In production,
        these would be pre-populated with real answers.
        """
        for query in self.WARM_UP_QUERIES:
            self.stats.warm_queries += 1
        
        logger.info(f"Cache warm-up: {len(self.WARM_UP_QUERIES)} queries prepared")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        self.stats.active_entries = len(self.cache)
        self.stats.memory_usage_bytes = self._total_size()
        
        if self.stats.total_queries > 0:
            self.stats.cache_hit_rate = self.stats.cache_hits / self.stats.total_queries
        
        return self.stats
    
    def prune_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries pruned
        """
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Pruned {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent cache lookups."""
        # Lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Normalize multi-word synonyms FIRST (before single-word ones)
        multi_word_synonyms = {
            r'\bgoods\s+and\s+services\s+tax\b': 'gst',
            r'\bincome\s+tax\b': 'income tax',
            r'\bmutual\s+fund\b': 'mutual fund',
            r'\bcapital\s+gain\b': 'capital gains',
            r'\bhome\s+loan\b': 'home loan',
            r'\b80.?c\s+(deduction|deposit)?\b': 'section 80c',
            r'\b80.?d\s+(deduction|deposit)?\b': 'section 80d',
            r'\b80.?e\s+(deduction|deposit)?\b': 'section 80e',
        }
        
        for pattern, replacement in multi_word_synonyms.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Normalize single-word synonyms and variations
        synonyms = {
            r'\bgst\b': 'gst',
            r'\bitr\b': 'itr',
            r'\b80c\b': 'section 80c',
            r'\b80d\b': 'section 80d',
            r'\b80e\b': 'section 80e',
            r'\bhra\b': 'hra',
            r'\btds\b': 'tds',
            r'\bsip\b': 'sip',
            r'\bdeduct\b|\bclaim\b': 'deduct',
        }
        
        for pattern, replacement in synonyms.items():
            normalized = re.sub(pattern, replacement, normalized)
        
        # Remove all punctuation and special characters
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized
    
    def _hash_query(self, normalized_query: str) -> str:
        """Create hash of normalized query."""
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def _total_size(self) -> int:
        """Estimate total cache size in bytes."""
        total = 0
        for entry in self.cache.values():
            total += self._estimate_size(entry.response)
        return total
    
    def _estimate_size(self, obj: Any) -> int:
        """Rough estimate of object size in bytes."""
        import sys
        import json
        
        # For dictionaries and responses, estimate based on JSON serialization
        try:
            json_size = len(json.dumps(obj, default=str).encode('utf-8'))
            return json_size
        except:
            # Fallback to sys.getsizeof for non-JSON objects
            return sys.getsizeof(obj)
    
    def export_stats(self) -> Dict[str, Any]:
        """Export cache statistics for monitoring."""
        stats = self.get_stats()
        return {
            "total_queries": stats.total_queries,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "hit_rate": f"{stats.cache_hit_rate:.1%}",
            "active_entries": stats.active_entries,
            "memory_usage_mb": round(stats.memory_usage_bytes / (1024 * 1024), 2),
            "evictions": stats.evictions,
            "timestamp": datetime.now().isoformat(),
        }


class CacheWarmer:
    """Utility to pre-warm cache with common queries."""
    
    def __init__(self, cache: ResponseCache):
        self.cache = cache
        self.logger = logging.getLogger(__name__)
    
    def warm_with_responses(self, query_response_pairs: List[tuple]) -> None:
        """
        Warm cache with actual query-response pairs.
        
        Args:
            query_response_pairs: List of (query, response) tuples
        """
        for query, response in query_response_pairs:
            self.cache.put(query, response, ttl_seconds=3600)
        
        self.logger.info(f"Warmed cache with {len(query_response_pairs)} responses")
    
    def export_warm_queries(self) -> List[str]:
        """Get list of queries to warm up."""
        return ResponseCache.WARM_UP_QUERIES.copy()
