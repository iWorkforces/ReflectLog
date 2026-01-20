# ReflectLogMCP Performance Characteristics

This document describes the performance characteristics, scaling behaviors, and tuning recommendations for ReflectLogMCP.

## Overview

ReflectLogMCP is a hybrid semantic + full-text memory storage system that combines:
- **USearch** with HNSW for semantic vector search
- **Tantivy** for BM25 full-text search
- **RRF Fusion** for combining results from both engines
- **Optional AI Reranking** (LLM or CrossEncoder) for relevance refinement

## Search Latency

### Expected Latency by Index Size

| Index Size | Semantic Search | Full-Text Search | RRF Fusion | LLM Rerank | Total (with Rerank) |
|------------|----------------|------------------|------------|------------|---------------------|
| 100        | 5-15ms         | 1-5ms            | <1ms       | 100-500ms  | 110-530ms           |
| 1,000      | 10-30ms        | 2-8ms            | <1ms       | 100-500ms  | 115-540ms           |
| 10,000     | 20-50ms        | 5-15ms           | <1ms       | 100-500ms  | 130-570ms           |
| 100,000    | 30-80ms        | 10-30ms          | <1ms       | 100-500ms  | 150-620ms           |

### Latency Optimization Tips

1. **Disable Reranking**: Set `RERANKER_ENGINE=none` for fastest searches (~100-200ms saved)
2. **Use CrossEncoder**: Set `RERANKER_ENGINE=cross_encoder` for faster reranking (~50-150ms vs 100-500ms)
3. **Reduce Overfetch**: Lower `OVERFETCH_MULTIPLIER` for small indexes (1.5x instead of 3x)
4. **Limit Results**: Lower `SEARCH_LIMIT` to reduce processing time

## Throughput

### Add Operations

| Operation Type | Messages/sec (batch) | Notes |
|----------------|---------------------|-------|
| Single add     | ~10-20/sec          | Includes embedding API call |
| Batch add (10) | ~30-50/sec          | Parallel duplicate detection |
| Batch add (100)| ~50-80/sec          | Efficient parallel processing |

**Phased Parallel Add Processing**: The system uses a 3-phase approach:
1. **Phase 1**: Parallel duplicate detection (configurable via `ADD_MAX_CONCURRENCY`)
2. **Phase 2**: Parallel smart replacement checks
3. **Phase 3**: Sequential database writes (SQLite constraint)

### Search Operations

| Reranker Type | Queries/sec (avg) | Notes |
|---------------|-------------------|-------|
| None          | ~50-100/sec       | RRF fusion only |
| CrossEncoder  | ~20-40/sec        | Local model, fast |
| LLM           | ~10-20/sec        | API latency bottleneck |

## Memory Usage

### Index Size Growth

| Index Size | USearch Size | Tantivy Size | Total (approx) |
|------------|--------------|--------------|----------------|
| 100 docs   | ~1 MB        | ~0.5 MB      | ~1.5 MB        |
| 1,000 docs | ~10 MB       | ~5 MB        | ~15 MB         |
| 10,000 docs| ~100 MB      | ~50 MB       | ~150 MB        |
| 100,000 docs| ~1 GB       | ~500 MB      | ~1.5 GB        |

### Memory Optimization

- **Exact vs Approximate Search**: Use `USEARCH_EXACT_SEARCH=true` for small indexes (<10k), `false` for large indexes
- **Tantivy Soft-Delete**: Reduces memory overhead by avoiding index rebuilds (enabled by default)
- **Cache Sizing**: Lower `EMBEDDING_CACHE_SIZE` to reduce memory footprint

## Embedding Performance

### Embedding Latency

| Provider | Model | Batch Size | Latency per Batch |
|----------|-------|------------|-------------------|
| OpenAI   | text-embedding-3-large | 1 | ~100-300ms |
| OpenAI   | text-embedding-3-large | 100 | ~500-800ms |
| OpenAI   | text-embedding-3-large | 512 | ~1-2s |
| Qwen     | qwen3-embedding-8b | 1 | ~50-150ms |
| Qwen     | qwen3-embedding-8b | 100 | ~200-400ms |
| Qwen     | qwen3-embedding-8b | 512 | ~500-800ms |

### Caching Strategy

Query embeddings are cached by default (LRU with 100 entries):
- **Hit Rate**: ~60-80% for repeated queries
- **Latency Savings**: ~100-300ms per cached query
- **Configuration**: Adjust via `EMBEDDING_CACHE_SIZE`

## Tuning Guidelines

### For Low Latency (<100ms per search)

```bash
# Disable reranking for fastest searches
RERANKER_ENGINE=none

# Reduce overfetch for small indexes
OVERFETCH_MULTIPLIER=1.5

# Use exact search for small indexes
USEARCH_EXACT_SEARCH=true

# Disable smart replacement (saves ~100-200ms on add)
ENABLE_SMART_REPLACE=false
```

### For High Quality (Best relevance)

```bash
# Enable LLM reranking for best quality
RERANKER_ENGINE=llm
LLM_MODEL=x-ai/grok-4.1-fast

# Enable smart replacement
ENABLE_SMART_REPLACE=true
SMART_REPLACE_THRESHOLD=0.7

# Use higher overfetch for better fusion quality
OVERFETCH_MULTIPLIER=3.0

# Enable recency boost for temporal context
ENABLE_RECENCY_BOOST=true
```

### For Cost Efficiency (Minimize API calls)

```bash
# Use local cross-encoder instead of LLM
RERANKER_ENGINE=cross_encoder

# Increase query cache size
EMBEDDING_CACHE_SIZE=500

# Disable smart replacement (uses LLM)
ENABLE_SMART_REPLACE=false
```

### For Large Indexes (>10k documents)

```bash
# Use approximate search for better performance
USEARCH_EXACT_SEARCH=false
USEARCH_EXACT_SEARCH_THRESHOLD=10000

# Enable adaptive overfetch
OVERFETCH_ADAPTIVE=true

# Use cross-encoder for faster reranking
RERANKER_ENGINE=cross_encoder
CROSS_ENCODER_DEVICE=cuda  # or mps for Apple Silicon
```

## Configuration Reference

### Key Performance Settings

| Setting | Default | Range | Impact |
|---------|---------|-------|--------|
| `SEARCH_LIMIT` | 5 | 1-50 | Higher = more results, slower |
| `OVERFETCH_MULTIPLIER` | 3 | 1.5-5 | Higher = better fusion, slower |
| `EMBEDDING_CACHE_SIZE` | 100 | 0-1000 | Higher = fewer API calls, more memory |
| `RERANKER_ENGINE` | llm | none/cross_encoder/llm | Affects quality and latency |
| `CROSS_ENCODER_DEVICE` | cpu | cpu/cuda/mps | GPU = faster inference |
| `USEARCH_EXACT_SEARCH` | true | true/false | Exact = slower, more accurate |
| `ADD_MAX_CONCURRENCY` | 4 | 1-16 | Higher = faster adds, more CPU |

## Benchmarks

### Benchmark Script

Run the included benchmark script:

```bash
uv run python scripts/benchmark_engines.py
```

This will test:
- Single message add latency
- Batch add throughput
- Search latency at various index sizes
- Memory usage over time

### Expected Benchmark Results (on typical hardware)

| Operation | Expected Performance |
|-----------|---------------------|
| Single add | ~100-300ms (includes embedding) |
| Batch add (100) | ~2-5 seconds total |
| Search (100 docs) | ~100-300ms |
| Search (10k docs) | ~150-400ms |
| Get all (10k docs) | ~50-100ms |

## Scaling Recommendations

### Small Projects (<1k memories)

- Use default settings
- Exact search is fine
- LLM reranking provides good value
- Smart replacement helpful for maintaining consistency

### Medium Projects (1k-10k memories)

- Consider switching to approximate search at 5k+
- Cross-encoder provides good balance of quality/speed
- Increase cache size for better hit rates
- Monitor memory usage

### Large Projects (>10k memories)

- Use approximate search
- Cross-encoder on GPU for best performance
- Consider disabling smart replacement if add latency is critical
- Monitor index size and compaction

### Production Checklist

- [ ] Set appropriate `USEARCH_EXACT_SEARCH_THRESHOLD`
- [ ] Configure `CROSS_ENCODER_DEVICE` for GPU acceleration
- [ ] Tune `OVERFETCH_MULTIPLIER` based on index size
- [ ] Set `EMBEDDING_CACHE_SIZE` based on query patterns
- [ ] Enable circuit breaker for external API calls: `CIRCUIT_BREAKER_ENABLED=true`
- [ ] Configure metrics for monitoring
- [ ] Set appropriate log level (`LOG_LEVEL=WARNING` for production)

## Monitoring

### Key Metrics to Track

1. **Search Latency (p50, p95, p99)**
   - Alert if p99 > 1s
   - Indicates performance degradation

2. **Add Operation Throughput**
   - Monitor for sudden drops
   - May indicate external API issues

3. **Index Size**
   - Monitor growth rate
   - Plan capacity accordingly

4. **Cache Hit Rate**
   - Target >60% for query cache
   - Adjust cache size if too low

5. **Error Rate**
   - Monitor for external API failures
   - Circuit breaker should prevent cascading failures

### Metrics Endpoint

Metrics are available in Prometheus format via the metrics registry:

```python
from reflectlog.application.utils import MetricsRegistry

metrics = MetricsRegistry()
prometheus_text = metrics.export_prometheus()
```

## Troubleshooting

### Slow Searches

1. Check if reranking is enabled (`RERANKER_ENGINE`)
2. Check overfetch multiplier (`OVERFETCH_MULTIPLIER`)
3. Check index size and search mode (`USEARCH_EXACT_SEARCH`)
4. Consider using CrossEncoder instead of LLM

### High Memory Usage

1. Check index size (`ls -lh indexes/*/`)
2. Reduce embedding cache size (`EMBEDDING_CACHE_SIZE`)
3. Enable compaction (`TANTIVY_COMPACTION_THRESHOLD_RATIO`)
4. Consider archiving old memories

### Slow Add Operations

1. Check if smart replacement is enabled (`ENABLE_SMART_REPLACE`)
2. Check batch size for embeddings (`EMBEDDING_BATCH_SIZE`)
3. Check concurrency settings (`ADD_MAX_CONCURRENCY`)
4. Check external API latency (circuit breaker status)

### Poor Search Quality

1. Check if reranking is enabled
2. Check fusion threshold (`FUSION_RANKING_THRESHOLD`)
3. Check recency boost settings (`ENABLE_RECENCY_BOOST`)
4. Consider adjusting search limit (`SEARCH_LIMIT`)
