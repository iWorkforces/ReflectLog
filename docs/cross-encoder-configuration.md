# Cross-Encoder Reranking Configuration

This document provides guidance on configuring the cross-encoder reranker for different use cases.

## Overview

When `RERANKER_ENGINE=cross_encoder`, CCMemoriesMCP uses a local cross-encoder model (default: `BAAI/bge-reranker-v2-m3`) to rerank search results based on semantic relevance. This approach is faster and incurs no API costs compared to LLM-based reranking.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` | HuggingFace cross-encoder model |
| `CROSS_ENCODER_TOP_K` | `20` | Max results to return after reranking |
| `CROSS_ENCODER_DEVICE` | `cpu` | Inference device: `cpu`, `cuda`, `mps` |
| `CROSS_ENCODER_BATCH_SIZE` | `32` | Batch size for inference |
| `CROSS_ENCODER_SCORE_THRESHOLD` | `0.5` | Min score to keep (0-1 when normalized) |
| `CROSS_ENCODER_USE_FP16` | `true` | Enable FP16 for faster inference |
| `CROSS_ENCODER_NORMALIZE` | `true` | Normalize scores to 0-1 with sigmoid |
| `CROSS_ENCODER_MAX_LENGTH` | `512` | Max token length for query-doc pairs |

## `CROSS_ENCODER_MAX_LENGTH` Recommendations

The `CROSS_ENCODER_MAX_LENGTH` parameter controls the maximum number of tokens the cross-encoder will process for each query-document pair. Content exceeding this limit is truncated.

### Recommended Values by Use Case

| Use Case | Recommended Value | Rationale |
|----------|-------------------|-----------|
| **Chat/Q&A memories** | **256-512** | Short conversational snippets, quick lookups. Optimal for typical memory storage use cases. |
| **Code snippets** | **512-1024** | Function definitions, class summaries. Code can be dense with important context. |
| **Article summaries** | **512-1024** | Paragraph-level content, blog excerpts, news snippets. |
| **Technical documentation** | **1024-2048** | Longer explanations, API docs, tutorials with examples. |
| **Research papers / Legal docs** | **2048-8192** | Full-context analysis requiring maximum comprehension. |

### Trade-offs

| Value Range | Pros | Cons |
|-------------|------|------|
| **Low (256-512)** | Faster inference, lower memory usage, quick responses | Truncates long documents, may miss important context |
| **Medium (512-1024)** | Good balance of speed and context | Moderate memory usage |
| **High (1024-2048)** | Better handling of long content | Slower inference, higher memory usage |
| **Maximum (2048-8192)** | Full document comprehension | Slowest, highest memory, may require GPU |

### Model Limits

Different cross-encoder models have different maximum context lengths:

| Model | Max Tokens | Notes |
|-------|------------|-------|
| `BAAI/bge-reranker-v2-m3` | 8192 | Default model, multilingual, high quality |
| `BAAI/bge-reranker-base` | 512 | Faster, English-focused |
| `BAAI/bge-reranker-large` | 512 | Better quality than base, English-focused |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 512 | Fast, lightweight |

Setting `CROSS_ENCODER_MAX_LENGTH` higher than the model's limit will result in automatic truncation by the model.

### Recommendations for CCMemoriesMCP

For typical memory storage use cases in CCMemoriesMCP:

- **Default (512)**: Optimal for most use cases. Memories are typically concise (the server enforces a 30KB text limit per message).
- **Increase to 1024**: If storing longer code snippets or technical documentation.
- **Increase to 2048+**: Only if storing very long documents and using a model that supports it (like `bge-reranker-v2-m3`).

## Performance Tuning

### For Speed (CPU)

```bash
CROSS_ENCODER_DEVICE=cpu
CROSS_ENCODER_USE_FP16=true
CROSS_ENCODER_MAX_LENGTH=256
CROSS_ENCODER_BATCH_SIZE=64
```

### For Quality (GPU)

```bash
CROSS_ENCODER_DEVICE=cuda  # or mps for Apple Silicon
CROSS_ENCODER_USE_FP16=true
CROSS_ENCODER_MAX_LENGTH=1024
CROSS_ENCODER_BATCH_SIZE=32
```

### For Memory-Constrained Environments

```bash
CROSS_ENCODER_DEVICE=cpu
CROSS_ENCODER_USE_FP16=true
CROSS_ENCODER_MAX_LENGTH=256
CROSS_ENCODER_BATCH_SIZE=16
CROSS_ENCODER_MODEL=BAAI/bge-reranker-base  # smaller model
```
