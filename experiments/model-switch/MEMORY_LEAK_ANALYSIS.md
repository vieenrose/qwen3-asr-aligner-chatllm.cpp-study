# Memory Leak Analysis - Experiment 3

## Executive Summary

Model switching between Qwen3-ASR 0.6B and 1.7B shows significant memory accumulation:
- **0.6B model**: Net memory *release* (-73 MB per switch)
- **1.7B model**: Memory *leak* (+110 MB per switch)
- **Combined leak rate**: ~32 MB/iteration

## Experimental Results

### 100-Iteration Model Switching Test

| Metric | 0.6B Model | 1.7B Model |
|--------|------------|------------|
| Avg Load Overhead (MB) | -4.66 | 633.17 |
| Avg Unload Release (MB) | 68.80 | 523.44 |
| **Avg Net Leak per Switch (MB)** | **-73.46** | **+109.73** |
| Avg TTFT (ms) | 5,071 | 9,976 |
| Avg WER (%) | 0.00 | 10.00 |

### Overall Memory Growth
- Initial: 28 MB
- Final (100 iterations): 3,243 MB
- **Total Growth: 3,215 MB (32 MB/iteration)**

## Root Cause Analysis

### Source Code Investigation

Memory management in chatllm.cpp follows RAII patterns with `std::unique_ptr` for most allocations. The destruction chain is:

```
chatllm_destroy(obj)
  → chat_objects.erase(it)
    → unique_ptr<Chat>::~unique_ptr()
      → Chat::~Chat()
        → Pipeline::~Pipeline()
          → ModelObject::~ModelObject()
            → AbstractModel::~AbstractModel()
              → HeterogeneousModel::~HeterogeneousModel()
                → delete layers/word_embeddings/etc.
```

### Potential Leak Points

| Issue | Location | Severity |
|-------|----------|----------|
| **1.7B model size** | Model architecture | High |
| Conditional destroy failure | `main.cpp:1579` | Medium |
| Raw pointer layers | `models.h:108` | Low |

### Valgrind Findings

Partial Valgrind analysis (test timed out due to 10-100x slowdown) revealed:

```
Mismatched new/delete size value: 8
   at operator delete(void*, unsigned long)
   by ~unique_ptr
   by HeterogeneousModel::~HeterogeneousModel() (models.cpp:1378)
```

This suggests a potential size mismatch in allocation/deallocation of model components.

## Key Finding: 1.7B Model is the Leak Source

The data clearly shows:
1. **0.6B releases memory** after unload (-73 MB net per switch)
2. **1.7B leaks memory** after unload (+110 MB net per switch)

The 1.7B model has ~2.7x more parameters (2.0B vs 0.75B), and the memory leak scales proportionally.

## Hypotheses

### H1: GGML Backend Resources Not Fully Freed
The larger model may create additional GGML backend buffers or contexts that aren't properly released in `BackendContext::~BackendContext()`.

### H2: Python Binding Reference Leak
The Python `ctypes` binding may hold references to the C++ object, preventing proper destruction.

### H3: mmap File Handles
Model files are memory-mapped. Larger model = larger mmap. The `MappedFile::~MappedFile()` may not be called in all code paths.

## Recommendations

### Immediate
1. **Use 0.6B model** - It's faster (5s vs 10s TTFT), more accurate (0% vs 10% WER), and doesn't leak
2. **Use `restart()` instead of destroy/recreate** - From exp2, this reduces leak by 192x

### Long-term
1. Add explicit `gc.collect()` in Python after `chat.destroy()`
2. Investigate GGML backend buffer cleanup in 1.7B model path
3. Consider adding `chatllm_gc()` function to force cleanup

## Tools Created

- `run_model_switch.py` - Model switching benchmark with detailed memory tracking
- `investigate_leak.py` - Valgrind integration for leak detection

## Related

- **Exp2**: Showed `restart()` has ~192x less memory growth than destroy/recreate
- **Exp3**: Shows 1.7B model is the primary leak source when model switching
