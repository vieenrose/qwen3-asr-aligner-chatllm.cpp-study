# Qwen3-ASR Development Issues & Solutions

## 1. `free(): invalid pointer` Crash (FIXED ✅)

### Issue
The Qwen3-ASR model crashed with `free(): invalid pointer` after generating initial ASR tokens. The crash occurred in the destructor of `CoreAttention`.

### Root Cause
In `chatllm.cpp/src/layers.h`, a `unique_ptr` was being initialized with the address of a stack-allocated variable (`&def_pos_helper`). When the `unique_ptr` went out of scope, it attempted to `delete` this stack address.

### Solution
Modified `chatllm.cpp/src/layers.h` to always allocate `def_pos_helper` on the heap when no external helper is provided:
```cpp
// Old
pos_helper(helper ? helper : &def_pos_helper)
// New
pos_helper(helper ? helper : new BaseTensorPosHelper(max_length))
```

---

## 2. Empty Transcription - Missing Language Parameter (FIXED ✅)

### Issue
The model output `language None<asr_text>` with no transcription. Qwen3-ASR requires an explicit `language` parameter (e.g., "Chinese"), but the Python bindings had no mechanism to pass this configuration.

### Solution
1.  **C API**: Added `chatllm_set_additional_args` to `chatllm.cpp/src/main.cpp` to properly parse and set model arguments.
2.  **Bindings**: Exposed this function in `chatllm.cpp/bindings/libchatllm.h` and `chatllm.py`.

---

## 3. Python Binding Double-Encoding Bug (FIXED ✅)

### Issue
Calls to `set_additional_args` failed silently. The `ChatLLM` class in Python encoded the string to bytes, and then the `LibChatLLM` wrapper encoded it *again*, passing double-encoded bytes to the C++ library.

### Solution
Updated `chatllm.cpp/bindings/chatllm.py` to check type and only encode if the input is a string.

---

## 4. Audio Loading Failure (FIXED ✅)

### Issue
Audio transcription failed silently or returned empty results because the audio file wasn't being loaded. Debugging `chatllm.cpp/src/audio_process.cpp` revealed that `popen` was failing.

### Root Cause
The code used `popen(cmd, "rb")`. On this Linux environment, `popen` strictly requires `"r"` or `"w"`. The `"b"` flag caused an "Invalid argument" error.

### Solution
Changed the `popen` mode to `"r"` in `chatllm.cpp/src/audio_process.cpp`.

---

## 5. Inference State & Memory Persistence (FIXED ✅)

### Issue
When running the benchmark loop (exp1):
1.  **Inconsistent Results**: Reusing the `ChatLLM` object caused alternating empty results or hallucinations, even with `restart()`. The multi-modal state (audio embeddings) wasn't being cleared completely.
2.  **Memory Leaks**: Creating new objects caused memory to balloon because the C++ objects weren't being destroyed.

### Solution
1.  **API Update**: Exposed `chatllm_destroy` and `chatllm_history_set_cursor` in `libchatllm.h` and `chatllm.py`.
2.  **Benchmark Strategy**: Updated `run_asr.py` to:
    *   Create a single `LibChatLLM` instance (holds heavy model weights).
    *   Create a *new* `ChatLLM` object for each iteration (clean state).
    *   Explicitly call `chat.destroy()` after each iteration to free the C++ `Chat` object.

This approach ensures stable transcription results (0.00% WER) and controlled memory usage (validating the memory leak test requirement).

---

## 6. ForcedAligner Crash with `parent_id` Gaps (FIXED ✅)

### Issue
The ForcedAligner model crashed during chunked audio processing (exp-8):
```
check failed (w.parent_id == (int)tok->timestamps.size())
at /app/chatllm.cpp/models/qwen.cpp:3511
```

### Root Cause
The timestamp building code in `v3_forcedaligner::Tokenizer::generate()` assumed `parent_id` values are contiguous (e.g., 0,0,0, 1,1, 2,2,2). However, when a sentence produces zero cleaned words after token filtering (e.g., punctuation-only sentence), a gap is created in the `parent_id` sequence.

**Code flow that creates gaps:**

1. `append_user()` splits text into sentences, assigning sequential IDs (0, 1, 2...)
2. `append_sentence(id, text)` processes each sentence's words through `clean_token()`
3. If `clean_token()` returns empty for ALL words in a sentence, that sentence's `parent_id` is never added to `cleaned_words`
4. Result: Gap in `parent_id` sequence (e.g., 0, 0, 2, 2 with missing 1)

**Example:**
```
Sentence 0: "Hello"  → clean_token("Hello") = "Hello" → ADD (parent_id=0)
Sentence 1: "..."    → clean_token("...")   = ""      → SKIP (nothing added)
Sentence 2: "World"  → clean_token("World") = "World" → ADD (parent_id=2)

cleaned_words = [{text:"Hello", parent_id:0}, {text:"World", parent_id:2}]
                                                          ↑ Gap! No parent_id=1
```

**Original code that crashes:**
```cpp
// Assumes parent_id is always contiguous
CHATLLM_CHECK(w.parent_id == (int)tok->timestamps.size());  // CRASH on gap
```

### Why foldl Didn't Handle This
foldl was **aware** that gaps could occur (hence the defensive `CHATLLM_CHECK` assertion), but chose to **fail fast** rather than handle the edge case gracefully. This design choice assumes:
- Input text is well-formed (every sentence has valid words)
- Punctuation-only sentences are considered invalid input
- Crashing with a clear error is preferable to silent degradation

The original check was written assuming every sentence produces at least one valid word after cleaning. This is true for normal transcripts, but fails with:
- Punctuation-only sentences
- Chunked audio with awkward boundaries
- Poorly formatted input

### Solution
Modified `chatllm.cpp/models/qwen.cpp` to handle gaps by adding placeholder timestamps:

```cpp
// Build timestamps, handling gaps in parent_id
tok->timestamps.clear();
int expected_parent = 0;
for (int i = 0; i < (int)tok->cleaned_words.size(); i++)
{
    auto &w = tok->cleaned_words[i];
    
    // Handle gaps: add empty placeholder timestamps for skipped parent_ids
    while (expected_parent < w.parent_id)
    {
        tok->timestamps.emplace_back(0.0, 0.0);
        expected_parent++;
    }
    
    if (w.parent_id >= (int)tok->timestamps.size())
    {
        tok->timestamps.emplace_back(timestamps[2 * i + 0], timestamps[2 * i + 1]);
        expected_parent++;
    }
    else
    {
        tok->timestamps.back().end = timestamps[2 * i + 1];
    }
}
```

### Commit
- `vieenrose/chatllm.cpp@fcc9a09`: Fix ForcedAligner crash when parent_id has gaps
- Merged into `feature/exp1-qwen3-asr` branch

### Potential Upstream
Consider upstreaming to `foldl/chatllm.cpp` - this is a robustness fix for edge cases where input text contains punctuation-only sentences.
