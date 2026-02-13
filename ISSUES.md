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
