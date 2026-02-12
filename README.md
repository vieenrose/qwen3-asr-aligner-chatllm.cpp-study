

# Experiments
## experiments/qwen3-asr-chatllm.cpp
1. create a docker container
2. to build chatllm.cpp library from source code ./chatllm.cpp 
3. then use or adapt its python binding ./chatllm.cpp/bindings/chatllm.py to use
   ./models/qwen3-asr-0.6b-q4_0.bin to transcribe phoneNumber1-zh-TW.wav in live with streaming output 
   then apply inverse text normalization from ./Chinese-ITN on it then show result after ITN in the end
