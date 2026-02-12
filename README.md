

# Experiments
## exp1: experiments/qwen3-asr-chatllm.cpp
0. store in a file environment variables to control
   a. number of threads used to build chatllm.cpp in build time
   b. number of threads used to run inference in runtime
   c. context lenghth use for inference 
1. create a docker container
2. to build chatllm.cpp library from source code ./chatllm.cpp 
3. then use or adapt its python binding ./chatllm.cpp/bindings/chatllm.py to run in container, 
   to use ./models/qwen3-asr-0.6b-q4_0.bin to 
   transcribe ./samples/phoneNumber1-zh-TW.wav in live with streaming output in zh-TW with opencc-python-reimplemented, 
   then apply inverse text normalization from ./Chinese-ITN on it then show result after ITN in the end
   show bencharmk result on chatllm.cpp inference in the end on
   a. time to 1st token 
   b. gneneration speed
   c. memory overhead
   d. final result WER againt ground truth ./samples/phoneNumber1-zh-TW.txt
4. memory leak test: iteraive 3. 10 times but reuse the model object to track memory usage
