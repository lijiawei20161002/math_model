#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --reasoning-parser deepseek_r1
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vllm serve mistralai/Mistral-7B-Instruct-v0.3 --tokenizer-mode mistral --dtype bfloat16 --max-model-len 32768
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 vllm serve agentica-org/DeepScaleR-1.5B-Preview