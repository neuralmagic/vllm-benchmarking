#!/bin/bash

gpu_type="4xA4000"

num_input_words=100
num_output_tokens=100
qps=5.0
num_prompts=20

model_name="mistralai/Mistral-7B-v0.1"
endpoint=/v1/completions

python3 benchmark_serving_synthetic.py \
    --model $model_name \
    --tokenizer $model_name \
    --endpoint $endpoint \
    --num-input-words $num_input_words \
    --num-output-tokens $num_output_tokens \
    --request-rate $qps \
    --num-prompts $num_prompts \
    --gpu-type $gpu_type