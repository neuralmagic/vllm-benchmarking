#!/bin/bash

model_ids=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "codellama/CodeLlama-34b-Instruct-hf" "meta-llama/Llama-2-70b-hf")

for model_id in ${model_ids[@]}; do
    echo "------ $model_id"
	echo ""

    python3 get_shapes.py --model $model_id

done
