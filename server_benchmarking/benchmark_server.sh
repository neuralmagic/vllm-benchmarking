
#!/bin/bash
model_name="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
num_input_words_lst=(128 256)
num_output_tokens_lst=(128 256)
qps_lst=(1.0 2.5 5.0 7.5 10.)
num_prompts=200
gpu_type="4xA4000"

for qps in ${qps_lst[@]}; do
    for num_input_words in ${num_input_words_lst[@]}; do
        for num_output_tokens in ${num_output_tokens_lst[@]}; do
            python3 benchmark_server.py \
                --model $model_name \
                --tokenizer $model_name \
                --endpoint /v1/completions \
                --num-input-words $num_input_words \
                --num-output-tokens $num_output_tokens \
                --request-rate $qps \
                --num-prompts $num_prompts \
                --gpu-type $gpu_type
        done
    done
done