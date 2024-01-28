### vLLM Server Benchmarking

#### Launch Server

Launch the server with the following:

```bash
python3 vllm/entrypoints/api_server.py \
    --model robertgshaw2/llama-2-13b-chat-marlin \
    --max-model-len 2048 \
    --disable-log-requests
```