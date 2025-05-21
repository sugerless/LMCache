python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --backend vllm \
  --port 8080 \
  --model Qwen3-32B-FP8 \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 100 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --request-rate 10

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 VLLM_USE_V1=0 LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8081 \
  --tensor-parallel-size 2 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 VLLM_USE_V1=0 LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2,3 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8081 \
  --tensor-parallel-size 2 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 VLLM_USE_V1=0 LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=4,5 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8082 \
  --tensor-parallel-size 2 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 VLLM_USE_V1=0 LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=6,7 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8083 \
  --tensor-parallel-size 2 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'

## test 2
LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8081 1> vllm_1.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8082 1> vllm_2.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8083 1> vllm_3.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=3 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.4 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8084 1> vllm_4.log 2>&1 &



LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=4 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8085 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' 1> vllm_5.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=5 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8086 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' 1> vllm_6.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=6 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8087 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' 1> vllm_7.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8088 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' 1> vllm_8.log 2>&1 &

# test3
PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/LMCache/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=6 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8081 1> vllm_1.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8082 1> vllm_2.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=2 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8083 1> vllm_3.log 2>&1 &

LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=3 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8084 1> vllm_4.log 2>&1 &



HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=4 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8085 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 1> vllm_5.log 2>&1 &

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=5 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8086 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 1> vllm_6.log 2>&1 &

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=6 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8087 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 1> vllm_7.log 2>&1 &

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=7 \
vllm serve /models/Qwen/Qwen3-32B-FP8 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 \
  --chat-template /models/Qwen/Qwen3-32B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-32B-FP8 \
  --served_model_name Qwen3-32B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port 8088 \
  --tensor-parallel-size 1 \
  --block-size 128 \
  --max-model-len 32000 \
  --max-num-batched-tokens 32000 1> vllm_8.log 2>&1 &


python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --port 8091 \
  --model Qwen3-32B-FP8 \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path /models/dataset/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json \
  --profile \
  --num-prompts 20000 \
  --request-rate 128 \
  --max_concurrency 128 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 

python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --port 8081 \
  --model Qwen3-32B-FP8 \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path /models/dataset/agent_QA_test_15000.json \
  --profile \
  --num-prompts 15000 \
  --request-rate 400 \
  --max_concurrency 400 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8

python3 /vllm-workspace/benchmarks/benchmark_serving.py \
  --backend openai-chat \
  --port 8088 \
  --model Qwen3-32B-FP8 \
  --endpoint /v1/chat/completions \
  --dataset-name sharegpt \
  --dataset-path /models/dataset/ShareGPT_first_1500.json \
  --profile \
  --num-prompts 3300 \
  --request-rate 400 \
  --max_concurrency 400 \
  --tokenizer /models/Qwen/Qwen3-32B-FP8 


# test4
LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=/workspace/examples/blank_test/lmcache_config_fs.yaml \
HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /models/Qwen/Qwen3-235B-A22B-FP8 \
  --tokenizer /models/Qwen/Qwen3-235B-A22B-FP8 \
  --chat-template /models/Qwen/Qwen3-235B-A22B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-235B-A22B-FP8 \
  --served_model_name Qwen3-235B-A22B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --block-size 64 \
  --max-model-len 32000 \
  --enforce-eager \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-transfer-config     '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port 8081 1> vllm_1.log 2>&1 &

HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
vllm serve /models/Qwen/Qwen3-235B-A22B-FP8 \
  --tokenizer /models/Qwen/Qwen3-235B-A22B-FP8 \
  --chat-template /models/Qwen/Qwen3-235B-A22B-FP8/qwen3_nonthinking.jinja \
  --generation-config /models/Qwen/Qwen3-235B-A22B-FP8 \
  --served_model_name Qwen3-235B-A22B-FP8 \
  --gpu_memory_utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 0.0.0.0 \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 2048 \
  --enforce-eager \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --port 8082 1> vllm_2.log 2>&1 &

# docker run --rm -it   --name vllm   --gpus '"device=0,1,2,3"'   --ipc=host --network host   -e VLLM_NO_USAGE_STATS=1   -v /data/models/Qwen/Qwen3-235B-A22B-FP8:/data/models   artifactory.nioint.com/docker-virtual/llm/vllm-openai:lmcache-v1-2025-05-13 --model /data/models   --host 0.0.0.0 --port 8000 --trust_remote_code -tp 4   --served-model-name Qwen3-235B-A22B-FP8   --enforce-eager   --enable-chunked-prefill   --enable-prefix-caching   --gpu-memory-utilization 0.9   --enable-prefix-caching   --enable-chunked-prefill   --max-num-batched-tokens 2048   --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'