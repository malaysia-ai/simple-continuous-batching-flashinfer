# simple-continuous-batching-flashinfer

Simple continuous batching CausalLM from HuggingFace Transformer using FlashInfer.

## how to install

Using PIP with git,

```bash
pip3 install git+https://github.com/huseinzol05/simple-continuous-batching-flashinfer
```

Or you can git clone,

```bash
git clone https://github.com/huseinzol05/simple-continuous-batching-flashinfer && cd simple-continuous-batching-flashinfer
```

## how to local

### Supported parameters

```bash
python3 -m simple_flashinfer.main --help
```

```text
usage: main.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--reload RELOAD] [--microsleep MICROSLEEP]
               [--max_sequence MAX_SEQUENCE] [--memory_utilization MEMORY_UTILIZATION]
               [--compare-sdpa-prefill COMPARE_SDPA_PREFILL] [--model MODEL] [--torch_dtype TORCH_DTYPE]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --reload RELOAD       Enable hot loading (default: False, env: RELOAD)
  --microsleep MICROSLEEP
                        microsleep to group batching to reduce CPU burden, 1 / 1e-4 = 10k steps for second (default: 0.0001,
                        env: MICROSLEEP)
  --max_sequence MAX_SEQUENCE
                        max sequence aka batch size per filling or decoding (default: 128, env: MAX_SEQUENCE)
  --memory_utilization MEMORY_UTILIZATION
                        memory utilization on free memory after load the model for automatic number of paging for paged
                        attention (default: 0.9, env: MEMORY_UTILIZATION)
  --compare-sdpa-prefill COMPARE_SDPA_PREFILL
                        Compare FlashInfer attention output with SDPA during prefill (default: False, env:
                        COMPARE_SDPA_PREFILL)
  --model MODEL         Model type (default: meta-llama/Llama-3.2-1B-Instruct, env: MODEL)
  --torch_dtype TORCH_DTYPE
                        Model type (default: float16, env: TORCH_DTYPE)
```

**We support both args and OS environment**.

### Run meta-llama/Llama-3.2-1B-Instruct

```bash
python3 -m simple_flashinfer.main \
--host 0.0.0.0 --port 7088 --model meta-llama/Llama-3.2-1B-Instruct
```

```bash
curl -X 'POST' \
  'http://localhost:7088/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "model",
  "temperature": 0.9,
  "top_p": 0,
  "top_k": 0,
  "max_tokens": 256,
  "repetition_penalty": 1,
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "stream": true
}'
```

Output,

```
data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": "Hello", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": "!", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " How", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " can", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " I", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " assist", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " you", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": " today", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}

data: {"id": "bd793d41-3f8f-4142-b762-aec323664b83", "choices": [{"delta": {"content": "?", "function_call": null, "role": null, "tool_calls": null}, "finish_reason": null, "index": 0, "logprobs": null}], "created": 1753606017, "model": "model", "object": "chat.completion.chunk", "system_fingerprint": null}
```

## Unit tests

The unit tests will cover page append, prefilling causal attention and step decoding causal attention,

```bash
python3 -m unittest test.manager
python3 -m unittest test.manager_append
```

## Stress test

```bash
locust -f stress_test.py -P 7001 -H http://localhost:7088 -r 10 -u 50 -t 60
```

<img src="pics/locust-v3.png" width="50%">