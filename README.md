# simple-continuous-batching-flashinfer

Simple continuous batching using FlashInfer.

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
usage: main.py [-h] [--host HOST] [--port PORT] [--loglevel LOGLEVEL] [--reload RELOAD] [--microsleep MICROSLEEP]
               [--max_sequence MAX_SEQUENCE] [--memory_utilization MEMORY_UTILIZATION] [--model MODEL]

Configuration parser

options:
  -h, --help            show this help message and exit
  --host HOST           host name to host the app (default: 0.0.0.0, env: HOSTNAME)
  --port PORT           port to host the app (default: 7088, env: PORT)
  --loglevel LOGLEVEL   Logging level (default: INFO, env: LOGLEVEL)
  --reload RELOAD       Enable hot loading (default: False, env: RELOAD)
  --microsleep MICROSLEEP
                        microsleep to group batching to reduce CPU burden, 1 / 1e-4 = 10k steps for second (default: 0.0001, env:
                        MICROSLEEP)
  --max_sequence MAX_SEQUENCE
                        max sequence aka batch size per filling or decoding (default: 128.0, env: MAX_SEQUENCE)
  --memory_utilization MEMORY_UTILIZATION
                        memory utilization on free memory after load the model for automatic number of paging for paged attention
                        (default: 0.9, env: MEMORY_UTILIZATION)
  --model MODEL         Model type (default: Qwen/Qwen3-0.6B-Base, env: MODEL)
```

**We support both args and OS environment**.

### Run Qwen/Qwen3-0.6B

```
python3.10 -m simple_flashinfer.main \
--host 0.0.0.0 --port 7088 --model Qwen/Qwen3-0.6B
```

## Unit tests

The unit tests will cover page append, prefilling causal attention and step decoding causal attention,

```bash
python3.10 -m unittest test.manager
```