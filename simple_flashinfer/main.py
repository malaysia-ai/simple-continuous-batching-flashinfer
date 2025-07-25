from simple_flashinfer.env import args
from fastapi import FastAPI, Request
from fastapi import HTTPException
from sse_starlette import EventSourceResponse
import asyncio
import flashinfer

def flashinfer_attention(
    module,
    query,
    key,
    value,
    prefill_wrapper,
    manager,
    batch_ids,
    prefill=True,
    **kwargs,
):
    manager.append_paged_kv_cache(batch_ids, key, value)
    if not prefill:
        key, value = manager.get_kv_cache(batch_ids)
    o = prefill_wrapper.run(query, key, value)
    return o

app = FastAPI()

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

