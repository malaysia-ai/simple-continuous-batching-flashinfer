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
    o = prefill_wrapper.run(query, manager.kv_cache[i])
    return o

app = FastAPI()

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

