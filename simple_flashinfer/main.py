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
    wrapper,
    manager,
    **kwargs,
):
    """
    For prefilling, it will pass flashinfer.BatchPrefillWithPagedKVCacheWrapper
    For step decoding, it will pass flashinfer.BatchDecodeWithPagedKVCacheWrapper
    """
    o = wrapper.run(query, manager.kv_cache[i])
    return o

app = FastAPI()

prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()

