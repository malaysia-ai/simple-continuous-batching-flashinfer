from simple_flashinfer.env import args, logging
from fastapi import FastAPI, Request
from fastapi import HTTPException
from sse_starlette import EventSourceResponse
from transformers import AttentionInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
from .manager import AutoKVCacheManager, ManagerCache
from .parameters import ChatCompletionForm

import torch
import asyncio
import flashinfer
import uvicorn
import time
import uuid

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

def load_model():
    global tokenizer, model, manager, cache
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="flashinfer_attention", 
        torch_dtype = torch.float16).cuda()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim
    manager = AutoKVCacheManager(num_layers, num_heads, head_dim, mem_utilization = args.memory_utilization)
    cache = ManagerCache(manager)

tokenizer = None
model = None
manager = None
cache = None
AttentionInterface.register("flashinfer_attention", flashinfer_attention)
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
prefill_queue = asyncio.Queue()
step_queue = asyncio.Queue()
app = FastAPI()

@app.middleware("http")
async def add_request_id_and_time(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    logging.info(f"{request_id} completed in {duration:.4f}s")

    return response

async def prefill():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.microsleep)
        
        need_sleep = True
        batch = []
        while not prefill_queue.empty():
            try:
                request = await asyncio.wait_for(prefill_queue.get(), timeout=1e-6)
                batch.append(request)
                if len(batch) >= args.max_sequence:
                    need_sleep = False
                    break

            except asyncio.TimeoutError:
                break
            
        if not len(batch):
            continue
        
        futures = [batch[i][0] for i in range(len(batch))]
        inputs = [batch[i][1] for i in range(len(batch))]
        uuids = [batch[i][2] for i in range(len(batch))]

        lengths = [inputs[i].shape[0] for i in range(len(batch))]
        position_ids = torch.cat([torch.arange(l) for l in lengths])[None].cuda()
        input_ids = torch.concat(inputs)[None].cuda()
        cache_position = {uuids[i]: lengths[i] for i in range(len(batch))}
        append_indptr = torch.cumsum(torch.tensor([0] + lengths), dim=-1).to(torch.int32).cuda()

        for no, l in enumerate(lengths):
            manager.allocate(uuids, l)
        
        kv_indices, kv_indptr, kv_last_page_len = manager.get_append_metadata(uuids)
        prefill_wrapper.plan(
            append_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            manager.block_size,
            causal=True,
        )

        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            wrapper=prefill_wrapper,
            manager=manager,
        )
        print(output)



async def step():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.microsleep)

async def stream(inputs, created, form, request):
    
    uuid = request.state.request_id
    initial_length = inputs.shape[0]
    for k in range(form.max_tokens):
        if k == 0:
            q = prefill_queue
        else:
            q = step_queue

        future = asyncio.Future()
        await q.put((future, inputs, uuid))
        break

@app.get('/')
async def index(request: Request = None):
    is_disconnected = await request.is_disconnected()
    return {'message': 'hello'}

@app.get('/kv_cache')
async def index(request: Request = None):
    total_kv_cache = manager.max_blocks * manager.block_size
    free_kv_cache = len(manager.free_blocks) * manager.block_size
    utilized_kv_cache = total_kv_cache - free_kv_cache
    return {
        'total_kv_cache': total_kv_cache,
        'free_kv_cache': free_kv_cache,
        'utilized_kv_cache': utilized_kv_cache,
    }

@app.post('/chat/completions')
async def chat_completions_main(
    form: ChatCompletionForm,
    request: Request = None,
):
    prompt = tokenizer.apply_chat_template(form.messages, tokenize=False)
    inputs = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False)[0]

    created = int(time.time())
    func = stream(inputs=inputs, created=created, form=form, request=request)

    if form.stream:
        r = func
    else:
        tokens = []
        async for data in func:
            if isinstance(data, ServerSentEvent):
                continue
            data = json.loads(data)
            tokens.append(data['choices'][0]['delta']['content'])

        data = {
            'id': request.state.request_id,
            'choices': [
                {'finish_reason': 'stop',
                 'index': 0,
                 'logprobs': None,
                 'message': {
                     'content': ''.join(tokens),
                     'role': 'assistant',
                     'function_call': None,
                     'tool_calls': None
                 },
                 'stop_reason': None
                 }
            ],
            'created': created,
            'model': 'model',
            'object': 'chat.completion',
            'system_fingerprint': None,
            'usage': {
                'completion_tokens': len(tokens),
                'prompt_tokens': len(inputs[0]),
                'total_tokens': len(inputs[0]) + len(tokens),
            }
        }
        r = data

    if form.stream:
        return EventSourceResponse(r, headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        })
    else:
        return r

@app.on_event("startup")
async def startup_event():
    load_model()
    app.state.background_prefill = asyncio.create_task(prefill())
    app.state.background_step = asyncio.create_task(step())
    
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.loglevel.lower(),
        access_log=True,
    )