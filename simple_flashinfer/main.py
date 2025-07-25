from simple_flashinfer.env import args, logging
from fastapi import FastAPI, Request
from fastapi import HTTPException
from sse_starlette import EventSourceResponse
from transformers import AttentionInterface
from transformers import AutoTokenizer, AutoModelForCausalLM
from .manager import AutoKVCacheManager
from .parameters import ChatCompletionForm

import torch
import json
import asyncio
import flashinfer
import uvicorn
import time
import uuid

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(
    logits,
    mask_penalty,
    temperature=1.0,
    top_k=None,
    top_p=None,
):
    logits = logits / mask_penalty
    if temperature > 0:
        logits = logits / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_p is not None:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            indices_to_remove = cumulative_probs > top_p
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0
            probs[sorted_indices[indices_to_remove]] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize

        idx_next = multinomial_sample_one_no_sync(probs)
    else:
        probs = logits
        idx_next = logits.argmax(-1, keepdim=True)

    return idx_next, probs

def flashinfer_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    """
    For prefilling, it will pass flashinfer.BatchPrefillWithPagedKVCacheWrapper
    For step decoding, it will pass flashinfer.BatchDecodeWithPagedKVCacheWrapper
    """
    wrapper = kwargs.get('wrapper')
    manager = kwargs.get('manager')
    prefill = kwargs.get('prefill')
    batch_ids = kwargs.get('batch_ids')
    append_indptr = kwargs.get('append_indptr')
    
    query = query[0].transpose(0, 1)
    key = key[0].transpose(0, 1)
    value = value[0].transpose(0, 1)

    layer_attr = 'prefill_layer_idx' if prefill else 'decode_layer_idx'
    layer_idx = getattr(manager, layer_attr)

    manager.append_paged_kv_cache(batch_ids, key, value, append_indptr, layer_idx)
    o = wrapper.run(query, manager.kv_cache[layer_idx])

    setattr(manager, layer_attr, layer_idx + 1)
    return o.transpose(0, 1)[None], None

def load_model():
    global tokenizer, model, manager
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="flashinfer_attention", 
        torch_dtype = torch.float16).cuda()
    num_layers = model.config.num_hidden_layers
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    manager = AutoKVCacheManager(
        num_layers, 
        num_key_value_heads, 
        head_dim, 
        mem_utilization=args.memory_utilization
    )

tokenizer = None
model = None
manager = None
AttentionInterface.register("flashinfer_attention", flashinfer_attention)
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
    response = None
    exception = None
    try:
        response = await call_next(request)
    except Exception as e:
        exception = e

    manager.free(request.state.request_id)
    duration = time.perf_counter() - start_time
    logging.info(f'freeing kv cache from {request.state.request_id}')
    logging.info(f"{request_id} completed in {duration:.4f}s")

    if exception is not None:
        raise exception
    return response

async def process_queue(queue, wrapper, is_prefill):
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(args.microsleep)

        need_sleep = True
        batch = []

        while not queue.empty():
            try:
                request = await asyncio.wait_for(queue.get(), timeout=1e-6)
                batch.append(request)
                if len(batch) >= args.max_sequence:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break

        if not batch:
            continue

        futures, inputs, position_ids, uuids = zip(*[(b[0], b[1], b[2], b[3]) for b in batch])
        lengths = [inp.shape[0] for inp in inputs]

        try:
            with torch.no_grad():
                position_ids = (
                    torch.cat([torch.arange(l) for l in lengths])
                    if is_prefill
                    else torch.tensor(position_ids)
                )[None].cuda()
                input_ids = torch.concat(inputs)[None].cuda()
                append_indptr = torch.cumsum(torch.tensor([0] + lengths), dim=-1).to(torch.int32).cuda()

                for no, l in enumerate(lengths):
                    if is_prefill:
                        manager.allocate(uuids[no], l)
                    else:
                        manager.append_tokens(uuids[no], l)

                kv_indices, kv_indptr, kv_last_page_len = manager.get_append_metadata(uuids)
                num_heads = model.config.num_attention_heads
                num_key_value_heads = model.config.num_key_value_heads
                head_dim = model.config.head_dim
                if is_prefill:
                    wrapper.plan(
                        append_indptr,
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        num_heads,
                        num_key_value_heads,
                        head_dim,
                        manager.block_size,
                        causal=True,
                    )
                else:
                    wrapper.plan(
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        num_heads,
                        num_key_value_heads,
                        head_dim,
                        manager.block_size,
                        pos_encoding_mode="NONE",
                        data_type=torch.float16
                    )
                setattr(manager, "prefill_layer_idx" if is_prefill else "decode_layer_idx", 0)

                output = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    use_cache=False,
                    wrapper=wrapper,
                    manager=manager,
                    prefill=is_prefill,
                    batch_ids=uuids,
                    append_indptr=append_indptr,
                )
                for i, fut in enumerate(futures):
                    fut.set_result((output.logits[0, lengths[i] - 1],))

        except Exception as e:
            for future in futures:
                if not future.done():
                    future.set_exception(e)

async def prefill():
    await process_queue(prefill_queue, prefill_wrapper, is_prefill=True)

async def step():
    await process_queue(step_queue, decode_wrapper, is_prefill=False)
    
async def stream(inputs, created, form, request):
    uuid = request.state.request_id
    mask_penalty = torch.ones((model.config.vocab_size,)).cuda()
    initial_length = inputs.shape[0]
    for k in range(form.max_tokens):
        is_disconnected = await request.is_disconnected()
        if is_disconnected:
            break
            
        if k == 0:
            q = prefill_queue
        else:
            q = step_queue

        l = k + initial_length
        future = asyncio.Future()
        await q.put((future, inputs, l, uuid))
        out = await future
        logits = out[0]
        idx_next, probs = logits_to_probs(
            logits,
            mask_penalty,
            temperature=form.temperature,
            top_k=form.top_k,
            top_p=form.top_p
        )
        
        mask_penalty[idx_next[0]] = form.repetition_penalty
        token = tokenizer.decode(idx_next)
        print(k, idx_next, token)

        if k == 0:
            request.state.time_first_token = time.time()
        
        if idx_next[0] == tokenizer.eos_token_id:
            break

        inputs = idx_next

        data = {
            'id': uuid,
            'choices': [
                {'delta': {
                    'content': token,
                    'function_call': None,
                    'role': None,
                    'tool_calls': None
                },
                    'finish_reason': None,
                    'index': 0,
                    'logprobs': None
                }
            ],
            'created': created,
            'model': 'model',
            'object': 'chat.completion.chunk',
            'system_fingerprint': None
        }
        yield json.dumps(data)
        await asyncio.sleep(0)

@app.get('/')
async def index(request: Request = None):
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
            if not isinstance(data, str):
                continue
            try:
                data = json.loads(data)
                tokens.append(data['choices'][0]['delta']['content'])
            except Exception as e:
                pass

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
                'prompt_tokens': len(inputs),
                'total_tokens': len(inputs) + len(tokens),
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