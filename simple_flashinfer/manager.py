import torch
import math
import pynvml
import flashinfer
from transformers.cache_utils import Cache

def get_total_free_memory(index=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.free

class AutoKVCacheManager:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        layout: str = "NHD",
        total_gpu_mem_bytes: int = None,
        mem_utilization: float = 0.9,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.layout = layout.upper()
        self.dtype_size = torch.tensor([], dtype=dtype).element_size()

        if total_gpu_mem_bytes is None:
            total_gpu_mem_bytes = get_total_free_memory()
        
        usable_mem = int(total_gpu_mem_bytes * mem_utilization)
        per_token_bytes = num_kv_heads * head_dim * self.dtype_size * 2
        per_block_per_layer = block_size * per_token_bytes
        self.max_blocks = usable_mem // (num_layers * per_block_per_layer)

        self.kv_cache = torch.zeros(
            (num_layers, self.max_blocks, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype, device="cuda"
        )

        self.free_blocks = list(range(self.max_blocks))
        self.batch_to_blocks = {}
        self.batch_to_page_lengths = {}
        self.batch_to_total_tokens = {}

    def get_qo_indptr(self, batch_ids):
        indptr = [0]
        for bid in batch_ids:
            total = self.batch_to_total_tokens.get(bid, 0)
            indptr.append(indptr[-1] + total)
        return torch.tensor(indptr, dtype=torch.int32, device="cuda")
    
    def get_num_pages_per_batch(self):
        return torch.tensor(
            [len(self.batch_to_blocks[bid]) for bid in batch_ids], 
            dtype=torch.int32, device="cuda")

    def allocate(self, batch_id, total_tokens):
        num_pages = math.ceil(total_tokens / self.block_size)
        if len(self.free_blocks) < num_pages:
            raise RuntimeError("Not enough KV cache blocks available")

        blocks = [self.free_blocks.pop() for _ in range(num_pages)]
        self.batch_to_blocks[batch_id] = blocks
        self.batch_to_page_lengths[batch_id] = total_tokens % self.block_size
        self.batch_to_total_tokens[batch_id] = total_tokens
        return blocks

    def free(self, batch_id):
        blocks = self.batch_to_blocks.pop(batch_id, [])
        self.free_blocks.extend(blocks)
        self.batch_to_page_lengths.pop(batch_id, None)

    def get_append_metadata(self, batch_ids):
        """Returns kv_indices, kv_indptr, kv_last_page_len for FlashInfer append."""
        kv_indices = []
        kv_indptr = [0]
        kv_last_page_len = []

        for bid in batch_ids:
            pages = self.batch_to_blocks[bid]
            kv_indices.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            kv_last_page_len.append(self.batch_to_page_lengths[bid])

        return (
            torch.tensor(kv_indices, dtype=torch.int32, device="cuda"),
            torch.tensor(kv_indptr, dtype=torch.int32, device="cuda"),
            torch.tensor(kv_last_page_len, dtype=torch.int32, device="cuda"),
        )
    
    def append_paged_kv_cache(self, batch_ids, key, value, append_indptr, layer_idx):
        kv_indices, kv_indptr, kv_last_page_len = self.get_append_metadata(batch_ids)

        seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, self.block_size)
        batch_indices, positions = flashinfer.get_batch_indices_positions(
            append_indptr, seq_lens, append_indptr[-1]
        )

        flashinfer.page.append_paged_kv_cache(
            append_key=key,
            append_value=value,
            batch_indices=batch_indices,
            positions=positions,
            paged_kv_cache=self.kv_cache[layer_idx],
            kv_indices=kv_indices,
            kv_indptr=kv_indptr,
            kv_last_page_len=kv_last_page_len,
            kv_layout=self.layout,
        )
    
    def get_kv_cache(self, batch_ids):
        pass


    def append_tokens(self, batch_id, num_new_tokens):

        current_len = self.batch_to_page_lengths.get(batch_id, 0)
        total_tokens = current_len + num_new_tokens

        full_new_pages = total_tokens // self.block_size
        remaining = total_tokens % self.block_size

        if full_new_pages > len(self.free_blocks):
            raise RuntimeError("Not enough free blocks to append tokens")

        for _ in range(full_new_pages):
            new_page = self.free_blocks.pop()
            self.batch_to_blocks[batch_id].append(new_page)

        self.batch_to_page_lengths[batch_id] = remaining

    def stream_decode_step(self, batch_id):
        """Handles one streaming decode token per request."""

        current_len = self.batch_to_page_lengths[batch_id]

        if current_len >= self.block_size:
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache blocks for decode step")
            new_page = self.free_blocks.pop()
            self.batch_to_blocks[batch_id].append(new_page)
            self.batch_to_page_lengths[batch_id] = 0

        self.batch_to_page_lengths[batch_id] += 1

"""
For 4.53.3, CacheLayerMixin not yet exist in 4.53.1
class ManagerLayer(CacheLayerMixin):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs = None,
    ) :
        cache_position = cache_kwargs.get('cache_position')
        batch_ids, lengths = [], []
        for k, v in cache_position.items():
            batch_ids.append(k)
            lengths.append(v)
        append_indptr = torch.cumsum(torch.tensor([0] + lengths), dim = -1).cuda()
        self.manager.append_paged_kv_cache(batch_ids, key, value, append_indptr, self.layer_idx)
        
        return None, None

class ManagerCache(Cache):
    def __init__(self, manager, *args, **kwargs):
        super().__init__(layer_classes=ManagerLayer, *args, **kwargs)
        for i in range(len(self.layers)):
            self.layers[i].layer_idx = i
            self.layers[i].manager = manager
"""

class ManagerCache(Cache):
    def __init__(self, manager) -> None:
        super().__init__()
        self.manager = manager
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs = None,
    ):
        cache_position = cache_kwargs.get('cache_position')
        batch_ids, lengths = [], []
        for k, v in cache_position.items():
            batch_ids.append(k)
            lengths.append(v)
        append_indptr = torch.cumsum(torch.tensor([0] + lengths), dim = -1).cuda()
        self.manager.append_paged_kv_cache(batch_ids, key, value, append_indptr, self.layer_idx)
        return None, None