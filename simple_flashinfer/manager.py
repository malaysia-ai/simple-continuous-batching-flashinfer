import torch
import math
from typing import Dict, List, Tuple

class AutoKVCacheManager:
    def __init__(
        self,
        total_gpu_mem_bytes: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        layout: str = "NHD",
        mem_utilization: float = 0.9,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.layout = layout.upper()
        self.dtype_size = torch.tensor([], dtype=dtype).element_size()
        
        usable_mem = int(total_gpu_mem_bytes * mem_utilization)
        per_token_bytes = num_kv_heads * head_dim * self.dtype_size * 2
        per_block_per_layer = block_size * per_token_bytes
        self.max_blocks = usable_mem // (num_layers * per_block_per_layer)

        self.k_cache = torch.zeros(
            (self.max_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype, device="cuda"
        )
        self.v_cache = torch.zeros_like(self.k_cache)

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
        self.batch_to_page_lengths[batch_id] = 0
        return blocks

    def free(self, batch_id):
        blocks = self.batch_to_blocks.pop(batch_id, [])
        self.free_blocks.extend(blocks)
        self.batch_to_page_lengths.pop(batch_id, None)

    def get_paged_kv_cache(self):
        return self.k_cache, self.v_cache

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

    def append_tokens(self, batch_id, num_new_tokens):
        """Update `kv_last_page_len` and allocate new pages if needed."""
        if batch_id not in self.batch_to_blocks:
            raise ValueError(f"Batch {batch_id} not allocated")

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
        if batch_id not in self.batch_to_blocks:
            raise ValueError(f"Batch {batch_id} not allocated")

        current_len = self.batch_to_page_lengths[batch_id]

        if current_len >= self.block_size:
            if not self.free_blocks:
                raise RuntimeError("Out of KV cache blocks for decode step")
            new_page = self.free_blocks.pop()
            self.batch_to_blocks[batch_id].append(new_page)
            self.batch_to_page_lengths[batch_id] = 0

        self.batch_to_page_lengths[batch_id] += 1