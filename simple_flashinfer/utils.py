import torch

def block_diagonal_concat_inverted(*masks, dtype=torch.bfloat16):
    total_size = sum(mask.size(0) for mask in masks)
    combined_mask = torch.zeros(total_size, total_size, dtype=dtype)

    current_pos = 0

    for mask in masks:
        size = mask.size(0)
        combined_mask[current_pos:current_pos + size, current_pos:current_pos + size] = mask
        current_pos += size

    min_value = torch.finfo(dtype).min if dtype.is_floating_point else torch.iinfo(dtype).min
    inverted_mask = torch.where(combined_mask == 1, torch.tensor(0, dtype=dtype), min_value)
    return inverted_mask.unsqueeze(0)

def step_attention_mask_flatten(seq_lens, dtype=torch.float16):
    num_queries = len(seq_lens)
    total_kv_cache_len = sum(seq_lens)
    neg_inf = torch.finfo(dtype).min
    mask = torch.full((1, 1, num_queries, total_kv_cache_len), neg_inf, dtype=dtype)
    start = 0
    for i, length in enumerate(seq_lens):
        end = start + length
        causal_mask = torch.triu(torch.full((length, length), neg_inf, dtype=dtype), diagonal=1)
        mask[:, :, i, start:end] = causal_mask[-1, :]
        start = end

    return mask