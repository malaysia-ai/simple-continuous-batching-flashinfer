import torch

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

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        if top_p > 0:
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