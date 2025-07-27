import unittest
import torch
import flashinfer
from simple_flashinfer.manager import AutoKVCacheManager
from simple_flashinfer.utils import block_diagonal_concat_inverted, step_attention_mask_flatten

class TestManager(unittest.TestCase):
    def setUp(self):
        self.num_heads = 16
        self.head_dim = 128
        self.num_layers = 2

        self.manager = AutoKVCacheManager(self.num_layers, self.num_heads, self.head_dim, mem_utilization=0.2)

        workspace_buffer_prefill = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        workspace_buffer_decode = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer_prefill, "NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer_decode, "NHD"
        )

    def test_prefill_and_decode(self):
        
        i = 0

        lengths = [2, 10, 100, 200]
        q_at_layer = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        k_at_layer = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        v_at_layer = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")

        batch_ids = []
        for no, l in enumerate(lengths):
            self.manager.allocate(no, l)
            batch_ids.append(no)

        append_indptr = torch.cumsum(torch.tensor([0] + lengths), dim=-1).to(torch.int32).cuda()
        
        kv_indices, kv_indptr, kv_last_page_len = self.manager.get_append_metadata(batch_ids)
        print(append_indptr, kv_indices, kv_indptr, kv_last_page_len)

        self.prefill_wrapper.plan(
            append_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            self.manager.block_size,
            causal=True,
        )

        self.manager.append_paged_kv_cache(batch_ids, k_at_layer[i], v_at_layer[i], append_indptr, i)
        o = self.prefill_wrapper.run(q_at_layer[i], self.manager.kv_cache[i])

        masks = []
        for l in lengths:
            masks.append(torch.tril(torch.ones(l, l)))
            
        masks = block_diagonal_concat_inverted(*masks, dtype = torch.float16).cuda()

        q = q_at_layer[i][None].transpose(1, 2)
        k = k_at_layer[i][None].transpose(1, 2)
        v = v_at_layer[i][None].transpose(1, 2)
        output_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = masks[None])
        output_sdpa = output_sdpa[0].transpose(0, 1).argmax(-1)

        mean_match = (output_sdpa == o.argmax(-1)).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "argmax mismatch")

        lengths = [1, 1, 1, 1]
        q_at_layer = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        k_at_layer_decode = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        v_at_layer_decode = torch.randn(
            self.num_layers, sum(lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")

        batch_ids = []
        for no, l in enumerate(lengths):
            self.manager.append_tokens(no, l)
            batch_ids.append(no)
        
        append_indptr_decode = torch.cumsum(torch.tensor([0] + lengths), dim=-1).to(torch.int32).cuda()
        
        kv_indices, kv_indptr, kv_last_page_len = self.manager.get_append_metadata(batch_ids)
        print(append_indptr_decode, kv_indices, kv_indptr, kv_last_page_len)

        self.decode_wrapper.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            self.manager.block_size,
            pos_encoding_mode="NONE",
            data_type=torch.float16
        )

        self.manager.append_paged_kv_cache(
            batch_ids, k_at_layer_decode[i], v_at_layer_decode[i], append_indptr_decode, i)
        o = self.decode_wrapper.run(q_at_layer[i], self.manager.kv_cache[i])

        # page last batch
        page = kv_indices[kv_indptr[-2]:kv_indptr[-1]]
        last_page = kv_last_page_len[-1]

        k = []
        v = []
        """
        kv_cache = torch.zeros(
            (num_layers, self.max_blocks, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype, device="cuda"
        )
        """
        for n in range(len(page) - 1):
            k.append(self.manager.kv_cache[i, page[n], 0])
            v.append(self.manager.kv_cache[i, page[n], 1])
        
        k.append(self.manager.kv_cache[i, page[-1], 0, :last_page])
        v.append(self.manager.kv_cache[i, page[-1], 1, :last_page])

        k = torch.concat(k).cuda()
        v = torch.concat(v).cuda()

        print(k.shape, v.shape)
        # shape should 200 from the first length + 1 from the second length
        self.assertGreaterEqual(k.shape[0], 201, "argmax mismatch")

        actual_k = [
            k_at_layer[i, append_indptr[-2]: append_indptr[-1]],
            k_at_layer_decode[i, append_indptr_decode[-2]: append_indptr_decode[-1]],
        ]
        actual_k = torch.concat(actual_k).cuda()
        mean_match = (k == actual_k).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "K from pagedattention is not same with actual K")

        actual_v = [
            v_at_layer[i, append_indptr[-2]: append_indptr[-1]],
            v_at_layer_decode[i, append_indptr_decode[-2]: append_indptr_decode[-1]],
        ]
        actual_v = torch.concat(actual_v).cuda()
        mean_match = (v == actual_v).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "V from pagedattention is not same with actual V")

        k = k.transpose(0, 1)[None]
        v = v.transpose(0, 1)[None]
        actual_k = actual_k.transpose(0, 1)[None]
        actual_v = actual_v.transpose(0, 1)[None]

        print(k.shape, actual_k.shape)

        q = q_at_layer[0, -1:].transpose(0, 1)[None]
        print(q.shape, k.shape, v.shape)

        output_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        output_sdpa = output_sdpa[0, :, 0]

        single_decode_with_kv_cache = flashinfer.decode.single_decode_with_kv_cache(
            q[0,:, 0], k[0].transpose(0, 1).contiguous(), v[0].transpose(0, 1).contiguous())

        print(
            output_sdpa.argmax(-1),
            o[-1].argmax(-1), 
            single_decode_with_kv_cache.argmax(-1),
        )

        mean_match = (output_sdpa.argmax(-1) == o[-1].argmax(-1)).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "argmax mismatch")

if __name__ == "__main__":
    unittest.main()