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
        self.lengths = [2, 10, 100, 200]

        self.manager = AutoKVCacheManager(self.num_layers, self.num_heads, self.head_dim, mem_utilization=0.2)

        self.q_at_layer = torch.randn(
            self.num_layers, sum(self.lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        self.k_at_layer = torch.randn(
            self.num_layers, sum(self.lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")
        self.v_at_layer = torch.randn(
            self.num_layers, sum(self.lengths), self.num_heads, self.head_dim
        ).half().to("cuda:0")

        self.batch_ids = []
        for no, l in enumerate(self.lengths):
            self.manager.allocate(no, l)
            self.batch_ids.append(no)

        self.append_indptr = torch.cumsum(torch.tensor([0] + self.lengths), dim=-1).to(torch.int32).cuda()
        self.manager.append_paged_kv_cache(
            self.batch_ids, self.k_at_layer[0], self.v_at_layer[0], self.append_indptr, 0
        )
        self.kv_indices, self.kv_indptr, self.kv_last_page_len = self.manager.get_append_metadata(
            self.batch_ids
        )

        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
        self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )
        self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "NHD"
        )

    def test_kv_cache_first_batch(self):
        i = 0
        mean_match = (
            self.manager.kv_cache[0][self.kv_indices[i]][0][: self.kv_last_page_len[i]]
            == self.k_at_layer[0, self.append_indptr[i] : self.append_indptr[i + 1]]
        ).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "First batch KV cache mismatch")

    def test_kv_cache_second_batch(self):
        i = 1
        mean_match = (
            self.manager.kv_cache[0][self.kv_indices[i]][0][: self.kv_last_page_len[i]]
            == self.k_at_layer[0, self.append_indptr[i] : self.append_indptr[i + 1]]
        ).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "Second batch KV cache mismatch")

    def test_prefill_attention(self):
        i = 0
        self.prefill_wrapper.plan(
            self.append_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            self.manager.block_size,
            causal=True,
        )
        o = self.prefill_wrapper.run(self.q_at_layer[i], self.manager.kv_cache[i])

        masks = []
        for l in self.lengths:
            masks.append(torch.tril(torch.ones(l, l)))
            
        masks = block_diagonal_concat_inverted(*masks, dtype = torch.float16).cuda()

        q = self.q_at_layer[i][None].transpose(1, 2)
        k = self.k_at_layer[i][None].transpose(1, 2)
        v = self.v_at_layer[i][None].transpose(1, 2)
        output_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = masks[None])
        output_sdpa = output_sdpa[0].transpose(0, 1).argmax(-1)

        mean_match = (output_sdpa == o.argmax(-1)).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "argmax mismatch")

    def test_decoding_attention(self):

        q_decode = torch.randn(len(self.lengths), self.num_heads, self.head_dim).half().to("cuda:0")

        i = 0
        self.decode_wrapper.plan(
            self.kv_indptr,
            self.kv_indices,
            self.kv_last_page_len,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            self.manager.block_size,
            pos_encoding_mode="NONE",
            data_type=torch.float16
        )
        
        o = self.decode_wrapper.run(q_decode, self.manager.kv_cache[i])

        q = q_decode[None].transpose(1, 2)
        k = self.k_at_layer[i][None].transpose(1, 2)
        v = self.v_at_layer[i][None].transpose(1, 2)

        step_mask = step_attention_mask_flatten(self.lengths).cuda()
        output_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = step_mask)
        output_sdpa = output_sdpa[0].transpose(0, 1).argmax(-1)

        mean_match = (output_sdpa == o.argmax(-1)).float().mean()
        self.assertGreaterEqual(mean_match, 0.99, "argmax mismatch")


if __name__ == "__main__":
    unittest.main()