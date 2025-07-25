import unittest
import torch
import flashinfer
from simple_flashinfer.manager import AutoKVCacheManager

class TestPrefill(unittest.TestCase):
    def setUp(self):
        self.num_heads = 16
        self.head_dim = 128
        self.num_layers = 2
        self.lengths = [2, 10, 100, 200]

        self.manager = AutoKVCacheManager(
            self.num_layers, self.num_heads, self.head_dim, mem_utilization=0.2
        )

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

        self.append_indptr = torch.cumsum(torch.tensor([0] + self.lengths), dim=-1).cuda()
        self.manager.append_paged_kv_cache(
            self.batch_ids, self.k_at_layer[0], self.v_at_layer[0], self.append_indptr, 0
        )
        self.kv_indices, self.kv_indptr, self.kv_last_page_len = self.manager.get_append_metadata(
            self.batch_ids
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


if __name__ == "__main__":
    unittest.main()