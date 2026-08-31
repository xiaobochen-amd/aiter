"""指针复用不得让缓存返回陈旧值。

_resolve_actual_max_splits 用 data_ptr 做键，而分配器会把释放掉的地址交给下一个
张量。旧实现下，一个新 CSR 落在旧地址上就会拿到别人的 split 上界 —— 而这个值
恰恰在 graph 捕获期无法重新推导，只能被信任。
"""
import unittest
from unittest import mock

import torch

import aiter.ops.flydsl.mla_reduce_kernels as R


class TestSplitCacheIdentity(unittest.TestCase):
    def setUp(self):
        R._ACTUAL_MAX_SPLITS_CACHE.clear()

    def test_hit_while_buffer_alive(self):
        t = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        with mock.patch.object(R, "derive_actual_max_splits", lambda x: 4):
            self.assertEqual(R._resolve_actual_max_splits(t), 4)
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            self.assertEqual(R._resolve_actual_max_splits(t), 4)

    def test_recycled_pointer_is_a_miss_not_a_stale_hit(self):
        t = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        key = (t.data_ptr(), t.numel())
        with mock.patch.object(R, "derive_actual_max_splits", lambda x: 4):
            R._resolve_actual_max_splits(t)
        self.assertIn(key, R._ACTUAL_MAX_SPLITS_CACHE)

        del t  # 地址回到分配器，缓存条目变陈旧
        # 构造一个落在同一 key 上的查询：捕获期必须报 miss，不能返回 4
        fake = torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
        with mock.patch.object(fake, "data_ptr", lambda: key[0]), \
             mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            got = R._resolve_actual_max_splits(fake)
        self.assertIsNone(got, "指针被复用后必须报 miss")
        self.assertNotIn(key, R._ACTUAL_MAX_SPLITS_CACHE, "陈旧条目应被清除")

    def test_capture_miss_returns_none(self):
        t = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: True):
            self.assertIsNone(R._resolve_actual_max_splits(t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
