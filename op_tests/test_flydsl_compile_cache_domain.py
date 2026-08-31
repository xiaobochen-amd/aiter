"""编译缓存容量必须覆盖 ni/ng 的定义域。

之前 maxsize 分别是 16 和 32，而两者的定义域都是 1..33 —— 一旦一个进程
见过的分裂数超过容量，就开始颠簸，每次重入约 20-31 ms 纯主机停顿。
本测试不需要 GPU：只检查 lru_cache 的容量声明。
"""
import unittest


import aiter.ops.flydsl.mla_reduce_kernels as R
import aiter.ops.flydsl.kernels.sparse_mla_decode as D

# 两个 guard 里写死的上界
NI_MAX = 33
NG_MAX = 33


class TestCompileCacheDomain(unittest.TestCase):
    def test_combine_cache_covers_ni_domain(self):
        info = R._compile_sparse_decode_direct_combine.cache_info()
        self.assertGreaterEqual(
            info.maxsize, NI_MAX,
            f"combine 缓存 {info.maxsize} 格 < ni 定义域 {NI_MAX}",
        )

    def test_partial_cache_covers_ng_domain(self):
        info = D.compile_sparse_mla_partial.cache_info()
        self.assertGreaterEqual(
            info.maxsize, NG_MAX,
            f"partial 缓存 {info.maxsize} 格 < ng 定义域 {NG_MAX}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
