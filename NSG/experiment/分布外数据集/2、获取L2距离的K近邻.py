import os
import time
import numpy as np


# =========================
# 固定文件路径
# =========================
BASE_PATH = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/base.1B.fbin.crop_nb_1000000"
QUERY_PATH = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/query.heldout.30K.fbin"
OUTPUT_IVECS_PATH = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/gt100-heldout.30K.l2.ivecs"

# =========================
# 参数
# =========================
TOPK = 100

# 先小规模测试时可设成 100 / 1000；全量跑时设为 None
QUERY_LIMIT = None

# 分块大小，按内存情况调整
QUERY_BLOCK_SIZE = 64
BASE_BLOCK_SIZE = 50000


def read_fbin_header(path):
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
    if header.size != 2:
        raise ValueError(f"无法读取文件头: {path}")
    n = int(header[0])
    d = int(header[1])
    return n, d


def memmap_fbin(path):
    """
    fbin 格式:
    [int32 n][int32 d][float32 data...]
    返回:
        n, d, memmap数组(shape=(n,d), dtype=float32)
    """
    n, d = read_fbin_header(path)
    data = np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, d))
    return n, d, data


def write_ivecs(path, ids):
    """
    ivecs 每行格式:
    [int32 k][int32 id0][int32 id1]...[int32 id(k-1)]
    """
    ids = np.asarray(ids, dtype=np.int32, order="C")
    nq, k = ids.shape

    with open(path, "wb") as f:
        for i in range(nq):
            np.array([k], dtype=np.int32).tofile(f)
            ids[i].tofile(f)


def exact_topk_l2(base, queries, topk=100, query_block_size=64, base_block_size=50000):
    """
    对 queries 在 base 上做精确 L2 top-k 搜索。
    返回:
        all_ids:    shape=(nq, topk), int32
        all_dists:  shape=(nq, topk), float32
    """
    nb, d = base.shape
    nq, d2 = queries.shape
    assert d == d2, f"维度不一致: base d={d}, query d={d2}"

    # 预先计算 base 的平方范数
    print("预计算 base 向量平方范数...")
    t0 = time.time()
    base_norms = np.sum(np.asarray(base, dtype=np.float32) ** 2, axis=1, dtype=np.float32)
    print(f"base_norms 计算完成，用时 {time.time() - t0:.3f}s")

    all_ids = np.empty((nq, topk), dtype=np.int32)
    all_dists = np.empty((nq, topk), dtype=np.float32)

    num_q_blocks = (nq + query_block_size - 1) // query_block_size
    num_b_blocks = (nb + base_block_size - 1) // base_block_size

    start_time = time.time()

    for qb_idx, q_start in enumerate(range(0, nq, query_block_size), start=1):
        q_end = min(q_start + query_block_size, nq)
        q = np.asarray(queries[q_start:q_end], dtype=np.float32)
        m = q.shape[0]

        print(f"\n处理 query block {qb_idx}/{num_q_blocks}, queries [{q_start}, {q_end})")

        # 当前 query block 的平方范数
        q_norms = np.sum(q * q, axis=1, dtype=np.float32)

        # 当前最优 top-k
        best_dists = np.full((m, topk), np.inf, dtype=np.float32)
        best_ids = np.full((m, topk), -1, dtype=np.int32)

        block_start_time = time.time()

        for bb_idx, b_start in enumerate(range(0, nb, base_block_size), start=1):
            b_end = min(b_start + base_block_size, nb)

            xb = np.asarray(base[b_start:b_end], dtype=np.float32)
            xb_norms = base_norms[b_start:b_end]

            # 使用公式:
            # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q·x
            # 这里用的是平方L2，不开根号，排序结果不变
            dots = q @ xb.T
            dists = q_norms[:, None] + xb_norms[None, :] - 2.0 * dots

            # 数值误差下可能出现极小负数，截断到0
            np.maximum(dists, 0.0, out=dists)

            local_k = min(topk, dists.shape[1])

            # 先取当前 base block 内每个 query 的局部 top-k
            local_idx = np.argpartition(dists, kth=local_k - 1, axis=1)[:, :local_k]
            local_dists = np.take_along_axis(dists, local_idx, axis=1)
            local_ids = local_idx.astype(np.int32) + b_start

            # 与已有 best top-k 合并
            cand_dists = np.concatenate([best_dists, local_dists], axis=1)
            cand_ids = np.concatenate([best_ids, local_ids], axis=1)

            sel = np.argpartition(cand_dists, kth=topk - 1, axis=1)[:, :topk]
            best_dists = np.take_along_axis(cand_dists, sel, axis=1)
            best_ids = np.take_along_axis(cand_ids, sel, axis=1)

            if bb_idx % 5 == 0 or bb_idx == num_b_blocks:
                print(
                    f"  base block {bb_idx}/{num_b_blocks} "
                    f"[{b_start}, {b_end}) done"
                )

        # 对最终 top-k 再按距离升序排序
        order = np.argsort(best_dists, axis=1)
        best_dists = np.take_along_axis(best_dists, order, axis=1)
        best_ids = np.take_along_axis(best_ids, order, axis=1)

        all_dists[q_start:q_end] = best_dists
        all_ids[q_start:q_end] = best_ids

        print(
            f"query block 完成，用时 {time.time() - block_start_time:.3f}s, "
            f"示例 top10 ids: {best_ids[0, :10].tolist()}"
        )

    total_time = time.time() - start_time
    print(f"\n全部完成，总用时 {total_time:.3f}s")

    return all_ids, all_dists


def main():
    print("读取 base...")
    nb, db, base = memmap_fbin(BASE_PATH)
    print(f"base: n={nb}, d={db}")

    print("读取 query...")
    nq, dq, queries = memmap_fbin(QUERY_PATH)
    print(f"query: n={nq}, d={dq}")

    if db != dq:
        raise ValueError(f"base/query 维度不一致: {db} vs {dq}")

    if QUERY_LIMIT is not None:
        nq = min(nq, QUERY_LIMIT)
        queries = queries[:nq]
        print(f"仅处理前 {nq} 个 query")

    print("\n开始精确 L2 top-100 搜索...")
    ids, dists = exact_topk_l2(
        base=base,
        queries=queries,
        topk=TOPK,
        query_block_size=QUERY_BLOCK_SIZE,
        base_block_size=BASE_BLOCK_SIZE,
    )

    print("\n写出 ivecs...")
    write_ivecs(OUTPUT_IVECS_PATH, ids)
    print(f"ivecs 已保存到: {OUTPUT_IVECS_PATH}")

    print("\n结果检查:")
    print("ids.shape =", ids.shape)
    print("dists.shape =", dists.shape)
    print("第1个query的前10个邻居ID =", ids[0, :10].tolist())
    print("第1个query的前10个平方L2距离 =", dists[0, :10].tolist())


if __name__ == "__main__":
    main()