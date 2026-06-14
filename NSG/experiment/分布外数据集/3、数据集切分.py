import numpy as np


# =========================
# 固定文件路径
# =========================
QUERY_PATH = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/query.heldout.30K.fvecs"
GT_PATH = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/gt100-heldout.30K.l2.ivecs"

QUERY_1W_OUT = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/query.heldout.30K.query1w.fvecs"
GT_1W_OUT = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/gt100-heldout.30K.l2.query1w.ivecs"

TRAIN_2W_OUT = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/query.heldout.30K.train2w.fvecs"
GT_TRAIN_2W_OUT = "/home/extra_home/lxx23125236/ali/big-ann-benchmarks-main/data/text2image1B/gt100-heldout.30K.l2.train2w.ivecs"


# =========================
# 参数
# =========================
QUERY_SIZE = 10000
SEED = 42


def read_fvecs(path):
    with open(path, "rb") as f:
        d = np.fromfile(f, dtype=np.int32, count=1)
        if d.size != 1:
            raise ValueError(f"无法读取 fvecs 维度: {path}")
        d = int(d[0])

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % (d + 1) != 0:
        raise ValueError(f"fvecs 文件大小异常: {path}")

    raw = raw.reshape(-1, d + 1)
    dims = raw[:, 0]
    if not np.all(dims == d):
        raise ValueError(f"fvecs 每行维度不一致: {path}")

    data = raw[:, 1:].copy().view(np.float32).reshape(-1, d)
    return data


def write_fvecs(path, data):
    data = np.asarray(data, dtype=np.float32, order="C")
    n, d = data.shape

    with open(path, "wb") as f:
        for i in range(n):
            np.array([d], dtype=np.int32).tofile(f)
            data[i].tofile(f)


def read_ivecs(path):
    with open(path, "rb") as f:
        k = np.fromfile(f, dtype=np.int32, count=1)
        if k.size != 1:
            raise ValueError(f"无法读取 ivecs 长度: {path}")
        k = int(k[0])

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size % (k + 1) != 0:
        raise ValueError(f"ivecs 文件大小异常: {path}")

    raw = raw.reshape(-1, k + 1)
    ks = raw[:, 0]
    if not np.all(ks == k):
        raise ValueError(f"ivecs 每行长度不一致: {path}")

    ids = raw[:, 1:].copy()
    return ids


def write_ivecs(path, ids):
    ids = np.asarray(ids, dtype=np.int32, order="C")
    n, k = ids.shape

    with open(path, "wb") as f:
        for i in range(n):
            np.array([k], dtype=np.int32).tofile(f)
            ids[i].tofile(f)


def main():
    query = read_fvecs(QUERY_PATH)
    gt = read_ivecs(GT_PATH)

    if len(query) != len(gt):
        raise ValueError(f"query 和 gt 行数不一致: {len(query)} vs {len(gt)}")

    n = len(query)
    if QUERY_SIZE <= 0 or QUERY_SIZE >= n:
        raise ValueError(f"QUERY_SIZE 必须在 1 到 {n-1} 之间")

    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    query_idx = indices[:QUERY_SIZE]
    train_idx = indices[QUERY_SIZE:]

    query_1w = query[query_idx]
    gt_1w = gt[query_idx]

    train_2w = query[train_idx]
    gt_train_2w = gt[train_idx]

    write_fvecs(QUERY_1W_OUT, query_1w)
    write_ivecs(GT_1W_OUT, gt_1w)
    write_fvecs(TRAIN_2W_OUT, train_2w)
    write_ivecs(GT_TRAIN_2W_OUT, gt_train_2w)

    print("切分完成")
    print(f"总样本数: {n}")
    print(f"查询集 1w: {query_1w.shape} -> {QUERY_1W_OUT}")
    print(f"查询集真值 1w: {gt_1w.shape} -> {GT_1W_OUT}")
    print(f"训练集 2w: {train_2w.shape} -> {TRAIN_2W_OUT}")
    print(f"训练集真值 2w: {gt_train_2w.shape} -> {GT_TRAIN_2W_OUT}")


if __name__ == "__main__":
    main()