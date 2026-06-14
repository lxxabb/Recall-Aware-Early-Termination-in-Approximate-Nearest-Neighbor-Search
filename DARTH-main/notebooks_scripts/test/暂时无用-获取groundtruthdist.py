import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def fvecs_write(fname, mat):
    """mat: (n, d) float32"""
    n, d = mat.shape
    header = np.full((n, 1), d, dtype='int32')
    out = np.hstack((header.view('float32'), mat)).astype('float32')
    out.tofile(fname)


# ----------- 读数据 -----------
# base = fvecs_read('/root/ann-data/GLOVE100/base.1183514.fvecs')          # (nb, d)
# queries = fvecs_read('/root/ann-data/GLOVE100/query.10K.fvecs')      # (nq, d)
# gt = ivecs_read('/root/ann-data/GLOVE100/query.groundtruth.10K.k1000.ivecs')     # (nq, 100)

# base = fvecs_read('/root/ann-data/GIST1M/gist_base.fvecs')          # (nb, d)
# queries = fvecs_read('/root/ann-data/GIST1M/gist_query.fvecs')      # (nq, d)
# gt = ivecs_read('/root/ann-data/GIST1M/gist_groundtruth.ivecs')     # (nq, 100)

# base = fvecs_read('/root/ann-data/DEEP10M/deep10M.fvecs')          # (nb, d)
# queries = fvecs_read('/root/ann-data/DEEP10M/deep10M_query_1wan.fvecs')      # (nq, d)
# gt = ivecs_read('/root/ann-data/DEEP10M/deep10M_query_1wan_groundtruth.ivecs')     # (nq, 100)
# gt = ivecs_read('/root/ann-data/DEEP10M/deep10M_query_1wan_groundtruth_distance.fvecs')     # (nq, 100)

base = fvecs_read('/root/ann-data/SIFT1M/sift_base.fvecs')          # (nb, d)
queries = fvecs_read('/root/ann-data/SIFT1M/sift_query.fvecs')      # (nq, d)
gt = ivecs_read('/root/ann-data/SIFT1M/sift_groundtruth.ivecs')     # (nq, 100)
nq, k = gt.shape
d = base.shape[1]

print(base.shape, queries.shape, gt.shape)


# ----------- 计算 GT 距离 -----------
# 结果：每个 query 有 k 个距离
gt_dist = np.empty((nq, k), dtype='float32')

for qi in range(nq):
    q = queries[qi]                 # (d,)
    neigh = base[gt[qi]]            # (k, d)

    diff = neigh - q                # (k, d)
    dist2 = np.sum(diff * diff, axis=1)   # L2-squared

    gt_dist[qi] = dist2


# ----------- 保存为 .fvecs -----------
# fvecs_write('/root/ann-data/GLOVE100/query.groundtruth.10K.k1000_distance.fvecs', gt_dist)

# print("Done. Saved to sift_groundtruth_distance.fvecs")
