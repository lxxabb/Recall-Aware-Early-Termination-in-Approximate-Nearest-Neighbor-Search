#!/usr/bin/env python3
import os
import argparse
import numpy as np


def fbin_header(path):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.int32, count=2)
    return int(n), int(d)


def convert_fbin_to_fvecs(infile, outfile, chunk_size=100000):
    n, d = fbin_header(infile)
    print(f"[FBIN->FVECS] {infile}")
    print(f"  n={n}, d={d}, outfile={outfile}")

    with open(infile, "rb") as fin, open(outfile, "wb") as fout:
        # 跳过 fbin 的 8 字节头
        fin.seek(8)

        written = 0
        while written < n:
            cur = min(chunk_size, n - written)
            x = np.fromfile(fin, dtype=np.float32, count=cur * d)
            if x.size != cur * d:
                raise RuntimeError(
                    f"Unexpected EOF: expect {cur*d} float32, got {x.size}"
                )
            x = x.reshape(cur, d)

            # fvecs: 每个向量前加一个 int32 的维度 d
            # 为了高效写盘，把 float32 的位模式直接 view 成 int32
            out = np.empty((cur, d + 1), dtype=np.int32)
            out[:, 0] = d
            out[:, 1:] = x.view(np.int32)
            out.tofile(fout)

            written += cur
            print(f"  written {written}/{n}")

    print(f"[DONE] {outfile}\n")


def read_knn_gt_fbin(infile):
    with open(infile, "rb") as f:
        n, k = np.fromfile(f, dtype=np.uint32, count=2)
        n, k = int(n), int(k)

        ids = np.fromfile(f, dtype=np.int32, count=n * k)
        if ids.size != n * k:
            raise RuntimeError("Failed to read neighbor ids from GT file")
        ids = ids.reshape(n, k)

        # 后面的距离可读可不读；这里读一下做基本校验
        dist = np.fromfile(f, dtype=np.float32, count=n * k)
        if dist.size != n * k:
            raise RuntimeError("Failed to read distances from GT file")
        dist = dist.reshape(n, k)

    return ids, dist


def convert_gt_fbin_to_ivecs(infile, outfile):
    print(f"[GT-FBIN->IVECS] {infile}")
    ids, dist = read_knn_gt_fbin(infile)
    n, k = ids.shape
    print(f"  nq={n}, topk={k}, outfile={outfile}")

    # ivecs: 每行前加一个 int32 的长度 k
    out = np.empty((n, k + 1), dtype=np.int32)
    out[:, 0] = k
    out[:, 1:] = ids
    out.tofile(outfile)

    print(f"  distances discarded: shape={dist.shape}")
    print(f"[DONE] {outfile}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, help="input base .fbin")
    parser.add_argument("--query-public", type=str, help="input public query .fbin")
    parser.add_argument("--query-heldout", type=str, help="input heldout query .fbin")
    parser.add_argument("--gt-heldout", type=str, help="input heldout gt .fbin")
    parser.add_argument("--chunk-size", type=int, default=100000)
    args = parser.parse_args()

    if args.base:
        convert_fbin_to_fvecs(
            args.base,
            os.path.splitext(args.base)[0] + ".fvecs",
            chunk_size=args.chunk_size,
            )

    if args.query_public:
        convert_fbin_to_fvecs(
            args.query_public,
            os.path.splitext(args.query_public)[0] + ".fvecs",
            chunk_size=args.chunk_size,
            )

    if args.query_heldout:
        convert_fbin_to_fvecs(
            args.query_heldout,
            os.path.splitext(args.query_heldout)[0] + ".fvecs",
            chunk_size=args.chunk_size,
            )

    if args.gt_heldout:
        convert_gt_fbin_to_ivecs(
            args.gt_heldout,
            os.path.splitext(args.gt_heldout)[0] + ".ivecs",
            )


if __name__ == "__main__":
    main()