[README.md](https://github.com/user-attachments/files/28401774/README.md)
# Recall-Aware-Early-Termination-in-Approximate-Nearest-Neighbor-Search
## Installation
The installation prerequisites can be found in the FAISS installation instructions, which can be found [here](./docs/INSTALL.md).
To run the project, it is required that LightGBM is installed and is put in  `/opt/OpenBLAS/`.
To compile FAISS with our project, use:

```bash
cmake -B build -S . # For the paper, we used: -G Ninja -DCMAKE_BUILD_TYPE=Release  -DFAISS_ENABLE_GPU=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=OpenBLAS -DBLAS_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so -DLAPACK_LIBRARIES=/opt/OpenBLAS/lib/libopenblas.so -DCMAKE_INCLUDE_PATH=/opt/OpenBLAS/include -DCMAKE_LIBRARY_PATH=/opt/OpenBLAS/lib
ninja -C build faiss -j # Compiles FAISS
ninja -C build -j hnsw_test # For the driver code of HNSW-based experiments
```

To use the python scripts and notebooks, install the required Python packages:
```bash
pip install -r requirements.txt
```

To reproduce experiments, see scripts in the `experiments` directory. 

## Datasets
Our evaluation utilized the following datasets: 
* [SIFT and GIST](http://corpus-texmex.irisa.fr/)
* [DEEP](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)
* [Glove](https://nlp.stanford.edu/projects/glove/)
* [T2I](https://github.com/harsha-simhadri/big-ann-benchmarks)



## About
Our experiments on the HNSW index were conducted based on [DARTH code](https://github.com/MChatzakis/DARTH).

This repository contains the implementation and integration of DARTH in the FAISS library, developed by Facebook Research. 

All original FAISS code and components remain under their respective licenses and rules as selected by the developers. 
Please refer to the FAISS license for details regarding using the original library. 
We do not claim any ownership or rights over the original FAISS library: all rights and acknowledgments are retained by the original authors.
