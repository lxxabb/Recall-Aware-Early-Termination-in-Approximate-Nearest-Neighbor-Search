#include "VectorDataLoader.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

const char query_type_str[4][20] = {
    "Training",
    "Validation",
    "Testing",
    "Noisy Testing"
};

float* fvecs_read(
        const char* fname,
        size_t* d_out,
        size_t* n_out,
        size_t limit) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        exit(1);
    }

    int d;
    fread(&d, 1, sizeof(int), f);

    assert((d > 0 && d < 1000000) && "unreasonable dimension");

    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    if (limit > 0 && n > limit) {
        n = limit;
        printf("Limiting dataset %s to %ld vectors\n", fname, n);
    }

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type)
    : query_type(query_type), noise_perc(""), dataset_name(dataset_name) {
}

VectorDataLoader::VectorDataLoader(std::string dataset_name, query_type_t query_type, std::string noise_perc, std::string directory_path)
    : VectorDataLoader(dataset_name, query_type) {
    this->directory_path = directory_path;
    this->noise_perc = noise_perc;
}

void VectorDataLoader::initializeDataMaps(){
    // SIFT1M
    baseVectorsMap["SIFT1M"] = directory_path + "SIFT1M/sift_base.fvecs";

    queryTypeToVectorsMap[TRAINING]["SIFT1M"] = directory_path + "SIFT1M/sift_learn.fvecs";
    queryTypeToVectorsMap[VALIDATION]["SIFT1M"] = directory_path + "SIFT1M/sift_validation_random1000.fvecs";
    queryTypeToVectorsMap[TESTING]["SIFT1M"] = directory_path + "SIFT1M/sift_query.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["SIFT1M"] = directory_path + "SIFT1M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["SIFT1M"] = directory_path + "SIFT1M/learn_groundtruth.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["SIFT1M"] = directory_path + "SIFT1M/sift_validation_random1000_groundtruth.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["SIFT1M"] = directory_path + "SIFT1M/sift_groundtruth.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["SIFT1M"] = directory_path + "SIFT1M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["SIFT1M"] = directory_path + "SIFT1M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["SIFT1M"] = directory_path + "SIFT1M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["SIFT1M"] = directory_path + "SIFT1M/sift_groundtruth_distance.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["SIFT1M"] = directory_path + "SIFT1M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    // GLOVE100
    baseVectorsMap["GLOVE100"] = directory_path + "GLOVE100/base.1183514.fvecs";

    queryTypeToVectorsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.100K.fvecs";
    queryTypeToVectorsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.10K.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.groundtruth.100K.k1000.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["GLOVE100"] = directory_path + "GLOVE100/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["GLOVE100"] = directory_path + "GLOVE100/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["GLOVE100"] = directory_path + "GLOVE100/query.groundtruth.10K.k1000_distance.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GLOVE100"] = directory_path + "GLOVE100/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    
    // DEEP10M
    baseVectorsMap["DEEP10M"] = directory_path + "DEEP10M/deep10M.fvecs";
    
    queryTypeToVectorsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/deep10M_learn_10w_new.fvecs";
    queryTypeToVectorsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.10K.fvecs";
    queryTypeToVectorsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/deep10M_query_1wan.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.noise" + noise_perc + ".fvecs";
    
    queryTypeToGroundtruthsMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/deep10M_learn_10wan_groundtruth_new.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.groundtruth.10K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/deep10M_query_1wan_groundtruth.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["DEEP10M"] = directory_path + "DEEP10M/learn.groundtruth.1M.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["DEEP10M"] = directory_path + "DEEP10M/validation.groundtruth.10K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["DEEP10M"] = directory_path + "DEEP10M/deep10M_query_1wan_groundtruth_distance.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["DEEP10M"] = directory_path + "DEEP10M/gauss_noisy_queries/query.10K.groundtruth.noise" + noise_perc + ".fvecs";

    // GIST1M
    baseVectorsMap["GIST1M"] = directory_path + "GIST1M/gist_base.fvecs";

    queryTypeToVectorsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/gist_learn.fvecs";
    queryTypeToVectorsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/gist_query.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/gist_learn_groundtruth_50wan.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["GIST1M"] = directory_path + "GIST1M/gist_groundtruth.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["GIST1M"] = directory_path + "GIST1M/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["GIST1M"] = directory_path + "GIST1M/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["GIST1M"] = directory_path + "GIST1M/gist_groundtruth_distance.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["GIST1M"] = directory_path + "GIST1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".fvecs";

    // GIST1M
    baseVectorsMap["T2I1M"] = directory_path + "T2I1M/base.1B.fbin.fvecs";

    queryTypeToVectorsMap[TRAINING]["T2I1M"] = directory_path + "T2I1M/query.heldout.30K.train2w.fvecs";
    queryTypeToVectorsMap[VALIDATION]["T2I1M"] = directory_path + "T2I1M/validation.1K.fvecs";
    queryTypeToVectorsMap[TESTING]["T2I1M"] = directory_path + "T2I1M/query.heldout.30K.query1w.fvecs";
    queryTypeToVectorsMap[NOISY_TESTING]["T2I1M"] = directory_path + "T2I1M/gauss_noisy_queries/query.1K.noise" + noise_perc + ".fvecs";

    queryTypeToGroundtruthsMap[TRAINING]["T2I1M"] = directory_path + "T2I1M/gt100-heldout.30K.l2.train2w.ivecs";
    queryTypeToGroundtruthsMap[VALIDATION]["T2I1M"] = directory_path + "T2I1M/validation.groundtruth.1K.k1000.ivecs";
    queryTypeToGroundtruthsMap[TESTING]["T2I1M"] = directory_path + "T2I1M/gt100-heldout.30K.l2.query1w.ivecs";
    queryTypeToGroundtruthsMap[NOISY_TESTING]["T2I1M"] = directory_path + "T2I1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".ivecs";

    queryTypeToGroundtruthDistancesMap[TRAINING]["T2I1M"] = directory_path + "T2I1M/learn.groundtruth.100K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[VALIDATION]["T2I1M"] = directory_path + "T2I1M/validation.groundtruth.1K.k1000.fvecs";
    queryTypeToGroundtruthDistancesMap[TESTING]["T2I1M"] = directory_path + "T2I1M/gist_groundtruth_distance.fvecs";
    queryTypeToGroundtruthDistancesMap[NOISY_TESTING]["T2I1M"] = directory_path + "T2I1M/gauss_noisy_queries/query.1K.groundtruth.noise" + noise_perc + ".fvecs";

}

float* VectorDataLoader::loadDB(size_t* d_out, size_t* n_out) {
    std::string baseVectorsPath = baseVectorsMap[dataset_name];
    
    printf(">> Loading base vectors from: %s\n", baseVectorsPath.c_str());
    float *db = fvecs_read(baseVectorsPath.c_str(), d_out, n_out);

    return db;
}

float* VectorDataLoader::loadQueries(size_t* d_out, size_t* n_out) {
    std::string queryVectorsPath = queryTypeToVectorsMap[query_type][dataset_name];
    
    printf(">> Loading queries from: %s\n", queryVectorsPath.c_str());
    float *queries = fvecs_read(queryVectorsPath.c_str(), d_out, n_out);

    // skip first 5000 queries:
    //*n_out -= 5000;
    //queries += 5000 * *d_out;

    return queries;
}

int* VectorDataLoader::loadQueriesGroundtruths(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthsPath = queryTypeToGroundtruthsMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruths from: %s\n", queryGroundtruthsPath.c_str());
    int *gt = ivecs_read(queryGroundtruthsPath.c_str(), k_out, n_out);

    return gt;
}

float* VectorDataLoader::loadQueriesGroundtruthDistances(size_t* k_out, size_t* n_out) {
    std::string queryGroundtruthDistancesPath = queryTypeToGroundtruthDistancesMap[query_type][dataset_name];
    
    printf(">> Loading queries groundtruth distances from: %s\n", queryGroundtruthDistancesPath.c_str());
    float *gt = fvecs_read(queryGroundtruthDistancesPath.c_str(), k_out, n_out);

    return gt;
}


