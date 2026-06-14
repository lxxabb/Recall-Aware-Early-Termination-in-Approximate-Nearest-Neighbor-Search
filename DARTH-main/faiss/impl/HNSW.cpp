/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/prefetch.h>
#include <sys/time.h>

#include <string>

#ifdef __AVX2__
#include <immintrin.h>
#include <fstream>
#include <limits>
#include <type_traits>
#endif

namespace faiss {

/**************************************************************
 * HNSW structure implementation
 **************************************************************/

    int HNSW::nb_neighbors(int layer_no) const {
        return cum_nneighbor_per_level[layer_no + 1] -
               cum_nneighbor_per_level[layer_no];
    }

    void HNSW::set_nb_neighbors(int level_no, int n) {
        FAISS_THROW_IF_NOT(levels.size() == 0);
        int cur_n = nb_neighbors(level_no);
        for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
            cum_nneighbor_per_level[i] += n - cur_n;
        }
    }

    int HNSW::cum_nb_neighbors(int layer_no) const {
        return cum_nneighbor_per_level[layer_no];
    }

    void HNSW::neighbor_range(idx_t no, int layer_no, size_t *begin, size_t *end)
    const {
        size_t o = offsets[no];
        *begin = o + cum_nb_neighbors(layer_no);
        *end = o + cum_nb_neighbors(layer_no + 1);
    }

    HNSW::HNSW(int M) : rng(12345) {
        set_default_probas(M, 1.0 / log(M));
        offsets.push_back(0);
    }

    int HNSW::random_level() {
        double f = rng.rand_float();
        // could be a bit faster with bissection
        for (int level = 0; level < assign_probas.size(); level++) {
            if (f < assign_probas[level]) {
                return level;
            }
            f -= assign_probas[level];
        }
        // happens with exponentially low probability
        return assign_probas.size() - 1;
    }

    void HNSW::set_default_probas(int M, float levelMult) {
        int nn = 0;
        cum_nneighbor_per_level.push_back(0);
        for (int level = 0;; level++) {
            float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
            if (proba < 1e-9)
                break;
            assign_probas.push_back(proba);
            nn += level == 0 ? M * 2 : M;
            cum_nneighbor_per_level.push_back(nn);
        }
    }

    void HNSW::clear_neighbor_tables(int level) {
        for (int i = 0; i < levels.size(); i++) {
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                neighbors[j] = -1;
            }
        }
    }

    void HNSW::reset() {
        max_level = -1;
        entry_point = -1;
        offsets.clear();
        offsets.push_back(0);
        levels.clear();
        neighbors.clear();
    }

    void HNSW::print_neighbor_stats(int level) const {
        FAISS_THROW_IF_NOT(level < cum_nneighbor_per_level.size());
        printf("stats on level %d, max %d neighbors per vertex:\n",
               level,
               nb_neighbors(level));
        size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+ : tot_neigh) reduction(+ : tot_common) \
    reduction(+ : tot_reciprocal) reduction(+ : n_node)
        for (int i = 0; i < levels.size(); i++) {
            if (levels[i] > level) {
                n_node++;
                size_t begin, end;
                neighbor_range(i, level, &begin, &end);
                std::unordered_set<int> neighset;
                for (size_t j = begin; j < end; j++) {
                    if (neighbors[j] < 0)
                        break;
                    neighset.insert(neighbors[j]);
                }
                int n_neigh = neighset.size();
                int n_common = 0;
                int n_reciprocal = 0;
                for (size_t j = begin; j < end; j++) {
                    storage_idx_t i2 = neighbors[j];
                    if (i2 < 0)
                        break;
                    FAISS_ASSERT(i2 != i);
                    size_t begin2, end2;
                    neighbor_range(i2, level, &begin2, &end2);
                    for (size_t j2 = begin2; j2 < end2; j2++) {
                        storage_idx_t i3 = neighbors[j2];
                        if (i3 < 0)
                            break;
                        if (i3 == i) {
                            n_reciprocal++;
                            continue;
                        }
                        if (neighset.count(i3)) {
                            neighset.erase(i3);
                            n_common++;
                        }
                    }
                }
                tot_neigh += n_neigh;
                tot_common += n_common;
                tot_reciprocal += n_reciprocal;
            }
        }
        float normalizer = n_node;
        printf("   nb of nodes at that level %zd\n", n_node);
        printf("   neighbors per node: %.2f (%zd)\n",
               tot_neigh / normalizer,
               tot_neigh);
        printf("   nb of reciprocal neighbors: %.2f\n",
               tot_reciprocal / normalizer);
        printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%zd)\n",
               tot_common / normalizer,
               tot_common);
    }

    void HNSW::fill_with_random_links(size_t n) {
        int max_level = prepare_level_tab(n);
        RandomGenerator rng2(456);

        for (int level = max_level - 1; level >= 0; --level) {
            std::vector<int> elts;
            for (int i = 0; i < n; i++) {
                if (levels[i] > level) {
                    elts.push_back(i);
                }
            }
            printf("linking %zd elements in level %d\n", elts.size(), level);

            if (elts.size() == 1)
                continue;

            for (int ii = 0; ii < elts.size(); ii++) {
                int i = elts[ii];
                size_t begin, end;
                neighbor_range(i, 0, &begin, &end);
                for (size_t j = begin; j < end; j++) {
                    int other = 0;
                    do {
                        other = elts[rng2.rand_int(elts.size())];
                    } while (other == i);

                    neighbors[j] = other;
                }
            }
        }
    }

    int HNSW::prepare_level_tab(size_t n, bool preset_levels) {
        size_t n0 = offsets.size() - 1;

        if (preset_levels) {
            FAISS_ASSERT(n0 + n == levels.size());
        } else {
            FAISS_ASSERT(n0 == levels.size());
            for (int i = 0; i < n; i++) {
                int pt_level = random_level();
                levels.push_back(pt_level + 1);
            }
        }

        int max_level = 0;
        for (int i = 0; i < n; i++) {
            int pt_level = levels[i + n0] - 1;
            if (pt_level > max_level)
                max_level = pt_level;
            offsets.push_back(offsets.back() + cum_nb_neighbors(pt_level + 1));
            neighbors.resize(offsets.back(), -1);
        }

        return max_level;
    }

/** Enumerate vertices from nearest to farthest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
    void HNSW::shrink_neighbor_list(
            DistanceComputer &qdis,
            std::priority_queue<NodeDistFarther> &input,
            std::vector<NodeDistFarther> &output,
            int max_size) {
        while (input.size() > 0) {
            NodeDistFarther v1 = input.top();
            input.pop();
            float dist_v1_q = v1.d;

            bool good = true;
            for (NodeDistFarther v2: output) {
                float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

                if (dist_v1_v2 < dist_v1_q) {
                    good = false;
                    break;
                }
            }

            if (good) {
                output.push_back(v1);
                if (output.size() >= max_size) {
                    return;
                }
            }
        }
    }

    namespace {

        using storage_idx_t = HNSW::storage_idx_t;
        using NodeDistCloser = HNSW::NodeDistCloser;
        using NodeDistFarther = HNSW::NodeDistFarther;

/**************************************************************
 * Addition subroutines
 **************************************************************/

/// remove neighbors from the list to make it smaller than max_size
        void shrink_neighbor_list(
                DistanceComputer &qdis,
                std::priority_queue<NodeDistCloser> &resultSet1,
                int max_size) {
            if (resultSet1.size() < max_size) {
                return;
            }
            std::priority_queue<NodeDistFarther> resultSet;
            std::vector<NodeDistFarther> returnlist;

            while (resultSet1.size() > 0) {
                resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
                resultSet1.pop();
            }

            HNSW::shrink_neighbor_list(qdis, resultSet, returnlist, max_size);

            for (NodeDistFarther curen2: returnlist) {
                resultSet1.emplace(curen2.d, curen2.id);
            }
        }

/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
        void add_link(
                HNSW &hnsw,
                DistanceComputer &qdis,
                storage_idx_t src,
                storage_idx_t dest,
                int level) {
            size_t begin, end;
            hnsw.neighbor_range(src, level, &begin, &end);
            if (hnsw.neighbors[end - 1] == -1) {
                // there is enough room, find a slot to add it
                size_t i = end;
                while (i > begin) {
                    if (hnsw.neighbors[i - 1] != -1)
                        break;
                    i--;
                }
                hnsw.neighbors[i] = dest;
                return;
            }

            // otherwise we let them fight out which to keep

            // copy to resultSet...
            std::priority_queue<NodeDistCloser> resultSet;
            resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
            for (size_t i = begin; i < end; i++) {  // HERE WAS THE BUG
                storage_idx_t neigh = hnsw.neighbors[i];
                resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
            }

            shrink_neighbor_list(qdis, resultSet, end - begin);

            // ...and back
            size_t i = begin;
            while (resultSet.size()) {
                hnsw.neighbors[i++] = resultSet.top().id;
                resultSet.pop();
            }
            // they may have shrunk more than just by 1 element
            while (i < end) {
                hnsw.neighbors[i++] = -1;
            }
        }

/// search neighbors on a single level, starting from an entry point
        void search_neighbors_to_add(
                HNSW &hnsw,
                DistanceComputer &qdis,
                std::priority_queue<NodeDistCloser> &results,
                int entry_point,
                float d_entry_point,
                int level,
                VisitedTable &vt) {
            // top is nearest candidate
            std::priority_queue<NodeDistFarther> candidates;

            NodeDistFarther ev(d_entry_point, entry_point);
            candidates.push(ev);
            results.emplace(d_entry_point, entry_point);
            vt.set(entry_point);

            while (!candidates.empty()) {
                // get nearest
                const NodeDistFarther &currEv = candidates.top();

                if (currEv.d > results.top().d) {
                    break;
                }
                int currNode = currEv.id;
                candidates.pop();

                // loop over neighbors
                size_t begin, end;
                hnsw.neighbor_range(currNode, level, &begin, &end);
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t nodeId = hnsw.neighbors[i];
                    if (nodeId < 0)
                        break;
                    if (vt.get(nodeId))
                        continue;
                    vt.set(nodeId);

                    float dis = qdis(nodeId);
                    NodeDistFarther evE1(dis, nodeId);

                    if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                        results.emplace(dis, nodeId);
                        candidates.emplace(dis, nodeId);
                        if (results.size() > hnsw.efConstruction) {
                            results.pop();
                        }
                    }
                }
            }
            vt.advance();
        }

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
        void greedy_update_nearest(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                int level,
                storage_idx_t &nearest,
                float &d_nearest) {
            for (;;) {
                storage_idx_t prev_nearest = nearest;

                size_t begin, end;
                hnsw.neighbor_range(nearest, level, &begin, &end);
                for (size_t i = begin; i < end; i++) {
                    storage_idx_t v = hnsw.neighbors[i];
                    if (v < 0)
                        break;
                    float dis = qdis(v);
                    if (dis < d_nearest) {
                        nearest = v;
                        d_nearest = dis;
                    }
                }
                if (nearest == prev_nearest) {
                    return;
                }
            }
        }

    }  // namespace

/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
    void HNSW::add_links_starting_from(
            DistanceComputer &ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            int level,
            omp_lock_t *locks,
            VisitedTable &vt) {
        std::priority_queue<NodeDistCloser> link_targets;

        search_neighbors_to_add(
                *this, ptdis, link_targets, nearest, d_nearest, level, vt);

        // but we can afford only this many neighbors
        int M = nb_neighbors(level);

        ::faiss::shrink_neighbor_list(ptdis, link_targets, M);

        std::vector<storage_idx_t> neighbors;
        neighbors.reserve(link_targets.size());
        while (!link_targets.empty()) {
            storage_idx_t other_id = link_targets.top().id;
            add_link(*this, ptdis, pt_id, other_id, level);
            neighbors.push_back(other_id);
            link_targets.pop();
        }

        omp_unset_lock(&locks[pt_id]);
        for (storage_idx_t other_id: neighbors) {
            omp_set_lock(&locks[other_id]);
            add_link(*this, ptdis, other_id, pt_id, level);
            omp_unset_lock(&locks[other_id]);
        }
        omp_set_lock(&locks[pt_id]);
    }

/**************************************************************
 * Building, parallel
 **************************************************************/

    void HNSW::add_with_locks(
            DistanceComputer &ptdis,
            int pt_level,
            int pt_id,
            std::vector<omp_lock_t> &locks,
            VisitedTable &vt) {
        //  greedy search on upper levels

        storage_idx_t nearest;
#pragma omp critical
        {
            nearest = entry_point;

            if (nearest == -1) {
                max_level = pt_level;
                entry_point = pt_id;
            }
        }

        if (nearest < 0) {
            return;
        }

        omp_set_lock(&locks[pt_id]);

        int level = max_level;  // level at which we start adding neighbors
        float d_nearest = ptdis(nearest);

        for (; level > pt_level; level--) {
            greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
        }

        for (; level >= 0; level--) {
            add_links_starting_from(
                    ptdis, pt_id, nearest, d_nearest, level, locks.data(), vt);
        }

        omp_unset_lock(&locks[pt_id]);

        if (pt_level > max_level) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

/**************************************************************
 * Searching
 **************************************************************/

    namespace {
        using MinimaxHeap = HNSW::MinimaxHeap;
        using Node = HNSW::Node;
        using C = HNSW::C;

/** Do a BFS on the candidates list */
// just used as a lower bound for the minmaxheap, but it is set for heap search
        int extract_k_from_ResultHandler(ResultHandler<C> &res) {
            using RH = HeapBlockResultHandler<C>;
            if (auto hres = dynamic_cast<RH::SingleResultHandler *>(&res)) {
                return hres->k;
            }
            return 1;
        }

        int search_from_candidates(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int nres = nres_in;
            int ndis = 0;

            // can be overridden by search params
            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;

            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);

                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // // baseline version
                // for (size_t j = begin; j < end; j++) {
                //     int v1 = hnsw.neighbors[j];
                //     if (v1 < 0)
                //         break;
                //     if (vt.get(v1)) {
                //         continue;
                //     }
                //     vt.set(v1);
                //     ndis++;
                //     float d = qdis(v1);
                //     if (!sel || sel->is_member(v1)) {
                //         if (nres < k) {
                //             faiss::maxheap_push(++nres, D, I, d, v1);
                //         } else if (d < D[0]) {
                //             faiss::maxheap_replace_top(nres, D, I, d, v1);
                //         }
                //     }
                //     candidates.push(v1, d);
                // }

                // the following version processes 4 neighbors at a time
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                ndis += jmax - begin;
                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                        }
                    }
                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            add_to_heap(saved_j[id4], dis[id4]);
                        }

                        counter = 0;
                    }
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    add_to_heap(saved_j[icnt], dis);
                }

                nstep++;
                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            return nres;
        }

        double per_query_recall_at_k(
                const idx_t *gt,
                const idx_t query_idx,
                const idx_t *retrieved,
                int k) {
            std::unordered_set<idx_t> gt_set(
                    gt + query_idx * k, gt + (query_idx + 1) * k);
            int matches = 0;

            for (int j = 0; j < k; ++j) {
                if (gt_set.count(retrieved[query_idx * k + j])) {
                    matches++;
                }
            }

            double recall = (double) matches / (double) k;
            return recall;
        }

// Maybe move this to header?
//double elapsed_secs() {
//  struct timeval tv;
//  gettimeofday(&tv, nullptr);
//  return tv.tv_sec + tv.tv_usec * 1e-6;
//}

        int search_from_candidates_declarative_recall_data_generation(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DeclarativeRecallDataCollectorHNSW data_collector,
                std::string *observations_table,
                idx_t query_idx_for_table,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            std::string query_observations_str = "";

            int nres = nres_in;
            int ndis = 0;

            int logging_interval = data_collector.logging_interval;

            bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int nstep = 0;
            int visited_points=0;
            int duration=1;

            int total_insertions = 0;
            float first_nn_dist = -1;
            float prev_recall = 0;

            double search_time_start = data_collector.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    // ndis++; // FAISS does not increase ndis in the first phase of the base layer search.
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (first_nn_dist == -1) {  // if the first NN distance is not yet set.
                            first_nn_dist = d;
                        }
                    }

                    // improvement: Do not call the data_collector if the recall is 0. (TODO)
                    double recall_k = data_collector.data_manager.get_recallk(query_idx);
                    double elapsed = data_collector.data_manager.elapsed_secs() - search_time_start;

                    query_observations_str += data_collector.get_observation_data_str(
                            query_idx,
                            nstep,
                            ndis,
                            total_insertions,
                            elapsed,
                            first_nn_dist,
                            recall_k,
                            0,
                            duration);
                }
                vt.set(v1);
            }

            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=data_collector.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }

                            if (logging_interval == INTERVAL_WHEN_RECALL_UPDATES_VALUE) {
                                float recall = data_collector.data_manager.get_recallk(query_idx);

                                if (recall != prev_recall) {
                                    query_observations_str +=
                                            data_collector.get_observation_data_str(
                                                    query_idx,
                                                    nstep,
                                                    ndis,
                                                    total_insertions,
                                                    data_collector.data_manager.elapsed_secs() - search_time_start,
                                                    first_nn_dist,
                                                    recall,
                                                    visited_points,
                                                    duration);
                                    prev_recall = recall;
                                }
                            }
                        }

                        // improvement. Do not call the data_collector if the recall is 0! (TODO)
                        if (logging_interval != INTERVAL_WHEN_RECALL_UPDATES_VALUE && ndis % logging_interval == 0) {
                            query_observations_str +=
                                    data_collector.get_observation_data_str(
                                            query_idx,
                                            nstep,
                                            ndis,
                                            total_insertions,
                                            data_collector.data_manager.elapsed_secs() - search_time_start,
                                            first_nn_dist,
                                            data_collector.data_manager.get_recallk(query_idx),
                                            visited_points,
                                            duration);
                        }
                    }

                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            add_to_heap(saved_j[id4], dis[id4]);
                        }

                        counter = 0;
                    }
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);

                    ndis++;
                    add_to_heap(saved_j[icnt], dis);
                }

                nstep++;
                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            // We always log the last observation.
            query_observations_str += data_collector.get_observation_data_str(
                    query_idx,
                    nstep,//nstep（pop-min 次数 → 搜索步数）
                    ndis,
                    total_insertions,//（加入 top-k 次数）
                    data_collector.data_manager.elapsed_secs() - search_time_start,//（搜索时间）
                    first_nn_dist,//（第一次命中的距离）
                    data_collector.data_manager.get_recallk(query_idx),
                    visited_points,
                    duration);//（当前 recall@k）

            if (observations_table) {
                observations_table[query_idx_for_table] = query_observations_str;
            }

            return nres;
        }
        //每个查询点在该准确率最多收集10条数据，准确率为1最多收集20条数据
        int search_from_candidates_declarative_recall_data_generation_ourTrainData(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DeclarativeRecallDataCollectorHNSW data_collector,
                std::string *observations_table,
                idx_t query_idx_for_table,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {

            std::unordered_map<int, int> recall_count_2dec;  // 每个 recall(两位小数) 的计数
            int recall_1_count = 0; // recall==1.00 的计数
            // 两位小数桶，比如：
            // 0.823 -> 82
            // 0.80  -> 80
            // 1.00  -> 100
            auto get_recall_bucket_2dec = [&](double recall) {
                return (int)std::round(recall * 100);
            };
            //是否可以写入日志的规则，即：每个查询点在该准确率最多收集10条数据，准确率为1最多收集20条数据
            auto allow_log = [&](double recall) {
                // 单独处理 1.00（浮点要宽松）
                if (std::fabs(recall - 1.0) < 1e-9) {
                    if (recall_1_count < 20) {
                        recall_1_count++;
                        return true;
                    }
                    return false;
                }

                int bucket = get_recall_bucket_2dec(recall);
                int &cnt = recall_count_2dec[bucket];

                if (cnt < 10) {
                    cnt++;
                    return true;
                }

                return false;
            };

            std::string query_observations_str = "";

            int nres = nres_in;
            int ndis = 0;

            int logging_interval = data_collector.logging_interval;

            bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int nstep = 0;
            int duration=1;
            int visited_points=0;

            int total_insertions = 0;
            float first_nn_dist = -1;
            float prev_recall = 0;

            double search_time_start = data_collector.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    // ndis++; // FAISS does not increase ndis in the first phase of the base layer search.
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (first_nn_dist == -1) {  // if the first NN distance is not yet set.
                            first_nn_dist = d;
                        }
                    }

                    // improvement: Do not call the data_collector if the recall is 0. (TODO)
                    double recall_k = data_collector.data_manager.get_recallk(query_idx);
                    double elapsed = data_collector.data_manager.elapsed_secs() - search_time_start;
                    if (allow_log(recall_k)){
                        query_observations_str += data_collector.get_observation_data_str(
                                query_idx,
                                nstep,
                                ndis,
                                total_insertions,
                                elapsed,
                                first_nn_dist,
                                recall_k,
                                0,
                                duration);
                    }

                }
                vt.set(v1);
            }

            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=data_collector.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }

                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                            float recall = data_collector.data_manager.get_recallk(query_idx);
                            if (allow_log(recall)){
                                query_observations_str +=
                                        data_collector.get_observation_data_str(
                                                query_idx,
                                                nstep,
                                                ndis,
                                                total_insertions,
                                                data_collector.data_manager.elapsed_secs() - search_time_start,
                                                first_nn_dist,
                                                recall,
                                                visited_points,
                                                duration);
                            }
                        }
                    }

                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            add_to_heap(saved_j[id4], dis[id4]);
                        }

                        counter = 0;
                    }
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);

                    ndis++;
                    add_to_heap(saved_j[icnt], dis);
                }

                nstep++;
                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            // We always log the last observation.
            query_observations_str += data_collector.get_observation_data_str(
                    query_idx,
                    nstep,//nstep（pop-min 次数 → 搜索步数）
                    ndis,
                    total_insertions,//（加入 top-k 次数）
                    data_collector.data_manager.elapsed_secs() - search_time_start,//（搜索时间）
                    first_nn_dist,//（第一次命中的距离）
                    data_collector.data_manager.get_recallk(query_idx),
                    visited_points,
                    duration);//（当前 recall@k）

            if (observations_table) {
                observations_table[query_idx_for_table] = query_observations_str;
            }

            return nres;
        }
        int search_from_candidates_RAET_CNS_data_generation(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DeclarativeRecallDataCollectorHNSW data_collector,
                std::string *observations_table,
                idx_t query_idx_for_table,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            std::string query_observations_str = "";

            int nres = nres_in;
            int ndis = 0;

            int logging_interval = data_collector.logging_interval;

            bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;
            //每个数据点的跳数
            int nstep = 0;
            int visited_points=0;
            //当step变化，但是visited_points不变化时，duration+1，表示，准确率在停留在visited_point/k 的时间
            int duration=1;
            int total_insertions = 0;
            float first_nn_dist = -1;
            float prev_recall = 0;

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();


            double search_time_start = data_collector.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    // ndis++; // FAISS does not increase ndis in the first phase of the base layer search.
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d< best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {  // if the first NN distance is not yet set.
                            first_nn_dist = d;
                        }
                    }

                }
                vt.set(v1);
            }

            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=data_collector.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                data_collector.data_manager.record_hop(v0, nstep, query_idx);

                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么收集特征
                if(d0>=threshold){
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    double ratio = -1.0;
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }

                    double recall_k = data_collector.data_manager.get_recallk(query_idx);
                    double elapsed = data_collector.data_manager.elapsed_secs() - search_time_start;

                    query_observations_str += data_collector.get_observation_CNS_data_str(
                            query_idx,
                            nstep,
                            ndis,
                            total_insertions,
                            elapsed,
                            first_nn_dist,
                            recall_k,
                            visited_points,
                            ratio,
                            best_dist_so_far);
                    break;
                }

                if (do_dis_check) {
                    // tricky stopping condition: there are more that ef
                    // distances that are processed already that are smaller
                    // than d0

                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            //跳数+1
                            data_collector.data_manager.record_hop(idx, nstep+1, query_idx);
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                //维护最近邻信息
                                if (0 < dis && dis < best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                                threshold = res.threshold;
                            }

                        }
                    }

                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {

                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }

                            ndis++;
                            add_to_heap(saved_j[id4], dis[id4]);
                        }

                        counter = 0;
                    }
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);

                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }

                    ndis++;
                    add_to_heap(saved_j[icnt], dis);
                }
                //跳数+1
                nstep++;

                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }
            if (observations_table) {
                observations_table[query_idx_for_table] = query_observations_str;
            }
            return nres;
        }
        int search_from_candidates_RAET_Select_data_generation(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETModelSelectPredictorHNSW raet_predictor,
                std::string *observations_table,
                idx_t query_idx_for_table,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            std::string query_observations_str = "";
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            //选择预测recall还是预测CNS的标志，0代表recall，1代表CNS
            int predict_select_recall_or_CNS=0;

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;

            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                raet_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts >= raet_predictor.stability_times){
                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                        flag_predict=true;
                        first_flag=false;
                    }
                }
                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_predictor.data_manager.record_hop(v0, nstep, query_idx);
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_predict && predict_select_recall_or_CNS==0){
                            if (prev_recall >= target || first_model_pre) {

                                double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;

                                query_observations_str += raet_predictor.get_observation_CNS_data_str(
                                        query_idx,
                                        nstep,
                                        ndis,
                                        total_insertions,
                                        elapsed,
                                        first_nn_dist,
                                        recall_k,
                                        visited_points,
                                        ratio,
                                        best_dist_so_far);
                                reached_target_recall = true;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }
            if (observations_table) {
                observations_table[query_idx_for_table] = query_observations_str;
            }
            return nres;
        }

        int search_from_candidates_baseline(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DeclarativeRecallDataCollectorHNSW data_collector,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int nres = nres_in;
            int ndis = 0;

            int total_insertions = 0;
            float first_nn_dist = -1;
            bool first_nn_dist_set = false;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            double search_time_start = data_collector.data_manager.elapsed_secs();

            bool do_naive_early_stop = data_collector.do_naive_early_stop;
            int dist_early_stop_threshold = data_collector.dist_early_stop_threshold;

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;
                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (!first_nn_dist_set) {
                            first_nn_dist = d;
                            first_nn_dist_set = true;
                        }
                    }
                }
                vt.set(v1);
            }

            float prev_recall = 0;
            int nstep = 0;
            bool stoppping_condition_met = false;
            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
                int visited_points=candidates.k-candidates.nvalid;
                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                            if (data_collector.logging_interval ==
                                INTERVAL_WHEN_RECALL_UPDATES_VALUE) {
                                float recall = data_collector.data_manager.get_recallk(query_idx);
                                if (recall > prev_recall) {
                                    data_collector.append_to_log(
                                            query_idx,
                                            nstep,
                                            ndis,
                                            total_insertions,
                                            data_collector.data_manager.elapsed_secs() - search_time_start,
                                            first_nn_dist,
                                            recall,
                                            visited_points);
                                    prev_recall = recall;
                                }
                            }
                        }
                    }
                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;

                            add_to_heap(saved_j[id4], dis[id4]);

                            if (do_naive_early_stop &&
                                ndis >= dist_early_stop_threshold) {
                                stoppping_condition_met = true;
                                break;
                            }
                        }

                        if (stoppping_condition_met) {
                            break;
                        }

                        counter = 0;
                    }
                }

                if (stoppping_condition_met) {
                    break;
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);

                    ndis++;

                    add_to_heap(saved_j[icnt], dis);

                    if (do_naive_early_stop && ndis >= dist_early_stop_threshold) {
                        stoppping_condition_met = true;
                        break;
                    }
                }

                if (stoppping_condition_met) {
                    break;
                }

                nstep++;
                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            data_collector.append_to_log(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    data_collector.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    data_collector.data_manager.get_recallk(query_idx),
                    candidates.k-candidates.nvalid);

            return nres;
        }
        int search_from_candidates_baseline_test(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DeclarativeRecallDataCollectorHNSW data_collector,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            //测试使用
            std::ofstream csv_file("/root/DARTH-main-data/sift-CNS500-搜索刚好达到99准确率时的状态.csv", std::ios::app); // 追加模式
            csv_file << query_idx <<"," ;
            bool flag_stable=false;
            bool flag_stop=false;

            int nres = nres_in;
            int ndis = 0;

            int total_insertions = 0;
            float first_nn_dist = -1;
            bool first_nn_dist_set = false;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            double search_time_start = data_collector.data_manager.elapsed_secs();

            bool do_naive_early_stop = data_collector.do_naive_early_stop;
            int dist_early_stop_threshold = data_collector.dist_early_stop_threshold;

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;
                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (!first_nn_dist_set) {
                            first_nn_dist = d;
                            first_nn_dist_set = true;
                        }
                    }
                }
                vt.set(v1);
            }

            float prev_recall = 0;
            int nstep = 0;
            bool stoppping_condition_met = false;
            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);
                int visited_points=candidates.k-candidates.nvalid;
                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                int current_insert_count=0;

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                            current_insert_count++;
                            if (data_collector.logging_interval ==
                                INTERVAL_WHEN_RECALL_UPDATES_VALUE) {
                                float recall = data_collector.data_manager.get_recallk(query_idx);
                                if (recall > prev_recall) {
                                    data_collector.append_to_log(
                                            query_idx,
                                            nstep,
                                            ndis,
                                            total_insertions,
                                            data_collector.data_manager.elapsed_secs() - search_time_start,
                                            first_nn_dist,
                                            recall,
                                            visited_points);
                                    prev_recall = recall;
                                }
                            }
                            //真实准确率
                            double recall_k = data_collector.data_manager.get_recallk(query_idx);
                            if(recall_k>=0.99) {
                                csv_file << flag_stable <<"," ;
                                flag_stop=true;
                            }
                        }
                    }
                    candidates.push(idx, dis);
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;

                            add_to_heap(saved_j[id4], dis[id4]);

                            if (do_naive_early_stop &&
                                ndis >= dist_early_stop_threshold) {
                                stoppping_condition_met = true;
                                break;
                            }
                        }

                        if (stoppping_condition_met) {
                            break;
                        }

                        counter = 0;
                    }
                }

                if (stoppping_condition_met) {
                    break;
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);

                    ndis++;

                    add_to_heap(saved_j[icnt], dis);

                    if (do_naive_early_stop && ndis >= dist_early_stop_threshold) {
                        stoppping_condition_met = true;
                        break;
                    }
                }

                if (stoppping_condition_met) {
                    break;
                }

                nstep++;
                if (!do_dis_check && nstep > efSearch) {
                    break;
                }
                //处于0.95的稳定状态
                if(current_insert_count<= 1) {
                    flag_stable=true;
                }else{
                    flag_stable=false;
                }

                if(flag_stop) {
                    csv_file << flag_stable <<"\n";
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            data_collector.append_to_log(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    data_collector.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    data_collector.data_manager.get_recallk(query_idx),
                    candidates.k-candidates.nvalid);

            return nres;
        }
        int search_from_candidates_DARTH(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DARTHPredictorHNSW darth_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的距离计算计数（用于判断是否触发预测）
            int ndis_interval = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            float prev_recall = 0;
            int total_insertions = 0;//插入到CNS前K位置次数
            int prediction_interval = darth_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = darth_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);

                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *prediction_interval,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                        }

                        if (ndis_interval % *prediction_interval == 0) {
                            (*query_predictor_calls)++;
                            double elapsed = darth_predictor.data_manager.elapsed_secs() - search_time_start;
                            *predictedRecallOut = darth_predictor.predict_recall(
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    elapsed,
                                    first_nn_dist,
                                    *query_predictor_calls,
                                    prediction_interval,
                                    predictor_time);

                            if (*predictedRecallOut >= darth_predictor.target_recall) {
                                reached_target_recall = true;
                            }

                            ndis_interval = 0;
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }

                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            ndis_interval++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall, &prediction_interval,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    ndis_interval++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall, &prediction_interval,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }

                nstep++;

                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }

                if (early_terminate) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            darth_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    darth_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
                    prediction_interval,
                    query_predictor_calls,
                    predictor_time);

            return nres;
        }
        int search_from_candidates_DARTH_test(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                DARTHPredictorHNSW darth_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {

            std::ofstream csv_file("/root/DARTH-main/experiments/1-k100-efSFull/5-our_RAET_CNS/0-0-CNS结果分析/无用-sift_darth.txt", std::ios::app); // 追加模式
//            if(query_idx==8 || query_idx==51){
//
//            }
            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的距离计算计数（用于判断是否触发预测）
            int ndis_interval = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            float prev_recall = 0;
            int total_insertions = 0;//插入到CNS前K位置次数
            int prediction_interval = darth_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = darth_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);

                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *prediction_interval,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                        }

                        if (ndis_interval % *prediction_interval == 0) {
                            (*query_predictor_calls)++;
                            double elapsed = darth_predictor.data_manager.elapsed_secs() - search_time_start;
                            *predictedRecallOut = darth_predictor.predict_recall(
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    elapsed,
                                    first_nn_dist,
                                    *query_predictor_calls,
                                    prediction_interval,
                                    predictor_time);
                            if(query_idx==8 || query_idx==51) {
                                csv_file << query_idx <<"," << ndis <<"," << *query_predictor_calls <<"," << *predictedRecallOut <<"\n";
                            }
                            if (*predictedRecallOut >= darth_predictor.target_recall) {
                                reached_target_recall = true;
                            }

                            ndis_interval = 0;
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }

                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            ndis_interval++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall, &prediction_interval,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    ndis_interval++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall, &prediction_interval,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }

                nstep++;

                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }

                if (early_terminate) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            darth_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    darth_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
                    prediction_interval,
                    query_predictor_calls,
                    predictor_time);

            return nres;
        }

        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;
            float d0;
            while (candidates.size() > 0) {
                d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }
//                if(stability_counts >= raet_predictor.stability_times && (outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= raet_predictor.target_recall || first_flag)){
//                    flag_predict=true;
//                    first_flag=false;
//                }
                if(stability_counts >= raet_predictor.stability_times){
                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                        flag_predict=true;
                        first_flag=false;
                    }
                }
//                if(stability_counts >= raet_predictor.stability_times){
//                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target|| first_flag || stability_counts==2*raet_predictor.stability_times){
//                        flag_predict=true;
//                        first_flag=false;
//                    }
//                }

//                if(stability_counts >= raet_predictor.stability_times && ndis>=2349 && query_predictor_calls<=7) {
//                    flag_predict=true;
//                }

                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_predict){
                            if (prev_recall >= target || first_model_pre) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration);
                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;

                                //更新外层，记录的是两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //更新外层模型预测值为最新值
                                outer_layer_predict_recall=*predictedRecallOut;
//                                flag_predict=false;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
                    );
            return nres;
        }

        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET_get_selector_label(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;
            float d0;
            while (candidates.size() > 0) {
                d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

//                if(stability_counts >= raet_predictor.stability_times && (outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= raet_predictor.target_recall || first_flag)){
//                    flag_predict=true;
//                    first_flag=false;
//                }
                    if(stability_counts >= raet_predictor.stability_times){
                        if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                            flag_predict=true;
                            first_flag=false;
                        }
                    }
//                if(stability_counts >= raet_predictor.stability_times){
//                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target|| first_flag || stability_counts==2*raet_predictor.stability_times){
//                        flag_predict=true;
//                        first_flag=false;
//                    }
//                }

//                if(stability_counts >= raet_predictor.stability_times && ndis>=2349 && query_predictor_calls<=7) {
//                    flag_predict=true;
//                }

                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_predict){
                            if (prev_recall >= target || first_model_pre) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration);
                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;

                                //更新外层，记录的是两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //更新外层模型预测值为最新值
                                outer_layer_predict_recall=*predictedRecallOut;
//                                flag_predict=false;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }
//          只有搜索到CNS中K号位置之后的数据点才进行收集，才能作为选择器的样本，记录搜索时间
            if(d0 > threshold){
                raet_predictor.log_final_recall_result(
                        query_idx,
                        nstep,
                        ndis,
                        total_insertions,
                        raet_predictor.data_manager.elapsed_secs() - search_time_start,
                        first_nn_dist,
                        predicted_recall,
//                    prediction_interval,
                        query_predictor_calls,
                        predictor_time,
                        visited_points,
                        duration
                );
            }

            return nres;
        }
        int search_from_candidates_RAET_test(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
//            std::ofstream csv_file("/root/DARTH-main/experiments/1-k100-efSFull/4-our_RAET/1-分析/sift_raet_new.txt", std::ios::app); // 追加模式

            int k = extract_k_from_ResultHandler(res);
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            int visited_points=0;
            int duration=1;

            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;
            //稳定状态标志
            bool flag_stable=false;
            //首次预测标志
            bool flag_first_predict=false;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

//                if(stability_counts >= raet_predictor.stability_times && (outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag)){
//                    flag_predict=true;
//                    first_flag=false;
//                }
                if(stability_counts >= raet_predictor.stability_times){
                    flag_first_predict=true;
                }
//                if(flag_stable && (outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target)){
//                    flag_predict=true;
//                }

//                if(stability_counts >= raet_predictor.stability_times && ndis>=2349 && query_predictor_calls<=7) {
//                    flag_predict=true;
//                }


                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_first_predict && (flag_predict || first_flag)){
                            if (prev_recall >= target || first_model_pre) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration);
//                                if(query_idx==8 || query_idx==51) {
//                                    double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
//                                    csv_file << query_idx <<"," << ndis <<","<< stability_counts <<","<<outer_layer_predict_recall<<","<< pre_now_stability_insert
//                                    <<"," <<prev_recall<<","<< current_insert_count <<"," << *query_predictor_calls <<"," << *predictedRecallOut <<"," << recall_k<<","<< 1 <<"\n";
//                                }

//                                double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
//                                csv_file << query_idx <<"," << ndis <<","<< stability_counts <<","<<outer_layer_predict_recall<<","<< pre_now_stability_insert
//                                         <<"," <<prev_recall<<","<< current_insert_count <<"," << *query_predictor_calls <<"," << *predictedRecallOut <<"," << recall_k<<","<< 1 <<"\n";

                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;

                                //更新外层，记录的是两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //更新外层模型预测值为最新值
                                outer_layer_predict_recall=*predictedRecallOut;
//                                flag_predict=false;
                                first_flag=false;
                            }
                        }
//                        else if(query_idx==8 || query_idx==51 || query_idx==23 || query_idx==47 ||query_idx==52 ||query_idx==60 ||query_idx==72 ||query_idx==93 ||query_idx==113 || query_idx==115){
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
//                            csv_file << query_idx <<"," << ndis <<","<< stability_counts <<","<<outer_layer_predict_recall<<","<< pre_now_stability_insert
//                                     <<"," <<prev_recall<<","<< current_insert_count <<"," << *query_predictor_calls <<"," << *predictedRecallOut <<","  << recall_k <<","<< 0 <<"\n";
//                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                    if(current_insert_count <= k * (1 - target)){
                        flag_stable=true;
                    }else {
                        flag_stable=false;
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }
        //基础搜索方法，只有一个稳定次数要求
        int search_from_candidates_RAET_base(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int k = extract_k_from_ResultHandler(res);

            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;


            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            int visited_points=0;
            int duration=1;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }
                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step步数为135\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis < best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
//                        if(query_idx==1&& stability_counts>=48){
//                            printf("111111\n");
//                        }
                        //如果达到稳定状态，
                        if(stability_counts >= raet_predictor.stability_times){
                            if (prev_recall >= target || first_model_pre) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration);
                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
                        }
                    }
                }
//                printf("查询id %d,stability_counts%d\n",query_idx,stability_counts);
//                if (!do_dis_check && nstep > efSearch) {//另一种早停
//                    break;
//                }
                if (early_terminate) {
                    break;
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }
        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET_New_method(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int k = extract_k_from_ResultHandler(res);
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            int visited_points=0;
            int duration=1;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;

            //首次达到要求稳定次数标志
            bool flag_first_achieve_train_stable=false;
            //上次是否稳定标志
            bool flag_last_stable=false;
            //首次模型预测标志
            bool flag_first_model_predict=true;
            //上一次模型预测值
            float prev_recall = 0;
            //两次模型预测之间插入到CNS前K位置数据点数量
            int pre_now_stability_insert = 0;

            //已经达到多少次稳定
            int has_achieve_stable_counts=0;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
                int visited_points=candidates.k-candidates.nvalid;
                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }
                if(stability_counts >= raet_predictor.stability_times){
                    flag_first_achieve_train_stable=true;
                }
//                在sift数据集上依然比不过darth，略差于使用stability_counts
//                if(has_achieve_stable_counts >= 65){
//                    flag_first_achieve_train_stable=true;
//                }

//                if(ndis >= 1174){
//                    flag_first_achieve_train_stable=true;
//                }

//                if(flag_last_stable){
//                    flag_first_achieve_train_stable=true;
//                }

                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        //可以进行模型预测的判断标准：第一次达到稳定或者上一次达到稳定与这次达到稳定之间插入数量/k + 上一次模型预测值 > target_recall
                        //首次达到指定的稳定次数，可以开始进行模型预测
//                        if(flag_first_achieve_train_stable && flag_CNS_noindex0){
                        if(flag_first_achieve_train_stable ){
                            //首次模型预测或者  上次的模型预测值到该次稳定插入前K位置数据点数量是否大于要求准确率
                            if ((flag_last_stable && prev_recall >= target) || flag_first_model_predict) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration);
                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //重置两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //将第一次模型预测置为false
                                flag_first_model_predict=false;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);

//                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
                        }
                    }
                    if(current_insert_count <= k * (1 - target)){
                        flag_last_stable=true;
                    }else {
                        flag_last_stable=false;
                    }
//                }
                if(current_insert_count <= k * (1 - target)){
                    has_achieve_stable_counts++;
                }


                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }
        int search_from_candidates_RAET_CNS(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETCNSPredictorHNSW raet_CNS_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int k = extract_k_from_ResultHandler(res);

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_CNS_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d< best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;


            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
                int visited_points = candidates.k-candidates.nvalid;
                raet_CNS_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么进行模型预测
                if(d0 > threshold && model_predict){
//                    double predictor_start_time = raet_CNS_predictor.data_manager.elapsed_secs();
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }
                    //回归模型
                    predict_CNS = raet_CNS_predictor.predict_CNS_test_Noquery_Noske_Nodensty
                            (
                            query_idx,
                            nstep,
                            ndis,
                            total_insertions,
                            raet_CNS_predictor.data_manager.elapsed_secs() - search_time_start,
                            first_nn_dist,
                            k,
                            &predictor_time,
                            efSearch,
                            ratio,
                            best_dist_so_far);
                    candidates.n=predict_CNS;
                    while (candidates.k > predict_CNS) {
                        if (candidates.ids[0] != -1)
                            candidates.nvalid--;

                        faiss::heap_pop<MinimaxHeap::HC>(
                                candidates.k--,
                                candidates.dis.data(),
                                candidates.ids.data()
                        );
                    }
//                    predictor_time = raet_CNS_predictor.data_manager.elapsed_secs() - predictor_start_time;
                    model_predict=false;
                }

                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,

                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_CNS_predictor.data_manager.record_hop(idx, nstep+1, query_idx);
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                        }

                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }

                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];
                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4],
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;

                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }

                    early_terminate = add_to_heap(saved_j[icnt], dis,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }


            raet_CNS_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_CNS_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    query_predictor_calls,
                    predictor_time,
                    best_dist_so_far,
                    ratio,
                    predict_CNS
            );
            return nres;
        }
        int search_from_candidates_RAET_CNS_Be_Used_By_Selector(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                RAETCNSPredictorHNSW raet_CNS_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            int k = extract_k_from_ResultHandler(res);

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_CNS_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d< best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;


            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
                int visited_points = candidates.k-candidates.nvalid;
                raet_CNS_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么进行模型预测
                if(d0 > threshold && model_predict){
//                    double predictor_start_time = raet_CNS_predictor.data_manager.elapsed_secs();
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }
                    //回归模型
                    predict_CNS = raet_CNS_predictor.predict_CNS_test_Noquery_Noske_Nodensty
                            (
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    raet_CNS_predictor.data_manager.elapsed_secs() - search_time_start,
                                    first_nn_dist,
                                    k,
                                    &predictor_time,
                                    efSearch,
                                    ratio,
                                    best_dist_so_far);
                    candidates.n=predict_CNS;
                    while (candidates.k > predict_CNS) {
                        if (candidates.ids[0] != -1)
                            candidates.nvalid--;

                        faiss::heap_pop<MinimaxHeap::HC>(
                                candidates.k--,
                                candidates.dis.data(),
                                candidates.ids.data()
                        );
                    }
//                    predictor_time = raet_CNS_predictor.data_manager.elapsed_secs() - predictor_start_time;
                    model_predict=false;
                }

                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,

                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_CNS_predictor.data_manager.record_hop(idx, nstep+1, query_idx);
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                        }

                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }

                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];
                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4],
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;

                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }

                    early_terminate = add_to_heap(saved_j[icnt], dis,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }


            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_CNS_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    0,
                    query_predictor_calls,
                    predictor_time,
                    0,
                    0
            );
            return nres;
        }
        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET_Select_CNS_Pred(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETModelSelectPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            //选择预测recall还是预测CNS的标志，0代表recall，1代表CNS
            int predict_select_recall_or_CNS=0;

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;

            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                raet_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts >= raet_predictor.stability_times){
                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                        flag_predict=true;
                        first_flag=false;
                    }
                }
                //todo 进行预测CNS的预测
                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么进行模型预测
                if(d0 > threshold && model_predict){
//                    double predictor_start_time = raet_CNS_predictor.data_manager.elapsed_secs();
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }

                    //回归模型
//                    predict_CNS = raet_predictor.predict_CNS_test_Noquery_Noske_Nodensty
//                    使用二分类模型预测进行方法选择
//                    predict_CNS = raet_predictor.predict_CNS_Model_Select_test_Noquery_Noske_Nodensty
//                    使用阈值进行方法选择
                    predict_CNS = raet_predictor.predict_CNS_Select_test_Noquery_Noske_Nodensty
                            (
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                                    first_nn_dist,
                                    k,
                                    &predictor_time,
                                    efSearch,
                                    ratio,
                                    best_dist_so_far);
                    //todo 如果*predictedRecallOut的值小于0，那么进行预测CNS的预测，否则进行预测准确率的预测
                    if(predict_CNS<0){
                        //使用预测准确率方法
                        predict_select_recall_or_CNS=0;
                        query_predictor_calls+=1;
                    }else{
                        predict_select_recall_or_CNS=1;
                        query_predictor_calls+=2;
                        candidates.n=predict_CNS;
                        while (candidates.k > predict_CNS) {
                            if (candidates.ids[0] != -1)
                                candidates.nvalid--;

                            faiss::heap_pop<MinimaxHeap::HC>(
                                    candidates.k--,
                                    candidates.dis.data(),
                                    candidates.ids.data()
                            );
                        }
                    }
                    model_predict=false;
                }

                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_predictor.data_manager.record_hop(v0, nstep, query_idx);
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_predict && predict_select_recall_or_CNS==0){
                            if (prev_recall >= target || first_model_pre) {
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量
                                (*query_predictor_calls)++;
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall(
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration,
                                        best_dist_so_far);

                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;

                                //更新外层，记录的是两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //更新外层模型预测值为最新值
                                outer_layer_predict_recall=*predictedRecallOut;
//                                flag_predict=false;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }
            //将nstep作为，使用了召回率预测（0），还是CNS规模预测（1）。
            nstep=predict_select_recall_or_CNS;
            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }

        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET_Select_Recall_Pred(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETModelSelectPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            //选择预测recall还是预测CNS的标志，0代表recall，1代表CNS
            int predict_select_recall_or_CNS=0;

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;

            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                raet_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts >= raet_predictor.stability_times){
                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                        flag_predict=true;
                        first_flag=false;
                    }
                }
                //todo 进行预测CNS的预测
                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么进行模型预测
                if(d0 > threshold && model_predict && predict_select_recall_or_CNS==1){
//                    double predictor_start_time = raet_CNS_predictor.data_manager.elapsed_secs();
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }

                    //回归模型
                    predict_CNS = raet_predictor.predict_CNS_test_Noquery_Noske_Nodensty
//                    predict_CNS = raet_predictor.predict_CNS_Model_Select_test_Noquery_Noske_Nodensty
                            (
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                                    first_nn_dist,
                                    k,
                                    &predictor_time,
                                    efSearch,
                                    ratio,
                                    best_dist_so_far);
                    if(predict_CNS<0){
                        //使用预测准确率方法
                        predict_select_recall_or_CNS=0;
                        query_predictor_calls+=1;
                    }else{
                        query_predictor_calls+=2;
                        candidates.n=predict_CNS;
                        while (candidates.k > predict_CNS) {
                            if (candidates.ids[0] != -1)
                                candidates.nvalid--;

                            faiss::heap_pop<MinimaxHeap::HC>(
                                    candidates.k--,
                                    candidates.dis.data(),
                                    candidates.ids.data()
                            );
                        }
                        model_predict=false;
                    }
                }

                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_predictor.data_manager.record_hop(v0, nstep, query_idx);
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(flag_predict && predict_select_recall_or_CNS==0){
                            if (prev_recall >= target || first_model_pre) {
                                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                                if (entry_neighbors_cnt > 0) {
                                    double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                                    ratio = entry_dist / mean_nbr;
                                }
                                //真实准确率
//                            double recall_k = raet_predictor.data_manager.get_recallk(query_idx);
                                //获取特征：CNS中已访问的数据点数量

                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
//                                *predictedRecallOut = raet_predictor.predict_recall
//                                (
//                                        query_idx,
//                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
//                                        ndis,
//                                        total_insertions,//插入到CNS前k位置数据点数量
//                                        elapsed,
//                                        first_nn_dist,//底层入口点到查询点距离
//                                        *query_predictor_calls,//模型预测次数
//                                        predictor_time,
//                                        visited_points,
//                                        duration,
//                                        best_dist_so_far);
                                *predictedRecallOut = raet_predictor.Model_Select_predict_recall
                                (
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration,
                                        best_dist_so_far,
                                        ratio);
                                (*query_predictor_calls)++;
                                if(*predictedRecallOut<0){
                                    predict_select_recall_or_CNS=1;
                                    (*query_predictor_calls)++;
                                }else{
                                    if (*predictedRecallOut >= target) {
                                        reached_target_recall = true;
                                    }
                                    //更新内层模型预测值为最新值
                                    prev_recall=*predictedRecallOut;
                                    //内层预测为false
                                    first_model_pre=false;

                                    //更新外层，记录的是两次稳定之间的插入量
                                    pre_now_stability_insert=0;
                                    //更新外层模型预测值为最新值
                                    outer_layer_predict_recall=*predictedRecallOut;
//                                flag_predict=false;
                                }

                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }


            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }

        //两次稳定之间统计，插入到前k位置的数量
        int search_from_candidates_RAET_Query_Model_Selector(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                RAETModelSelectPredictorHNSW raet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {
            //二分类模型选择器
            int num_rows = 1;
            int num_feats = raet_predictor.data_manager.d;
            std::vector<double> feats(static_cast<size_t>(num_feats));
            for (int i = 0; i < raet_predictor.data_manager.d; i++) {
                feats[static_cast<size_t>(i)] =raet_predictor.data_manager.queries[query_idx * raet_predictor.data_manager.d + i];
            }
            double out_result_selector[1];
            long int out_len_selector;
            int res_selector = LGBM_BoosterPredictForMatSingleRow(
                    raet_predictor.booster_selector,
                    feats.data(),
                    C_API_DTYPE_FLOAT64,
                    num_feats,
                    1,
                    C_API_PREDICT_NORMAL,
                    0,
                    -1,
                    "",
                    &out_len_selector,
                    out_result_selector);
            if (res_selector != 0) {
                printf("Error predicting recall\n");
                exit(1);
            }
            // probability of class=1
            double prob = out_result_selector[0];
            // threshold = 0.5 (you can tune it)
            //选择预测recall还是预测CNS的标志，1代表recall，0代表CNS
            int predict_select_recall_or_CNS = (prob >= 0.5) ? 1 : 0;

            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;
            int k = extract_k_from_ResultHandler(res);

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();

            int nres = nres_in;
            int ndis = 0;
            int visited_points=0;
            int duration=1;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
            bool entry_point_processed = true;
            float entry_dist = 0.0;
            double entry_neighbors_sum = 0.0;
            int entry_neighbors_cnt = 0;

            double search_time_start = raet_predictor.data_manager.elapsed_secs();

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d < best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            //外层预测，即上次稳定性被破坏时的模型预测准确率,是否可以进行模型预测标志
            float outer_layer_predict_recall=0;
            int pre_now_stability_insert = 0;
            //是否可以进行模型预测标志
            bool flag_predict=false;
            bool first_flag=true;

            //模型预测只预测一次
            bool model_predict = true;
            double ratio = -1.0;
            int predict_CNS = 0;

            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离
//                int visited_points=candidates.k-candidates.nvalid;
                int visited_points_now=raet_predictor.data_manager.get_visited_points_k_query(query_idx,vt);
                if(visited_points==visited_points_now) {
                    duration++;
                }else{
                    duration=1;
                    visited_points=visited_points_now;
                }
                raet_predictor.data_manager.record_hop(v0, nstep, query_idx);

                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                if (entry_point_processed) {
                    entry_dist = d0;
                }

                if(best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }

                if(stability_counts >= raet_predictor.stability_times){
                    if(outer_layer_predict_recall + (1.0/k)*pre_now_stability_insert >= target || first_flag || stability_counts==raet_predictor.stability_times_r1){
                        flag_predict=true;
                        first_flag=false;
                    }
                }
                //todo 进行预测CNS的预测
                //如果当前访问的数据点到查询点距离大于 CNS中k-1号位置到查询点距离，那么进行模型预测
                if(predict_select_recall_or_CNS==0 && d0 > threshold && model_predict){
//                    double predictor_start_time = raet_CNS_predictor.data_manager.elapsed_secs();
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_neighbors_cnt > 0) {
                        double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                        ratio = entry_dist / mean_nbr;
                    }

                    //回归模型
                    predict_CNS = raet_predictor.predict_CNS_test_Noquery_Noske_Nodensty
                            (
                                    query_idx,
                                    nstep,
                                    ndis,
                                    total_insertions,
                                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                                    first_nn_dist,
                                    k,
                                    &predictor_time,
                                    efSearch,
                                    ratio,
                                    best_dist_so_far);
                    query_predictor_calls+=1;
                    candidates.n=predict_CNS;
                    while (candidates.k > predict_CNS) {
                        if (candidates.ids[0] != -1)
                            candidates.nvalid--;
                        faiss::heap_pop<MinimaxHeap::HC>(
                                candidates.k--,
                                candidates.dis.data(),
                                candidates.ids.data()
                        );
                    }
                    model_predict=false;
                }

                //当前搜索的数据点邻居范围内 上一次模型预测值
                float prev_recall = 0;
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;
//                if(nstep==135){
//                    printf("step为135:\n");
//                }

                if (do_dis_check) {
                    //统计candidate中，距离小于d0的元素个数
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            raet_predictor.data_manager.record_hop(v0, nstep, query_idx);
                            total_insertions++;
                            current_insert_count++;
                            pre_now_stability_insert++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis< best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                            prev_recall+=1.0/k;
                        }
                        //如果达到稳定状态，且上一次稳定状态 到 这次稳定状态 插入前K位置数据点数量  * 1/k + outer_layer_predict_recall 与target_recall比较
                        if(predict_select_recall_or_CNS==1 && flag_predict){
                            if (prev_recall >= target || first_model_pre) {
                                //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                                if (entry_neighbors_cnt > 0) {
                                    double mean_nbr = entry_neighbors_sum / entry_neighbors_cnt;
                                    ratio = entry_dist / mean_nbr;
                                }
                                double elapsed = raet_predictor.data_manager.elapsed_secs() - search_time_start;
                                *predictedRecallOut = raet_predictor.predict_recall
                                (
                                        query_idx,
                                        nstep,//是搜索跳数，搜索到当前为止，所经过跳数
                                        ndis,
                                        total_insertions,//插入到CNS前k位置数据点数量
                                        elapsed,
                                        first_nn_dist,//底层入口点到查询点距离
                                        *query_predictor_calls,//模型预测次数
                                        predictor_time,
                                        visited_points,
                                        duration,
                                        best_dist_so_far);
                                (*query_predictor_calls)++;
                                if (*predictedRecallOut >= target) {
                                    reached_target_recall = true;
                                }
                                //更新内层模型预测值为最新值
                                prev_recall=*predictedRecallOut;
                                //内层预测为false
                                first_model_pre=false;
                                //更新外层，记录的是两次稳定之间的插入量
                                pre_now_stability_insert=0;
                                //更新外层模型预测值为最新值
                                outer_layer_predict_recall=*predictedRecallOut;
                            }
                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }
                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                            if (entry_point_processed) {
                                entry_neighbors_sum += dis[id4];
                                entry_neighbors_cnt++;
                            }
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }
                    if (early_terminate) {
                        break;
                    }
                }
                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    //入口点到查询点距离 与 入口点邻居距离均值之间的比值特征
                    if (entry_point_processed) {
                        entry_neighbors_sum += dis;
                        entry_neighbors_cnt++;
                    }
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
//                bool is_not_nearest = (best_id_so_far != -1 && v0 != best_id_so_far);
                if(flag_CNS_noindex0) {
                    //因为在稳定中，且当前数据点的邻居中没有邻居数据点被加入到前K位置，稳定状态+1
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        //使用插入前K位置数据点数量来作为稳定性判断，插入数量少于K*(1-acc_thold)，即相似数量为 K*acc_thold ，那么认为一次稳定
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            //稳定状态被破坏，重置为0
                            stability_counts = 0;
//                            //更新外层，记录的是两次稳定之间的插入量
//                            pre_now_stability_insert=0;
//                            //更新外层模型预测值为最新值
//                            outer_layer_predict_recall=prev_recall;
                        }
                    }
                }
                if (!do_dis_check && nstep > efSearch) {//另一种早停
                    break;
                }
                if (early_terminate) {
                    break;
                }
                //重置为false
                flag_predict=false;
                //入口点标志，设置为false，表示，之后的数据点不是入口点
                entry_point_processed=false;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }


            raet_predictor.log_final_recall_result(
                    query_idx,
                    nstep,
                    ndis,
                    total_insertions,
                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    predicted_recall,
//                    prediction_interval,
                    query_predictor_calls,
                    predictor_time,
                    visited_points,
                    duration
            );
            return nres;
        }

        int search_from_candidates_RAET_GET_STABILITY_TIMES(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats_RAET &stats,
                int level,
                idx_t query_idx,
                RAETPredictorHNSW raet_predictor,
                int nres_in = 0,
                const std::string& stability_log_path="",
                const SearchParametersHNSW *params = nullptr) {
            //将稳定次数写入文件，追加方式，重新收集，记得删除
            std::ofstream stability_log;
            if (!stability_log_path.empty()){
                stability_log.open(stability_log_path, std::ios::app);
            }

            int k = extract_k_from_ResultHandler(res);
            double target = raet_predictor.target_recall;

            target = std::round(target * 100.0) / 100.0;

            // 维护一个搜索过的数据点中距离查询点最近的数据点，即res（CNS）中距离查询点最近的数据点,方便判断，访问CNS中非0位置数据点时进行模型预测
            idx_t best_id_so_far = -1;
            float best_dist_so_far = std::numeric_limits<float>::max();
            int visited_points=0;
            int duration=1;
            int nres = nres_in;
            int ndis = 0;
            //自上一次预测以来的稳定性次数（用于判断是否触发预测）
            int stability_counts = 0;
            //已经达到多少次稳定
            int has_achieve_stable_counts=0;

            bool do_dis_check = params ? params->check_relative_distance
                                       : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            int total_insertions = 0;//插入到CNS前K位置次数
//            int prediction_interval = raet_predictor.initial_prediction_interval;//两次调用 predictor 间隔（会被 predict_recall 通过指针修改）
            int query_predictor_calls = 0;//当前 query 已调用 predictor 的次数。
            double predictor_time = 0;//累积 predictor 花费的时间（秒）。

            float first_nn_dist = -1;//入口点到查询点距离

            double search_time_start = raet_predictor.data_manager.elapsed_secs();


            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                            //维护最近邻信息
                            if (0< d && d< best_dist_so_far) {
                                best_dist_so_far = d;
                                best_id_so_far = v1;
                            }
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            float predicted_recall = 0.0;
            //判断当前访问数据点是否是CNS中0号位置数据点
            bool flag_CNS_noindex0=false;
            while (candidates.size() > 0) {
                float d0 = 0;//当前candidate中距离查询点最近数据点到查询点距离
                int v0 = candidates.pop_min(&d0);//d0是当前candidate中距离查询点最近数据点到查询点距离

                //判断当前访问数据点是否是CNS中0号位置数据点
                if (best_id_so_far != -1 && v0 != best_id_so_far){
                    flag_CNS_noindex0=true;
                }
                //当前访问的数据点，第一次模型调用,给查询点，在稳定后一次机会
                bool first_model_pre=true;
                //当前数据点，有多少邻居加入到CNS前k位置
                int current_insert_count=0;

                if (do_dis_check) {
                    int n_dis_below = candidates.count_below(d0);
                    if (n_dis_below >= efSearch) {
                        break;
                    }
                }

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                // the following version processes 4 neighbors at a time 按 4 批量距离计算
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;
                    //预读（prefetch_L2）+ jmax：为了提高缓存与分支局部性，把要处理的 neighbor count 先求出（jmax = begin + num_neighbors），然后按 4 的块来计算距离。
                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                // ndis += jmax - begin;

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx,
                                       const float dis,
                                       float *predictedRecallOut,
                                       int *query_predictor_calls,
                                       double *predictor_time) {
                    bool reached_target_recall = false;
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            current_insert_count++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                                //维护最近邻信息
                                if (0< dis && dis < best_dist_so_far) {
                                    best_dist_so_far = dis;
                                    best_id_so_far = idx;
                                }
                            }
                        }
//                        if (flag_CNS_noindex0){
//                            double recall = raet_predictor.data_manager.get_recallk(query_idx);
//                            if(recall>=target) {
//                                reached_target_recall=true;
//                            }
//                        }
                    }
                    candidates.push(idx, dis);

                    if (reached_target_recall) {
                        return true;
                    }

                    return false;
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            early_terminate = add_to_heap(
                                    saved_j[id4], dis[id4], &predicted_recall,
                                    &query_predictor_calls, &predictor_time);
                            if (early_terminate) {
                                break;
                            }
                        }
                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }
                //剩余不足 4 的 neighbors 在循环后以单个 qdis(saved_j[icnt]) 计算处理。
                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    early_terminate = add_to_heap(saved_j[icnt], dis, &predicted_recall,
                                                  &query_predictor_calls, &predictor_time);
                    if (early_terminate) {
                        break;
                    }
                }
                nstep++;
                //如果当前访问的数据点不是CNS中0号位置数据点，那么开始进行稳定次数判定
                if(flag_CNS_noindex0){
                    if (stability_counts > 0 && current_insert_count == 0)
                    {
                        stability_counts++;
                    }
                    else
                    {
                        if (current_insert_count <= k * (1 - target))
                        {
                            stability_counts++;
                        }
                        else
                        {
                            stability_counts = 0;
                        }
                    }
                    if(current_insert_count <= k * (1 - target)){
                        has_achieve_stable_counts++;
                    }
                }

                if(!stability_log_path.empty()){
                    // === 写入日志（追加）===
                    double recall = raet_predictor.data_manager.get_recallk(query_idx);
                    if (stability_log.is_open()) {
                        stability_log
                                << query_idx << ","    // 当前 query ID
                                << nstep << ","        // 当前 step（或你想写的指标）
                                << stability_counts<< ","
                                << recall<< std::endl;
                    }
                }
//                if (!do_dis_check && nstep > efSearch) {//另一种早停
//                    break;
//                }
                if (early_terminate) {
                    break;
                }
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
                stats.stability_counts+=stability_counts;
                stats.has_achieve_stable_counts+=has_achieve_stable_counts;;
            }

//            raet_predictor.log_final_recall_result(
//                    query_idx,
//                    nstep,
//                    ndis,
//                    total_insertions,
//                    raet_predictor.data_manager.elapsed_secs() - search_time_start,
//                    first_nn_dist,
//                    predicted_recall,
////                    prediction_interval,
//                    query_predictor_calls,
//                    predictor_time,
//                    visited_points,
//                    duration);
            return nres;
        }


        int search_from_candidates_LAET(
                const HNSW &hnsw,
                DistanceComputer &qdis,
                ResultHandler<C> &res,
                MinimaxHeap &candidates,
                VisitedTable &vt,
                HNSWStats &stats,
                int level,
                idx_t query_idx,
                LAETPredictorHNSW laet_predictor,
                int nres_in = 0,
                const SearchParametersHNSW *params = nullptr) {


            int nres = nres_in;
            int ndis = 0;
            int ndis_interval = 0;

            bool do_dis_check = params ? params->check_relative_distance : hnsw.check_relative_distance;
            int efSearch = params ? params->efSearch : hnsw.efSearch;
            const IDSelector *sel = params ? params->sel : nullptr;

            float prev_recall = 0;
            int total_insertions = 0;

            float first_nn_dist = -1;

            double search_time_start = laet_predictor.data_manager.elapsed_secs();

            double predictor_time = 0;

            C::T threshold = res.threshold;
            for (int i = 0; i < candidates.size(); i++) {
                idx_t v1 = candidates.ids[i];
                float d = candidates.dis[i];
                FAISS_ASSERT(v1 >= 0);
                if (!sel || sel->is_member(v1)) {
                    if (d < threshold) {
                        total_insertions++;

                        if (res.add_result(d, v1)) {
                            threshold = res.threshold;
                        }

                        if (first_nn_dist == -1) {
                            first_nn_dist = d;
                        }
                    }
                }
                vt.set(v1);
            }

            int nstep = 0;
            bool early_terminate = false;
            int ndis_target = std::numeric_limits<int>::max();
            while (candidates.size() > 0) {
                float d0 = 0;
                int v0 = candidates.pop_min(&d0);

                size_t begin, end;
                hnsw.neighbor_range(v0, level, &begin, &end);

                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt.visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                threshold = res.threshold;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (!sel || sel->is_member(idx)) {
                        if (dis < threshold) {
                            total_insertions++;
                            if (res.add_result(dis, idx)) {
                                threshold = res.threshold;
                            }
                        }
                    }

                    candidates.push(idx, dis);

                    // LAET CMU term: This should be called **only** once per query
                    if (ndis == laet_predictor.fixed_amount_of_distance_calcs) {
                        ndis_target = laet_predictor.predict_distance_calcs(
                                query_idx,
                                nstep,
                                ndis,
                                laet_predictor.data_manager.elapsed_secs() - search_time_start,
                                first_nn_dist,
                                &predictor_time);
                    }

                    if (ndis >= ndis_target) {
                        early_terminate = true;
                    }
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt.get(v1);
                    vt.set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            ndis++;
                            add_to_heap(saved_j[id4], dis[id4]);

                            if (early_terminate) {
                                break;
                            }
                        }

                        counter = 0;
                    }

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    ndis++;
                    add_to_heap(saved_j[icnt], dis);

                    if (early_terminate) {
                        break;
                    }
                }

                if (early_terminate) {
                    break;
                }

                nstep++;
            }

            if (level == 0) {
                stats.n1++;
                if (candidates.size() == 0) {
                    stats.n2++;
                }
                stats.ndis += ndis;
            }

            laet_predictor.log_final_result(
                    query_idx,
                    nstep,
                    ndis,
                    laet_predictor.data_manager.elapsed_secs() - search_time_start,
                    first_nn_dist,
                    ndis_target,
                    predictor_time);

            return nres;
        }

        std::priority_queue<HNSW::Node> search_from_candidate_unbounded(
                const HNSW &hnsw,
                const Node &node,
                DistanceComputer &qdis,
                int ef,
                VisitedTable *vt,
                HNSWStats &stats) {
            int ndis = 0;
            std::priority_queue<Node> top_candidates;
            std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

            top_candidates.push(node);
            candidates.push(node);

            vt->set(node.second);

            while (!candidates.empty()) {
                float d0;
                storage_idx_t v0;
                std::tie(d0, v0) = candidates.top();

                if (d0 > top_candidates.top().first) {
                    break;
                }

                candidates.pop();

                size_t begin, end;
                hnsw.neighbor_range(v0, 0, &begin, &end);

                // the following version processes 4 neighbors at a time
                size_t jmax = begin;
                for (size_t j = begin; j < end; j++) {
                    int v1 = hnsw.neighbors[j];
                    if (v1 < 0)
                        break;

                    prefetch_L2(vt->visited.data() + v1);
                    jmax += 1;
                }

                int counter = 0;
                size_t saved_j[4];

                ndis += jmax - begin;

                auto add_to_heap = [&](const size_t idx, const float dis) {
                    if (top_candidates.top().first > dis ||
                        top_candidates.size() < ef) {
                        candidates.emplace(dis, idx);
                        top_candidates.emplace(dis, idx);

                        if (top_candidates.size() > ef) {
                            top_candidates.pop();
                        }
                    }
                };

                for (size_t j = begin; j < jmax; j++) {
                    int v1 = hnsw.neighbors[j];

                    bool vget = vt->get(v1);
                    vt->set(v1);
                    saved_j[counter] = v1;
                    counter += vget ? 0 : 1;

                    if (counter == 4) {
                        float dis[4];
                        qdis.distances_batch_4(
                                saved_j[0],
                                saved_j[1],
                                saved_j[2],
                                saved_j[3],
                                dis[0],
                                dis[1],
                                dis[2],
                                dis[3]);

                        for (size_t id4 = 0; id4 < 4; id4++) {
                            add_to_heap(saved_j[id4], dis[id4]);
                        }

                        counter = 0;
                    }
                }

                for (size_t icnt = 0; icnt < counter; icnt++) {
                    float dis = qdis(saved_j[icnt]);
                    add_to_heap(saved_j[icnt], dis);
                }
            }

            ++stats.n1;
            if (candidates.size() == 0) {
                ++stats.n2;
            }
            stats.ndis += ndis;

            return top_candidates;
        }

    }  // anonymous namespace

    HNSWStats HNSW::search(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                search_from_candidates(
                        *this, qdis, res, candidates, vt, stats, 0, 0, params);
            } else {
                std::priority_queue<Node> top_candidates =
                        search_from_candidate_unbounded(
                                *this,
                                Node(d_nearest, nearest),
                                qdis,
                                ef,
                                &vt,
                                stats);

                while (top_candidates.size() > k) {
                    top_candidates.pop();
                }

                while (!top_candidates.empty()) {
                    float d;
                    storage_idx_t label;
                    std::tie(d, label) = top_candidates.top();
                    res.add_result(d, label);
                    top_candidates.pop();
                }
            }

            vt.advance();

        } else {
            int candidates_size = upper_beam;
            MinimaxHeap candidates(candidates_size);

            std::vector<idx_t> I_to_next(candidates_size);
            std::vector<float> D_to_next(candidates_size);

            HeapBlockResultHandler<C> block_resh(
                    1, D_to_next.data(), I_to_next.data(), candidates_size);
            HeapBlockResultHandler<C>::SingleResultHandler resh(block_resh);

            int nres = 1;
            I_to_next[0] = entry_point;
            D_to_next[0] = qdis(entry_point);

            for (int level = max_level; level >= 0; level--) {
                // copy I, D -> candidates

                candidates.clear();

                for (int i = 0; i < nres; i++) {
                    candidates.push(I_to_next[i], D_to_next[i]);
                }

                if (level == 0) {
                    nres = search_from_candidates(
                            *this, qdis, res, candidates, vt, stats, 0);
                } else {
                    resh.begin(0);
                    nres = search_from_candidates(
                            *this, qdis, resh, candidates, vt, stats, level);
                    resh.end();
                }
                vt.advance();
            }
        }

        return stats;
    }

    HNSWStats HNSW::search_declarative_recall_data_generation(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            DeclarativeRecallDataCollectorHNSW data_collector,
            std::string *observations_table,
            idx_t query_idx_for_table,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
                search_from_candidates_declarative_recall_data_generation_ourTrainData
//                search_from_candidates_declarative_recall_data_generation
                (
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        data_collector,
                        observations_table,
                        query_idx_for_table,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }
        return stats;
    }
    HNSWStats HNSW::search_RAET_CNS_data_generation(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            DeclarativeRecallDataCollectorHNSW data_collector,
            std::string *observations_table,
            idx_t query_idx_for_table,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                search_from_candidates_RAET_CNS_data_generation(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        data_collector,
                        observations_table,
                        query_idx_for_table,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    HNSWStats HNSW::search_RAET_Select_data_generation(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETModelSelectPredictorHNSW data_collector,
            std::string *observations_table,
            idx_t query_idx_for_table,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                search_from_candidates_RAET_Select_data_generation(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        data_collector,
                        observations_table,
                        query_idx_for_table,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    HNSWStats HNSW::search_baseline(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            DeclarativeRecallDataCollectorHNSW data_collector,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
                search_from_candidates_baseline(
//                search_from_candidates_baseline_test(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        data_collector,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    HNSWStats HNSW::search_DARTH(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            DARTHPredictorHNSW recall_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
                search_from_candidates_DARTH(
//                search_from_candidates_DARTH_test(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    HNSWStats HNSW::search_RAET(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETPredictorHNSW recall_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
//                search_from_candidates_RAET_base(
//                常用算法
//                search_from_candidates_RAET(
//                获取选择器样本和标签算法
                  search_from_candidates_RAET_get_selector_label(
//                  search_from_candidates_RAET_New_method(
//                  search_from_candidates_RAET_test(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }
    HNSWStats HNSW::search_RAET_CNS(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETCNSPredictorHNSW recall_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
                search_from_candidates_RAET_CNS(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }
    HNSWStats HNSW::search_RAET_CNS_Be_Used_By_Selector(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETPredictorHNSW recall_Predictor,
            RAETCNSPredictorHNSW recall_CNS_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
                search_from_candidates_RAET_CNS_Be_Used_By_Selector(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_Predictor,
                        recall_CNS_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }
    HNSWStats HNSW::search_RAET_Select(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETModelSelectPredictorHNSW recall_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);
//                search_from_candidates_RAET_Query_Model_Selector
//                search_from_candidates_RAET_Select_Recall_Pred
                search_from_candidates_RAET_Select_CNS_Pred
                (
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }


    HNSWStats_RAET HNSW::search_RAET_GET_STABILITY_TIMES(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            RAETPredictorHNSW recall_predictor,
            const std::string& stability_log_path,
            const SearchParametersHNSW *params) const {
        HNSWStats_RAET stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                search_from_candidates_RAET_GET_STABILITY_TIMES(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        recall_predictor,
                        0,
                        stability_log_path,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    HNSWStats HNSW::search_LAET(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            VisitedTable &vt,
            idx_t query_idx,
            LAETPredictorHNSW laet_predictor,
            const SearchParametersHNSW *params) const {
        HNSWStats stats;
        if (entry_point == -1) {
            return stats;
        }
        int k = extract_k_from_ResultHandler(res);

        if (upper_beam == 1) {
            //  greedy search on upper levels
            storage_idx_t nearest = entry_point;
            float d_nearest = qdis(nearest);

            for (int level = max_level; level >= 1; level--) {
                greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
            }

            int ef = std::max(params ? params->efSearch : efSearch, k);
            if (search_bounded_queue) {  // this is the most common branch
                MinimaxHeap candidates(ef);

                candidates.push(nearest, d_nearest);

                search_from_candidates_LAET(
                        *this,
                        qdis,
                        res,
                        candidates,
                        vt,
                        stats,
                        0,
                        query_idx,
                        laet_predictor,
                        0,
                        params);
            } else {
                assert(false || !"Declarative recall is implemented with default HNSW.");
            }

            vt.advance();

        } else {
            assert(false || !"Declarative recall is implemented with default HNSW.");
        }

        return stats;
    }

    void HNSW::search_level_0(
            DistanceComputer &qdis,
            ResultHandler<C> &res,
            idx_t nprobe,
            const storage_idx_t *nearest_i,
            const float *nearest_d,
            int search_type,
            HNSWStats &search_stats,
            VisitedTable &vt) const {
        const HNSW &hnsw = *this;
        int k = extract_k_from_ResultHandler(res);
        if (search_type == 1) {
            int nres = 0;

            for (int j = 0; j < nprobe; j++) {
                storage_idx_t cj = nearest_i[j];

                if (cj < 0)
                    break;

                if (vt.get(cj))
                    continue;

                int candidates_size = std::max(hnsw.efSearch, k);
                MinimaxHeap candidates(candidates_size);

                candidates.push(cj, nearest_d[j]);

                nres = search_from_candidates(
                        hnsw, qdis, res, candidates, vt, search_stats, 0, nres);
            }
        } else if (search_type == 2) {
            int candidates_size = std::max(hnsw.efSearch, int(k));
            candidates_size = std::max(candidates_size, int(nprobe));

            MinimaxHeap candidates(candidates_size);
            for (int j = 0; j < nprobe; j++) {
                storage_idx_t cj = nearest_i[j];

                if (cj < 0)
                    break;
                candidates.push(cj, nearest_d[j]);
            }

            search_from_candidates(
                    hnsw, qdis, res, candidates, vt, search_stats, 0);
        }
    }

    void HNSW::permute_entries(const idx_t *map) {
        // remap levels
        storage_idx_t ntotal = levels.size();
        std::vector<storage_idx_t> imap(ntotal);  // inverse mapping
        // map: new index -> old index
        // imap: old index -> new index
        for (int i = 0; i < ntotal; i++) {
            assert(map[i] >= 0 && map[i] < ntotal);
            imap[map[i]] = i;
        }
        if (entry_point != -1) {
            entry_point = imap[entry_point];
        }
        std::vector<int> new_levels(ntotal);
        std::vector<size_t> new_offsets(ntotal + 1);
        std::vector<storage_idx_t> new_neighbors(neighbors.size());
        size_t no = 0;
        for (int i = 0; i < ntotal; i++) {
            storage_idx_t o = map[i];  // corresponding "old" index
            new_levels[i] = levels[o];
            for (size_t j = offsets[o]; j < offsets[o + 1]; j++) {
                storage_idx_t neigh = neighbors[j];
                new_neighbors[no++] = neigh >= 0 ? imap[neigh] : neigh;
            }
            new_offsets[i + 1] = no;
        }
        assert(new_offsets[ntotal] == offsets[ntotal]);
        // swap everyone
        std::swap(levels, new_levels);
        std::swap(offsets, new_offsets);
        std::swap(neighbors, new_neighbors);
    }

/**************************************************************
 * MinimaxHeap
 **************************************************************/

    void HNSW::MinimaxHeap::push(storage_idx_t i, float v) {
        if (k == n) {
            if (v >= dis[0])
                return;
            if (ids[0] != -1) {//ids为-1，说明是无效值，即已经被pop，也即已经被访问过，
                --nvalid;//这行代码意思是，如果ids[0] != -1，说明堆顶是有效值，要插入一个元素，需要把有效值减1
            }
            faiss::heap_pop<HC>(k--, dis.data(), ids.data());
        }
        faiss::heap_push<HC>(++k, dis.data(), ids.data(), v, i);
        ++nvalid;
    }

    float HNSW::MinimaxHeap::max() const {
        return dis[0];
    }

    int HNSW::MinimaxHeap::size() const {
        return nvalid;
    }

    void HNSW::MinimaxHeap::clear() {
        nvalid = k = 0;
    }

#ifdef __AVX2__
    int HNSW::MinimaxHeap::pop_min(float* vmin_out) {
      assert(k > 0);
      static_assert(
          std::is_same<storage_idx_t, int32_t>::value,
          "This code expects storage_idx_t to be int32_t");

      int32_t min_idx = -1;
      float min_dis = std::numeric_limits<float>::infinity();

      size_t iii = 0;

      __m256i min_indices = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, -1);
      __m256 min_distances =
          _mm256_set1_ps(std::numeric_limits<float>::infinity());
      __m256i current_indices = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
      __m256i offset = _mm256_set1_epi32(8);

      // The baseline version is available in non-AVX2 branch.

      // The following loop tracks the rightmost index with the min distance.
      // -1 index values are ignored.
      const int k8 = (k / 8) * 8;
      for (; iii < k8; iii += 8) {
        __m256i indices =
            _mm256_loadu_si256((const __m256i*)(ids.data() + iii));
        __m256 distances = _mm256_loadu_ps(dis.data() + iii);

        // This mask filters out -1 values among indices.
        __m256i m1mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), indices);

        __m256i dmask = _mm256_castps_si256(
            _mm256_cmp_ps(min_distances, distances, _CMP_LT_OS));
        __m256 finalmask = _mm256_castsi256_ps(_mm256_or_si256(m1mask, dmask));

        const __m256i min_indices_new = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(current_indices),
            _mm256_castsi256_ps(min_indices),
            finalmask));

        const __m256 min_distances_new =
            _mm256_blendv_ps(distances, min_distances, finalmask);

        min_indices = min_indices_new;
        min_distances = min_distances_new;

        current_indices = _mm256_add_epi32(current_indices, offset);
      }

      // Vectorizing is doable, but is not practical
      int32_t vidx8[8];
      float vdis8[8];
      _mm256_storeu_ps(vdis8, min_distances);
      _mm256_storeu_si256((__m256i*)vidx8, min_indices);

      for (size_t j = 0; j < 8; j++) {
        if (min_dis > vdis8[j] || (min_dis == vdis8[j] && min_idx < vidx8[j])) {
          min_idx = vidx8[j];
          min_dis = vdis8[j];
        }
      }

      // process last values. Vectorizing is doable, but is not practical
      for (; iii < k; iii++) {
        if (ids[iii] != -1 && dis[iii] <= min_dis) {
          min_dis = dis[iii];
          min_idx = iii;
        }
      }

      if (min_idx == -1) {
        return -1;
      }

      if (vmin_out) {
        *vmin_out = min_dis;
      }
      int ret = ids[min_idx];
      ids[min_idx] = -1;
      --nvalid;
      return ret;
    }

#else

// baseline non-vectorized version
    int HNSW::MinimaxHeap::pop_min(float *vmin_out) {
        assert(k > 0);
        // returns min. This is an O(n) operation
        int i = k - 1;
        while (i >= 0) {
            if (ids[i] != -1) {
                break;
            }
            i--;
        }
        if (i == -1) {
            return -1;
        }
        int imin = i;
        float vmin = dis[i];
        i--;
        while (i >= 0) {
            if (ids[i] != -1 && dis[i] < vmin) {
                vmin = dis[i];
                imin = i;
            }
            i--;
        }
        if (vmin_out) {
            *vmin_out = vmin;
        }
        int ret = ids[imin];
        ids[imin] = -1;
        --nvalid;

        return ret;
    }

#endif

    int HNSW::MinimaxHeap::count_below(float thresh) {
        int n_below = 0;
        for (int i = 0; i < k; i++) {
            if (dis[i] < thresh) {
                n_below++;
            }
        }

        return n_below;
    }

}  // namespace faiss
