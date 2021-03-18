/*******************************************************************************
 *     ___                  _   ____  ____
 *    / _ \ _   _  ___  ___| |_|  _ \| __ )
 *   | | | | | | |/ _ \/ __| __| | | |  _ \
 *   | |_| | |_| |  __/\__ \ |_| |_| | |_) |
 *    \__\_\\__,_|\___||___/\__|____/|____/
 *
 *  Copyright (c) 2014-2019 Appsicle
 *  Copyright (c) 2019-2020 QuestDB
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 ******************************************************************************/
#include "jni.h"
#include <cstring>
#include <xmmintrin.h>
#include "util.h"
#include "simd.h"
#include "asmlib/asmlib.h"
#include "vec_agg.h"

#ifdef __APPLE__
#define __JLONG_REINTERPRET_CAST__(type, var)  (type)var
#else
#define __JLONG_REINTERPRET_CAST__(type, var)  reinterpret_cast<type>(var)
#endif

typedef struct {
    uint64_t c8[256];
    uint64_t c7[256];
    uint64_t c6[256];
    uint64_t c5[256];
    uint64_t c4[256];
    uint64_t c3[256];
    uint64_t c2[256];
    uint64_t c1[256];
} rscounts_t;

typedef struct index_t {
    uint64_t ts;
    uint64_t i;

    bool operator<(int64_t other) const {
        return ts < other;
    }

    bool operator>(int64_t other) const {
        return ts > other;
    }

    bool operator==(index_t other) const {
        return ts == other.ts;
    }
} index_t;

#define RADIX_SHUFFLE 0

#if RADIX_SHUFFLE == 0

template<uint16_t sh>
__SIMD_MULTIVERSION__
inline void radix_shuffle(uint64_t *counts, index_t *src, index_t *dest, uint64_t size) {
    _mm_prefetch(counts, _MM_HINT_NTA);
    for (uint64_t x = 0; x < size; x++) {
        const auto digit = (src[x].ts >> sh) & 0xffu;
        dest[counts[digit]] = src[x];
        counts[digit]++;
        _mm_prefetch(src + x + 64, _MM_HINT_T2);
    }
}

#elif RADIX_SHUFFLE == 1

__SIMD_MULTIVERSION__
template<uint16_t sh>
inline void radix_shuffle(uint64_t *counts, index_t *src, index_t *dest, uint64_t size) {
    _mm_prefetch(counts, _MM_HINT_NTA);
    Vec4q vec;
    Vec4q digitVec;
    int64_t values[4];
    int64_t digits[4];
    for (uint64_t x = 0; x < size; x += 4) {
        _mm_prefetch(src + x + 64, _MM_HINT_T0);
        vec.load(src + x);
        digitVec = (vec >> sh) & 0xff;

        vec.store(values);
        digitVec.store(digits);

        dest[counts[digits[0]]] = values[0];
        counts[digits[0]]++;

        dest[counts[digits[1]]] = values[1];
        counts[digits[1]]++;

        dest[counts[digits[2]]] = values[2];
        counts[digits[2]]++;

        dest[counts[digits[3]]] = values[3];
        counts[digits[3]]++;
    }
}

#elif RADIX_SHUFFLE == 2
template<uint16_t sh>
inline void radix_shuffle(uint64_t* counts, int64_t* src, int64_t* dest, uint64_t size) {
    _mm_prefetch(counts, _MM_HINT_NTA);
    Vec8q vec;
    Vec8q digitVec;
    int64_t values[8];
    int64_t digits[8];
    for (uint64_t x = 0; x < size; x += 8) {
        _mm_prefetch(src + x + 64, _MM_HINT_T0);
        vec.load(src + x);
        digitVec = (vec >> sh) & 0xff;

        vec.store(values);
        digitVec.store(digits);

        dest[counts[digits[0]]] = values[0];
        counts[digits[0]]++;

        dest[counts[digits[1]]] = values[1];
        counts[digits[1]]++;

        dest[counts[digits[2]]] = values[2];
        counts[digits[2]]++;

        dest[counts[digits[3]]] = values[3];
        counts[digits[3]]++;

        dest[counts[digits[4]]] = values[4];
        counts[digits[4]]++;

        dest[counts[digits[5]]] = values[5];
        counts[digits[5]]++;

        dest[counts[digits[6]]] = values[6];
        counts[digits[6]]++;

        dest[counts[digits[7]]] = values[7];
        counts[digits[7]]++;
    }
}
#endif

__SIMD_MULTIVERSION__
void radix_sort_long_index_asc_in_place(index_t *array, uint64_t size) {
    rscounts_t counts;
    memset(&counts, 0, 256 * 8 * sizeof(uint64_t));
    auto *cpy = (index_t *) malloc(size * sizeof(index_t));
    int64_t o8 = 0, o7 = 0, o6 = 0, o5 = 0, o4 = 0, o3 = 0, o2 = 0, o1 = 0;
    int64_t t8, t7, t6, t5, t4, t3, t2, t1;
    int64_t x;

    // calculate counts
    _mm_prefetch(counts.c8, _MM_HINT_NTA);
    for (x = 0; x < size; x++) {
        t8 = array[x].ts & 0xffu;
        t7 = (array[x].ts >> 8u) & 0xffu;
        t6 = (array[x].ts >> 16u) & 0xffu;
        t5 = (array[x].ts >> 24u) & 0xffu;
        t4 = (array[x].ts >> 32u) & 0xffu;
        t3 = (array[x].ts >> 40u) & 0xffu;
        t2 = (array[x].ts >> 48u) & 0xffu;
        t1 = (array[x].ts >> 56u) & 0xffu;
        counts.c8[t8]++;
        counts.c7[t7]++;
        counts.c6[t6]++;
        counts.c5[t5]++;
        counts.c4[t4]++;
        counts.c3[t3]++;
        counts.c2[t2]++;
        counts.c1[t1]++;
        _mm_prefetch(array + x + 64, _MM_HINT_T2);
    }

    // convert counts to offsets
    _mm_prefetch(&counts, _MM_HINT_NTA);
    for (x = 0; x < 256; x++) {
        t8 = o8 + counts.c8[x];
        t7 = o7 + counts.c7[x];
        t6 = o6 + counts.c6[x];
        t5 = o5 + counts.c5[x];
        t4 = o4 + counts.c4[x];
        t3 = o3 + counts.c3[x];
        t2 = o2 + counts.c2[x];
        t1 = o1 + counts.c1[x];
        counts.c8[x] = o8;
        counts.c7[x] = o7;
        counts.c6[x] = o6;
        counts.c5[x] = o5;
        counts.c4[x] = o4;
        counts.c3[x] = o3;
        counts.c2[x] = o2;
        counts.c1[x] = o1;
        o8 = t8;
        o7 = t7;
        o6 = t6;
        o5 = t5;
        o4 = t4;
        o3 = t3;
        o2 = t2;
        o1 = t1;
    }

    // radix
    radix_shuffle<0u>(counts.c8, array, cpy, size);
    radix_shuffle<8u>(counts.c7, cpy, array, size);
    radix_shuffle<16u>(counts.c6, array, cpy, size);
    radix_shuffle<24u>(counts.c5, cpy, array, size);
    radix_shuffle<32u>(counts.c4, array, cpy, size);
    radix_shuffle<40u>(counts.c3, cpy, array, size);
    radix_shuffle<48u>(counts.c2, array, cpy, size);
    radix_shuffle<56u>(counts.c1, cpy, array, size);
    free(cpy);
}

__SIMD_MULTIVERSION__
inline void swap(index_t *a, index_t *b) {
    const auto t = *a;
    *a = *b;
    *b = t;
}

/**
 * This function takes last element as pivot, places
 *  the pivot element at its correct position in sorted
 *   array, and places all smaller (smaller than pivot)
 *  to left of pivot and all greater elements to right
 *  of pivot
 *
 **/
__SIMD_MULTIVERSION__
uint64_t partition(index_t *index, uint64_t low, uint64_t high) {
    const auto pivot = index[high].ts;    // pivot
    auto i = (low - 1);  // Index of smaller element

    for (uint64_t j = low; j <= high - 1; j++) {
        // If current element is smaller than or
        // equal to pivot
        if (index[j].ts <= pivot) {
            i++;    // increment index of smaller element
            swap(&index[i], &index[j]);
        }
    }
    swap(&index[i + 1], &index[high]);
    return (i + 1);
}

/**
 * The main function that implements QuickSort
 * arr[] --> Array to be sorted,
 * low  --> Starting index,
 * high  --> Ending index
 **/
__SIMD_MULTIVERSION__
void quick_sort_long_index_asc_in_place(index_t *arr, int64_t low, int64_t high) {
    if (low < high) {
        /* pi is partitioning index, arr[p] is now
           at right place */
        uint64_t pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quick_sort_long_index_asc_in_place(arr, low, pi - 1);
        quick_sort_long_index_asc_in_place(arr, pi + 1, high);
    }
}

__SIMD_MULTIVERSION__
inline void sort(index_t *index, int64_t size) {
    if (size < 600) {
        quick_sort_long_index_asc_in_place(index, 0, size - 1);
    } else {
        radix_sort_long_index_asc_in_place(index, size);
    }
}

typedef struct {
    uint64_t value;
    uint32_t index_index;
} loser_node_t;

typedef struct {
    index_t *index;
    uint64_t pos;
    uint64_t size;
} index_entry_t;

typedef struct {
    index_t *index;
    int64_t size;
} java_index_entry_t;

template<typename T, int alignment, typename l_iteration>
__SIMD_MULTIVERSION__
inline int64_t align_to_store_nt(T *address, const int64_t max_count, const l_iteration iteration) {

    const auto unaligned = ((uint64_t) address) % alignment;
    constexpr int64_t iteration_increment = sizeof(T);
    if (unaligned != 0) {

        if (unaligned % iteration_increment == 0) {

            const auto head_iteration_count = std::min<int64_t>(max_count,
                                                                (alignment - unaligned) / iteration_increment);
            for (int i = 0; i < head_iteration_count; i++) {
                iteration(i);
            }

            return head_iteration_count;
        } else {
            return -1;
        }
    }
    return 0;
}

template<typename T, typename TVec, typename lambda_iteration, typename lambda_vec_iteration>
__SIMD_MULTIVERSION__
inline void run_vec_bulk(T *dest,
                         const int64_t count,
                         const lambda_iteration l_iteration,
                         const lambda_vec_iteration l_vec_iteration) {

    int64_t i = align_to_store_nt<T, TVec::store_nt_alignment()>(dest, count, l_iteration);

    if (i >= 0) {
        constexpr int64_t increment = TVec::size();
        const int64_t bulk_stop = count - increment + 1;
        for (; i < bulk_stop; i += increment) {
            l_vec_iteration(i);
        }
    } else {
        i = 0;
    }

    for (; i < count; i++) {
        l_iteration(i);
    }
}

template<class T, typename TVec, int vecsize>
__SIMD_MULTIVERSION__
inline void re_shuffle(const jlong pSrc, jlong pDest, const jlong pIndex, const jlong count) {

    static_assert(vecsize == 8 || vecsize == 16);
    auto src = reinterpret_cast<T *>(pSrc);
    auto dest = reinterpret_cast<T *>(pDest);
    auto index = reinterpret_cast<index_t *>(pIndex);

    const auto l_iteration = [dest, src, index](int64_t i) {
        dest[i] = src[index[i].i];
    };

    const auto bulk_reshuffle = [src, dest, index](const int64_t i) {
        if constexpr (vecsize == 16) {
            TVec(src[index[i + 0].i],
                   src[index[i + 1].i],
                   src[index[i + 2].i],
                   src[index[i + 3].i],
                   src[index[i + 4].i],
                   src[index[i + 5].i],
                   src[index[i + 6].i],
                   src[index[i + 7].i],
                   src[index[i + 8].i],
                   src[index[i + 9].i],
                   src[index[i + 10].i],
                   src[index[i + 11].i],
                   src[index[i + 12].i],
                   src[index[i + 13].i],
                   src[index[i + 14].i],
                   src[index[i + 15].i]).store_nt(dest + i);
        } else {
            TVec(src[index[i + 0].i],
                   src[index[i + 1].i],
                   src[index[i + 2].i],
                   src[index[i + 3].i],
                   src[index[i + 4].i],
                   src[index[i + 5].i],
                   src[index[i + 6].i],
                   src[index[i + 7].i]).store_nt(dest + i);
        }
    };

    run_vec_bulk<T, TVec>(
            dest,
            count,
            l_iteration,
            bulk_reshuffle
    );
}

template<typename T, typename TVec, int vecsize>
inline void merge_shuffle(const jlong pSrc1, const jlong pSrc2, jlong pDest, const jlong pIndex, const jlong count) {
    static_assert(vecsize == 8 || vecsize == 16);

    uint32_t pick_arr[16];
    auto src1 = reinterpret_cast<T *>(pSrc1);
    auto src2 = reinterpret_cast<T *>(pSrc2);
    auto dest = reinterpret_cast<T *>(pDest);
    auto index = reinterpret_cast<index_t *>(pIndex);
    const T *sources[] = {src2, src1};

    const auto merge = [dest, index, &sources](int64_t i) {
        const auto r = reinterpret_cast<uint64_t>(index[i].i);
        const uint64_t pick = r >> 63u;
        const auto row = r & ~(1LLu << 63u);
        dest[i] = sources[pick][row];
    };

    const auto bulk_merge = [&pick_arr, dest, index, sources](const int64_t i) {
        constexpr uint64_t row_mask = ~(1LLu << 63u);
        if constexpr (vecsize == 16) {
            // index is 2 longs
            // that is 4 ints. Source flag is highest bit
            // Take it as last bit of 4th integer
            Vec16ui ind = gather16i<3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63>(index + i);
            ind >>= 31u;
            ind.store(pick_arr);

            TVec(sources[pick_arr[0]][(index[i + 0].i) & row_mask],
                 sources[pick_arr[1]][(index[i + 1].i) & row_mask],
                 sources[pick_arr[2]][(index[i + 2].i) & row_mask],
                 sources[pick_arr[3]][(index[i + 3].i) & row_mask],
                 sources[pick_arr[4]][(index[i + 4].i) & row_mask],
                 sources[pick_arr[5]][(index[i + 5].i) & row_mask],
                 sources[pick_arr[6]][(index[i + 6].i) & row_mask],
                 sources[pick_arr[7]][(index[i + 7].i) & row_mask],
                 sources[pick_arr[8]][(index[i + 8].i) & row_mask],
                 sources[pick_arr[9]][(index[i + 9].i) & row_mask],
                 sources[pick_arr[10]][(index[i + 10].i) & row_mask],
                 sources[pick_arr[11]][(index[i + 11].i) & row_mask],
                 sources[pick_arr[12]][(index[i + 12].i) & row_mask],
                 sources[pick_arr[13]][(index[i + 13].i) & row_mask],
                 sources[pick_arr[14]][(index[i + 14].i) & row_mask],
                 sources[pick_arr[15]][(index[i + 15].i) & row_mask]
            ).store_nt(dest + i);
        } else {
            Vec8ui ind = gather8i<3, 7, 11, 15, 19, 23, 27, 31>(index + i);
            ind >>= 31u;
            ind.store(pick_arr);

            TVec(sources[pick_arr[0]][(index[i + 0].i) & row_mask],
                 sources[pick_arr[1]][(index[i + 1].i) & row_mask],
                 sources[pick_arr[2]][(index[i + 2].i) & row_mask],
                 sources[pick_arr[3]][(index[i + 3].i) & row_mask],
                 sources[pick_arr[4]][(index[i + 4].i) & row_mask],
                 sources[pick_arr[5]][(index[i + 5].i) & row_mask],
                 sources[pick_arr[6]][(index[i + 6].i) & row_mask],
                 sources[pick_arr[7]][(index[i + 7].i) & row_mask]
            ).store_nt(dest + i);
        }
    };

    run_vec_bulk<T, TVec>(
            reinterpret_cast<T *>(dest),
            count,
            merge,
            bulk_merge
    );
}

template<typename T, typename TVec, int vecsize>
inline void
merge_shuffle_top(const jlong pSrc1, const jlong pSrc2, jlong pDest, const jlong pIndex, const jlong count,
                  const jlong topOffset) {
    static_assert(vecsize == 8 || vecsize == 16);

    uint32_t pick_arr[16];
    auto src1 = reinterpret_cast<T *>(pSrc1);
    auto src2 = reinterpret_cast<T *>(pSrc2);
    auto dest = reinterpret_cast<T *>(pDest);
    auto index = reinterpret_cast<index_t *>(pIndex);
    constexpr int64_t sz = sizeof(T);
    const int64_t shifts[] = {0, static_cast<int64_t>(topOffset / sz)};
    const T *sources[] = {src2, src1};

    const auto merge = [dest, index, &shifts, &sources](int64_t i) {
        const auto r = reinterpret_cast<uint64_t>(index[i].i);
        const int64_t pick = r >> 63u;
        const auto row = r & ~(1LLu << 63u);
        dest[i] = sources[pick][row + shifts[pick]];
    };

    const auto bulk_merge = [&pick_arr, dest, index, &shifts, &sources](const int64_t i) {
        constexpr uint64_t row_mask = ~(1LLu << 63u);

        if constexpr (vecsize == 16) {
            // index is 2 longs
            // that is 4 ints. Source flag is highest bit
            // Take it as last bit of 4th integer
            Vec16ui ind = gather16i<3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63>(index + i);
            ind >>= 31u;
            ind.store(pick_arr);

            TVec(sources[pick_arr[0]][((index[i + 0].i) & row_mask) + shifts[pick_arr[0]]],
                 sources[pick_arr[1]][((index[i + 1].i) & row_mask) + shifts[pick_arr[1]]],
                 sources[pick_arr[2]][((index[i + 2].i) & row_mask) + shifts[pick_arr[2]]],
                 sources[pick_arr[3]][((index[i + 3].i) & row_mask) + shifts[pick_arr[3]]],
                 sources[pick_arr[4]][((index[i + 4].i) & row_mask) + shifts[pick_arr[4]]],
                 sources[pick_arr[5]][((index[i + 5].i) & row_mask) + shifts[pick_arr[5]]],
                 sources[pick_arr[6]][((index[i + 6].i) & row_mask) + shifts[pick_arr[6]]],
                 sources[pick_arr[7]][((index[i + 7].i) & row_mask) + shifts[pick_arr[7]]],
                 sources[pick_arr[8]][((index[i + 8].i) & row_mask) + shifts[pick_arr[8]]],
                 sources[pick_arr[9]][((index[i + 9].i) & row_mask) + shifts[pick_arr[9]]],
                 sources[pick_arr[10]][((index[i + 10].i) & row_mask) + shifts[pick_arr[10]]],
                 sources[pick_arr[11]][((index[i + 11].i) & row_mask) + shifts[pick_arr[11]]],
                 sources[pick_arr[12]][((index[i + 12].i) & row_mask) + shifts[pick_arr[12]]],
                 sources[pick_arr[13]][((index[i + 13].i) & row_mask) + shifts[pick_arr[13]]],
                 sources[pick_arr[14]][((index[i + 14].i) & row_mask) + shifts[pick_arr[14]]],
                 sources[pick_arr[15]][((index[i + 15].i) & row_mask) + shifts[pick_arr[15]]]
            ).store_nt(dest + i);
        } else {
            Vec8ui ind = gather8i<3, 7, 11, 15, 19, 23, 27, 31>(index + i);
            ind >>= 31u;
            ind.store(pick_arr);

            TVec(sources[pick_arr[0]][(((index[i + 0].i)) & row_mask) + shifts[pick_arr[0]]],
                 sources[pick_arr[1]][(((index[i + 1].i)) & row_mask) + shifts[pick_arr[1]]],
                 sources[pick_arr[2]][(((index[i + 2].i)) & row_mask) + shifts[pick_arr[2]]],
                 sources[pick_arr[3]][(((index[i + 3].i)) & row_mask) + shifts[pick_arr[3]]],
                 sources[pick_arr[4]][(((index[i + 4].i)) & row_mask) + shifts[pick_arr[4]]],
                 sources[pick_arr[5]][(((index[i + 5].i)) & row_mask) + shifts[pick_arr[5]]],
                 sources[pick_arr[6]][(((index[i + 6].i)) & row_mask) + shifts[pick_arr[6]]],
                 sources[pick_arr[7]][(((index[i + 7].i)) & row_mask) + shifts[pick_arr[7]]]
            ).store_nt(dest + i);
        }
    };

    run_vec_bulk<T, TVec>(
            reinterpret_cast<T *>(dest),
            count,
            merge,
            bulk_merge
    );
}

__SIMD_MULTIVERSION__
void k_way_merge_long_index(
        index_entry_t *indexes,
        uint32_t entries_count,
        uint32_t sentinels_at_start,
        index_t *dest
) {

    // calculate size of the tree
    uint32_t tree_size = entries_count * 2;
    uint64_t merged_index_pos = 0;
    uint32_t sentinels_left = entries_count - sentinels_at_start;

    loser_node_t tree[tree_size];

    // seed the bottom of the tree with index values
    for (uint32_t i = 0; i < entries_count; i++) {
        if (indexes[i].index != nullptr) {
            tree[entries_count + i].value = indexes[i].index->ts;
        } else {
            tree[entries_count + i].value = L_MAX;
        }
        tree[entries_count + i].index_index = entries_count + i;
    }

    // seed the entire tree from bottom up
    for (uint32_t i = tree_size - 1; i > 1; i -= 2) {
        uint32_t winner;
        if (tree[i].value < tree[i - 1].value) {
            winner = i;
        } else {
            winner = i - 1;
        }
        tree[i / 2] = tree[winner];
    }

    // take the first winner
    auto winner_index = tree[1].index_index;
    index_entry_t *winner = indexes + winner_index - entries_count;
    if (winner->pos < winner->size) {
        dest[merged_index_pos++] = winner->index[winner->pos];
    } else {
        sentinels_left--;
    }


    // full run
    while (sentinels_left > 0) {

        // back fill the winning index
        if (PREDICT_TRUE(++winner->pos < winner->size)) {
            tree[winner_index].value = winner->index[winner->pos].ts;
        } else {
            tree[winner_index].value = L_MAX;
            sentinels_left--;
        }

        if (sentinels_left == 0) {
            break;
        }

        _mm_prefetch(tree, _MM_HINT_NTA);
        while (PREDICT_TRUE(winner_index > 1)) {
            const auto right_child = winner_index % 2 == 1 ? winner_index - 1 : winner_index + 1;
            const auto target = winner_index / 2;
            if (tree[winner_index].value < tree[right_child].value) {
                tree[target] = tree[winner_index];
            } else {
                tree[target] = tree[right_child];
            }
            winner_index = target;
        }
        winner_index = tree[1].index_index;
        winner = indexes + winner_index - entries_count;
        _mm_prefetch(winner, _MM_HINT_NTA);
        dest[merged_index_pos++] = winner->index[winner->pos];
    }
}

inline void make_timestamp_index(const int64_t *data, int64_t low, int64_t high, index_t *dest) {

    // This code assumes that index_t is 16 bytes, 8 bytes ts and 8 bytes i
    static_assert(sizeof(index_t) == 16);

    int64_t l = low;
    Vec8q vec_i((low + 0) | (1ull << 63),
                (low + 1) | (1ull << 63),
                (low + 2) | (1ull << 63),
                (low + 3) | (1ull << 63),
                (low + 4) | (1ull << 63),
                (low + 5) | (1ull << 63),
                (low + 6) | (1ull << 63),
                (low + 7) | (1ull << 63));
    const Vec8q vec8(8);
    Vec8q vec_ts;

    for (; l <= high - 7; l += 8) {
        vec_ts.load(data + l);

        // save vec_ts into even 8b positions as index ts
        scatter<0, 2, 4, 6, 8, 10, 12, 14>(vec_ts, dest + l - low);

        // save vec_i into odd 8b positions as index i
        scatter<1, 3, 5, 7, 9, 11, 13, 15>(vec_i, dest + l - low);
        vec_i += vec8;
    }

    // tail
    for (; l <= high; l++) {
        dest[l - low].ts = data[l];
        dest[l - low].i = l | (1ull << 63);
    }
}

template<class T>
__SIMD_MULTIVERSION__
inline void merge_copy_var_column(
        index_t *merge_index,
        int64_t merge_index_size,
        int64_t *src_data_fix,
        char *src_data_var,
        int64_t *src_ooo_fix,
        char *src_ooo_var,
        int64_t *dst_fix,
        char *dst_var,
        int64_t dst_var_offset,
        T mult
) {
    int64_t *src_fix[] = {src_ooo_fix, src_data_fix};
    char *src_var[] = {src_ooo_var, src_data_var};

    for (int64_t l = 0; l < merge_index_size; l++) {
        _mm_prefetch(merge_index + 64, _MM_HINT_T0);
        dst_fix[l] = dst_var_offset;
        const uint64_t row = merge_index[l].i;
        const uint32_t bit = (row >> 63);
        const uint64_t rr = row & ~(1ull << 63);
        const int64_t offset = src_fix[bit][rr];
        char *src_var_ptr = src_var[bit] + offset;
        auto len = *reinterpret_cast<T *>(src_var_ptr);
        auto char_count = len > 0 ? len * mult : 0;
        reinterpret_cast<T *>(dst_var + dst_var_offset)[0] = len;
        __MEMCPY(dst_var + dst_var_offset + sizeof(T), src_var_ptr + sizeof(T), char_count);
        dst_var_offset += char_count + sizeof(T);
    }
}

template<class T>
__SIMD_MULTIVERSION__
inline void merge_copy_var_column_top(
        index_t *merge_index,
        int64_t merge_index_size,
        int64_t *src_data_fix,
        int64_t src_data_fix_offset,
        char *src_data_var,
        int64_t *src_ooo_fix,
        char *src_ooo_var,
        int64_t *dst_fix,
        char *dst_var,
        int64_t dst_var_offset,
        T mult
) {
    int64_t *src_fix[] = {src_ooo_fix, src_data_fix};
    char *src_var[] = {src_ooo_var, src_data_var};
    int64_t fix_shifts[] = {0, src_data_fix_offset / 8};

    for (int64_t l = 0; l < merge_index_size; l++) {
        _mm_prefetch(merge_index + 64, _MM_HINT_NTA);
        dst_fix[l] = dst_var_offset;
        const uint64_t row = merge_index[l].i;
        const uint32_t bit = (row >> 63);
        const uint64_t rr = row & ~(1ull << 63);
        const int64_t offset = src_fix[bit][rr + fix_shifts[bit]];
        char *src_var_ptr = src_var[bit] + offset;
        auto len = *reinterpret_cast<T *>(src_var_ptr);
        auto char_count = len > 0 ? len * mult : 0;
        reinterpret_cast<T *>(dst_var + dst_var_offset)[0] = len;
        __MEMCPY(dst_var + dst_var_offset + sizeof(T), src_var_ptr + sizeof(T), char_count);
        dst_var_offset += char_count + sizeof(T);
    }
}

template<typename T, typename TVec>
inline void set_memory_vanilla(T *addr, const T value, const int64_t count) {

    const auto l_iteration = [addr, value](int64_t i) {
        addr[i] = value;
    };

    const TVec vec(value);
    const auto l_bulk = [&vec, addr](const int64_t i) {
        vec.store_nt(addr + i);
    };

    run_vec_bulk<T, TVec>(addr, count, l_iteration, l_bulk);
}

template<class T>
inline void set_var_refs(int64_t *addr, const int64_t offset, const int64_t count) {

    const auto size = sizeof(T);
    const auto vec_inc = 8 * size;
    auto l_set_address = [addr, offset](int64_t i) { addr[i] = offset + i * size; };
    int64_t i = align_to_store_nt<int64_t, Vec8q::store_nt_alignment()>(addr, count, l_set_address);

    if (i >= 0) {
        Vec8q add(vec_inc);
        Vec8q v_addr(offset + (i + 0) * size,
                     offset + (i + 1) * size,
                     offset + (i + 2) * size,
                     offset + (i + 3) * size,
                     offset + (i + 4) * size,
                     offset + (i + 5) * size,
                     offset + (i + 6) * size,
                     offset + (i + 7) * size);

        for (; i < count - 7; i += 8) {
            v_addr.store_nt(addr + i);
            v_addr += add;
        }
    } else {
        // Pointer cannot be aligned
        i = 0;
    }

    // tail
    for (; i < count; i++) {
        addr[i] = offset + i * size;
    }
}


inline void copy_index(const index_t *index, const int64_t count, int64_t *dest) {
    auto l_iteration = [dest, index](int64_t i) {
        dest[i] = index[i].ts;
    };
    auto l_bulk = [dest, index] (int64_t i) {
        gather8q<0, 2, 4, 6, 8, 10, 12, 14>(index + i).store_nt(dest + i);
    };
    run_vec_bulk<int64_t, Vec8q>(dest, count, l_iteration, l_bulk);
}

extern "C" {

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_oooMergeCopyStrColumn(JNIEnv *env, jclass cl,
                                               jlong merge_index,
                                               jlong merge_index_size,
                                               jlong src_data_fix,
                                               jlong src_data_var,
                                               jlong src_ooo_fix,
                                               jlong src_ooo_var,
                                               jlong dst_fix,
                                               jlong dst_var,
                                               jlong dst_var_offset) {
    merge_copy_var_column<int32_t>(
            reinterpret_cast<index_t *>(merge_index),
            __JLONG_REINTERPRET_CAST__(int64_t, merge_index_size),
            reinterpret_cast<int64_t *>(src_data_fix),
            reinterpret_cast<char *>(src_data_var),
            reinterpret_cast<int64_t *>(src_ooo_fix),
            reinterpret_cast<char *>(src_ooo_var),
            reinterpret_cast<int64_t *>(dst_fix),
            reinterpret_cast<char *>(dst_var),
            __JLONG_REINTERPRET_CAST__(int64_t, dst_var_offset),
            2
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_oooMergeCopyStrColumnWithTop(JNIEnv *env, jclass cl,
                                                      jlong merge_index,
                                                      jlong merge_index_size,
                                                      jlong src_data_fix,
                                                      jlong src_data_fix_offset,
                                                      jlong src_data_var,
                                                      jlong src_ooo_fix,
                                                      jlong src_ooo_var,
                                                      jlong dst_fix,
                                                      jlong dst_var,
                                                      jlong dst_var_offset) {
    merge_copy_var_column_top<int32_t>(
            reinterpret_cast<index_t *>(merge_index),
            __JLONG_REINTERPRET_CAST__(int64_t, merge_index_size),
            reinterpret_cast<int64_t *>(src_data_fix),
            src_data_fix_offset,
            reinterpret_cast<char *>(src_data_var),
            reinterpret_cast<int64_t *>(src_ooo_fix),
            reinterpret_cast<char *>(src_ooo_var),
            reinterpret_cast<int64_t *>(dst_fix),
            reinterpret_cast<char *>(dst_var),
            __JLONG_REINTERPRET_CAST__(int64_t, dst_var_offset),
            2);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_oooMergeCopyBinColumnWithTop(JNIEnv *env, jclass cl,
                                                      jlong merge_index,
                                                      jlong merge_index_size,
                                                      jlong src_data_fix,
                                                      jlong src_data_fix_offset,
                                                      jlong src_data_var,
                                                      jlong src_ooo_fix,
                                                      jlong src_ooo_var,
                                                      jlong dst_fix,
                                                      jlong dst_var,
                                                      jlong dst_var_offset) {
    merge_copy_var_column_top<int64_t>(
            reinterpret_cast<index_t *>(merge_index),
            __JLONG_REINTERPRET_CAST__(int64_t, merge_index_size),
            reinterpret_cast<int64_t *>(src_data_fix),
            src_data_fix_offset,
            reinterpret_cast<char *>(src_data_var),
            reinterpret_cast<int64_t *>(src_ooo_fix),
            reinterpret_cast<char *>(src_ooo_var),
            reinterpret_cast<int64_t *>(dst_fix),
            reinterpret_cast<char *>(dst_var),
            __JLONG_REINTERPRET_CAST__(int64_t, dst_var_offset),
            1);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_oooMergeCopyBinColumn(JNIEnv *env, jclass cl,
                                               jlong merge_index,
                                               jlong merge_index_size,
                                               jlong src_data_fix,
                                               jlong src_data_var,
                                               jlong src_ooo_fix,
                                               jlong src_ooo_var,
                                               jlong dst_fix,
                                               jlong dst_var,
                                               jlong dst_var_offset) {
    merge_copy_var_column<int64_t>(
            reinterpret_cast<index_t *>(merge_index),
            __JLONG_REINTERPRET_CAST__(int64_t, merge_index_size),
            reinterpret_cast<int64_t *>(src_data_fix),
            reinterpret_cast<char *>(src_data_var),
            reinterpret_cast<int64_t *>(src_ooo_fix),
            reinterpret_cast<char *>(src_ooo_var),
            reinterpret_cast<int64_t *>(dst_fix),
            reinterpret_cast<char *>(dst_var),
            __JLONG_REINTERPRET_CAST__(int64_t, dst_var_offset),
            1
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_sortLongIndexAscInPlace(JNIEnv *env, jclass cl, jlong pLong, jlong len) {
    sort(reinterpret_cast<index_t *>(pLong), len);
}

__SIMD_MULTIVERSION__
JNIEXPORT jlong JNICALL
Java_io_questdb_std_Vect_mergeLongIndexesAsc(JNIEnv *env, jclass cl, jlong pIndexStructArray, jint count) {
    // prepare merge entries
    // they need to have mutable current position "pos" in index

    if (count < 1) {
        return 0;
    }

    const java_index_entry_t *java_entries = reinterpret_cast<java_index_entry_t *>(pIndexStructArray);
    if (count == 1) {
        return reinterpret_cast<jlong>(java_entries[0].index);
    }

    uint32_t size = ceil_pow_2(count);
    index_entry_t entries[size];
    uint64_t merged_index_size = 0;
    for (jint i = 0; i < count; i++) {
        entries[i].index = java_entries[i].index;
        entries[i].pos = 0;
        entries[i].size = java_entries[i].size;
        merged_index_size += java_entries[i].size;
    }

    if (count < size) {
        for (uint32_t i = count; i < size; i++) {
            entries[i].index = nullptr;
            entries[i].pos = 0;
            entries[i].size = -1;
        }
    }

    auto *merged_index = reinterpret_cast<index_t *>(malloc(merged_index_size * sizeof(index_t)));
    k_way_merge_long_index(entries, size, size - count, merged_index);
    return reinterpret_cast<jlong>(merged_index);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_freeMergedIndex(JNIEnv *env, jclass cl, jlong pIndex) {
    free(reinterpret_cast<void *>(pIndex));
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_indexReshuffle32Bit(JNIEnv *env, jclass cl, jlong pSrc, jlong pDest, jlong pIndex,
                                             jlong count) {
    re_shuffle<int32_t, Vec16i, 16>(pSrc, pDest, pIndex, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_indexReshuffle64Bit(JNIEnv *env, jclass cl, jlong pSrc, jlong pDest, jlong pIndex,
                                             jlong count) {
    re_shuffle<int64_t, Vec8q, 8>(pSrc, pDest, pIndex, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_indexReshuffle16Bit(JNIEnv *env, jclass cl, jlong pSrc, jlong pDest, jlong pIndex,
                                             jlong count) {
    re_shuffle<int16_t, Vec16s, 16>(pSrc, pDest, pIndex, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_indexReshuffle8Bit(JNIEnv *env, jclass cl, jlong pSrc, jlong pDest, jlong pIndex,
                                            jlong count) {
    re_shuffle<int8_t, Vec16c, 16>(pSrc, pDest, pIndex, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffle8Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest, jlong index,
                                          jlong count) {
    merge_shuffle<int8_t, Vec16c, 16>(src1, src2, dest, index, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffle16Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest, jlong index,
                                           jlong count) {
    merge_shuffle<int16_t, Vec16s, 16>(src1, src2, dest, index, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffle32Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest, jlong index,
                                           jlong count) {
    merge_shuffle<int32_t, Vec16i, 16>(src1, src2, dest, index, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffle64Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest, jlong index,
                                           jlong count) {
    merge_shuffle<int64_t, Vec8q, 8>(src1, src2, dest, index, count);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffleWithTop64Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest,
                                                  jlong index,
                                                  jlong count, jlong topOffset) {
    merge_shuffle_top<int64_t, Vec8q, 8>(src1, src2, dest, index, count, topOffset);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffleWithTop32Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest,
                                                  jlong index,
                                                  jlong count, jlong topOffset) {
    merge_shuffle_top<int32_t, Vec16i, 16>(src1, src2, dest, index, count, topOffset);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffleWithTop16Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest,
                                                  jlong index,
                                                  jlong count, jlong topOffset) {
    merge_shuffle_top<int16_t, Vec16s, 16>(src1, src2, dest, index, count, topOffset);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_mergeShuffleWithTop8Bit(JNIEnv *env, jclass cl, jlong src1, jlong src2, jlong dest,
                                                 jlong index,
                                                 jlong count, jlong topOffset) {
    merge_shuffle_top<int8_t, Vec16c, 16>(src1, src2, dest, index, count, topOffset);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_flattenIndex(JNIEnv *env, jclass cl, jlong pIndex,
                                      jlong count) {

    auto *index = reinterpret_cast<index_t *>(pIndex);
    Vec8q v_i = Vec8q(0, 1, 2, 3, 4, 5, 6, 7);
    Vec8q v_inc = Vec8q(8);
    int64_t i = 0;
    for (; i < count - 7; i += 8) {
        scatter<1, 3, 5, 7, 9, 11, 13, 15>(v_i, index + i);
        v_i += v_inc;
    }

    // tail.
    for (int64_t i = 0; i < count; i++) {
        index[i].i = i;
    }
}

__SIMD_MULTIVERSION__
JNIEXPORT jlong JNICALL
Java_io_questdb_std_Vect_binarySearch64Bit(JNIEnv *env, jclass cl, jlong pData, jlong value, jlong low,
                                           jlong high, jint scan_dir) {
    return binary_search<int64_t>(reinterpret_cast<int64_t *>(pData), value, low, high, scan_dir);
}

__SIMD_MULTIVERSION__
JNIEXPORT jlong JNICALL
Java_io_questdb_std_Vect_binarySearchIndexT(JNIEnv *env, jclass cl, jlong pData, jlong value, jlong low,
                                            jlong high, jint scan_dir) {
    return binary_search<index_t>(reinterpret_cast<index_t *>(pData), value, low, high, scan_dir);
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_makeTimestampIndex(JNIEnv *env, jclass cl, jlong pData, jlong low,
                                            jlong high, jlong pIndex) {
    make_timestamp_index(
            reinterpret_cast<int64_t *>(pData),
            low,
            high,
            reinterpret_cast<index_t *>(pIndex)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setMemoryLong(JNIEnv *env, jclass cl, jlong pData, jlong value,
                                       jlong count) {
    set_memory_vanilla<int64_t, Vec8q>(
            reinterpret_cast<int64_t *>(pData),
            __JLONG_REINTERPRET_CAST__(int64_t, value),
            (int64_t) (count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setMemoryInt(JNIEnv *env, jclass cl, jlong pData, jint value,
                                      jlong count) {
    set_memory_vanilla<jint, Vec16i>(
            reinterpret_cast<jint *>(pData),
            value,
            (int64_t) (count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setMemoryDouble(JNIEnv *env, jclass cl, jlong pData, jdouble value,
                                         jlong count) {
    set_memory_vanilla<jdouble, Vec8d>(
            reinterpret_cast<jdouble *>(pData),
            value,
            (int64_t) (count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setMemoryFloat(JNIEnv *env, jclass cl, jlong pData, jfloat value,
                                        jlong count) {
    set_memory_vanilla<jfloat, Vec16f>(
            reinterpret_cast<jfloat *>(pData),
            value,
            (int64_t) (count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setMemoryShort(JNIEnv *env, jclass cl, jlong pData, jshort value,
                                        jlong count) {
    set_memory_vanilla<jshort, Vec32s>(
            reinterpret_cast<jshort *>(pData),
            value,
            (int64_t) (count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setVarColumnRefs32Bit(JNIEnv *env, jclass cl, jlong pData, jlong offset,
                                               jlong count) {
    set_var_refs<int32_t>(
            reinterpret_cast<int64_t *>(pData),
            __JLONG_REINTERPRET_CAST__(int64_t, offset),
            __JLONG_REINTERPRET_CAST__(int64_t, count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_setVarColumnRefs64Bit(JNIEnv *env, jclass cl, jlong pData, jlong offset,
                                               jlong count) {
    set_var_refs<int64_t>(
            reinterpret_cast<int64_t *>(pData),
            __JLONG_REINTERPRET_CAST__(int64_t, offset),
            __JLONG_REINTERPRET_CAST__(int64_t, count)
    );
}

__SIMD_MULTIVERSION__
JNIEXPORT void JNICALL
Java_io_questdb_std_Vect_oooCopyIndex(JNIEnv *env, jclass cl, jlong pIndex, jlong index_size,
                                      jlong pDest) {
    copy_index(
            reinterpret_cast<index_t *>(pIndex),
            __JLONG_REINTERPRET_CAST__(int64_t, index_size),
            reinterpret_cast<int64_t *>(pDest)
    );
}
}

// ====================================================================
// OOO SIMD optimisation benchmarking
/// ===================================================================

inline __attribute__((always_inline))
void man_memcpy(char *__restrict__ destb, const char *__restrict__ srcb, size_t count) {
    char *d = destb;
    const char *s = srcb;
    while (count--)
        *d++ = *s++;
}


#define oooMergeCopyStrColumnMv(_SUFFIX, _MEMCPYSTYLE) \
template<class T>\
inline \
void merge_copy_var_column##_SUFFIX(\
        index_t *merge_index,\
        int64_t merge_index_size,\
        int64_t *src_data_fix,\
        char *src_data_var,\
        int64_t *src_ooo_fix,\
        char *src_ooo_var,\
        int64_t *dst_fix,\
        char *dst_var,\
        int64_t dst_var_offset,\
        T mult\
) {\
    int64_t *src_fix[] = {src_ooo_fix, src_data_fix};\
    char *src_var[] = {src_ooo_var, src_data_var};\
    for (int64_t l = 0; l < merge_index_size; l++) {\
        _mm_prefetch(merge_index + 64, _MM_HINT_T0);\
        dst_fix[l] = dst_var_offset;\
        const uint64_t row = merge_index[l].i;\
        const uint32_t bit = (row >> 63);\
        const uint64_t rr = row & ~(1ull << 63);\
        const int64_t offset = src_fix[bit][rr];\
        char *src_var_ptr = src_var[bit] + offset;\
        auto len = *reinterpret_cast<T *>(src_var_ptr);\
        auto char_count = len > 0 ? len * mult : 0;\
        reinterpret_cast<T *>(dst_var + dst_var_offset)[0] = len;\
        _MEMCPYSTYLE(dst_var + dst_var_offset + sizeof(T), src_var_ptr + sizeof(T), char_count);\
        dst_var_offset += char_count + sizeof(T);\
    }\
}\
                                                       \
extern "C" {                                                       \
JNIEXPORT void JNICALL \
__attribute__((target_clones("avx2", "avx", "avx512f", "sse4.1", "default"))) \
Java_io_questdb_std_Vect_oooMergeCopyStrColumn##_SUFFIX(JNIEnv *env, jclass cl,   \
                                                 jlong merge_index,             \
                                                 jlong merge_index_size,        \
                                                 jlong src_data_fix,\
                                                 jlong src_data_var,\
                                                 jlong src_ooo_fix,\
                                                 jlong src_ooo_var,\
                                                 jlong dst_fix,\
                                                 jlong dst_var,\
                                                 jlong dst_var_offset) {\
        merge_copy_var_column##_SUFFIX<int32_t>(\
                reinterpret_cast<index_t *>(merge_index),\
                __JLONG_REINTERPRET_CAST__(int64_t,merge_index_size),\
                reinterpret_cast<int64_t *>(src_data_fix),\
                reinterpret_cast<char *>(src_data_var),\
                reinterpret_cast<int64_t *>(src_ooo_fix),\
                reinterpret_cast<char *>(src_ooo_var),\
                reinterpret_cast<int64_t *>(dst_fix),\
                reinterpret_cast<char *>(dst_var),\
                __JLONG_REINTERPRET_CAST__(int64_t,dst_var_offset),\
                2\
        );\
    }                                           \
}                                                      \


#define oooMergeCopyStrColumnInline(_SUFFIX, _MEMCPYSTYLE) \
template<class T>\
inline \
void merge_copy_var_column##_SUFFIX(\
        index_t *merge_index,\
        int64_t merge_index_size,\
        int64_t *src_data_fix,\
        char *src_data_var,\
        int64_t *src_ooo_fix,\
        char *src_ooo_var,\
        int64_t *dst_fix,\
        char *dst_var,\
        int64_t dst_var_offset,\
        T mult\
) {\
    int64_t *src_fix[] = {src_ooo_fix, src_data_fix};\
    char *src_var[] = {src_ooo_var, src_data_var};\
    for (int64_t l = 0; l < merge_index_size; l++) {\
        _mm_prefetch(merge_index + 64, _MM_HINT_T0);\
        dst_fix[l] = dst_var_offset;\
        const uint64_t row = merge_index[l].i;\
        const uint32_t bit = (row >> 63);\
        const uint64_t rr = row & ~(1ull << 63);\
        const int64_t offset = src_fix[bit][rr];\
        char *src_var_ptr = src_var[bit] + offset;\
        auto len = *reinterpret_cast<T *>(src_var_ptr);\
        auto char_count = len > 0 ? len * mult : 0;\
        reinterpret_cast<T *>(dst_var + dst_var_offset)[0] = len;\
        _MEMCPYSTYLE(dst_var + dst_var_offset + sizeof(T), src_var_ptr + sizeof(T), char_count);\
        dst_var_offset += char_count + sizeof(T);\
    }\
}                                                          \
                                                           \
extern "C" {                                                           \
JNIEXPORT void JNICALL                                                          \
Java_io_questdb_std_Vect_oooMergeCopyStrColumn##_SUFFIX(JNIEnv *env, jclass cl,   \
                                                 jlong merge_index,             \
                                                 jlong merge_index_size,        \
                                                 jlong src_data_fix,\
                                                 jlong src_data_var,\
                                                 jlong src_ooo_fix,\
                                                 jlong src_ooo_var,\
                                                 jlong dst_fix,\
                                                 jlong dst_var,\
                                                 jlong dst_var_offset) {\
        merge_copy_var_column##_SUFFIX<int32_t>(\
                reinterpret_cast<index_t *>(merge_index),\
                __JLONG_REINTERPRET_CAST__(int64_t,merge_index_size),\
                reinterpret_cast<int64_t *>(src_data_fix),\
                reinterpret_cast<char *>(src_data_var),\
                reinterpret_cast<int64_t *>(src_ooo_fix),\
                reinterpret_cast<char *>(src_ooo_var),\
                reinterpret_cast<int64_t *>(dst_fix),\
                reinterpret_cast<char *>(dst_var),\
                __JLONG_REINTERPRET_CAST__(int64_t,dst_var_offset),\
                2\
        );\
    }                                                      \
}                                                          \

#ifdef ENABLE_MULTIVERSION

oooMergeCopyStrColumnMv(MvMemcpy, memcpy)

oooMergeCopyStrColumnMv(MvAMemcpy, A_memcpy)

oooMergeCopyStrColumnMv(MvManMemcpy, man_memcpy)
#else

#endif

oooMergeCopyStrColumnInline(InlMemcpy, memcpy)

oooMergeCopyStrColumnInline(InlAMemcpy, A_memcpy)

oooMergeCopyStrColumnInline(InlManMemcpy, man_memcpy)

