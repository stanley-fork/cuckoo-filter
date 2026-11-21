#pragma once

#include <thrust/device_vector.h>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <mutex>
#include <vector>
#include "bucket_policies.cuh"
#include "hashutil.cuh"
#include "helpers.cuh"

namespace cuckoogpu {

/**
 * @brief Eviction policy for the Cuckoo Filter.
 */
enum class EvictionPolicy {
    BFS,  ///< Breadth-first search (default)
    DFS   ///< Pure depth-first search
};

/**
 * @brief Configuration structure for the Cuckoo Filter.
 *
 * This struct defines the compile-time parameters for the Cuckoo Filter,
 * including key type, tag size, bucket size, and eviction policies.
 *
 * @tparam T The type of the keys stored in the filter.
 * @tparam bitsPerTag_ Number of bits per fingerprint tag.
 * @tparam maxEvictions_ Maximum number of evictions during insertion before failure.
 * @tparam blockSize_ CUDA block size for kernels.
 * @tparam bucketSize_ Number of slots per bucket.
 * @tparam AltBucketPolicy_ Policy for calculating alternate bucket indices.
 * @tparam evictionPolicy_ Policy for eviction during insertion (DFS or BFS).
 * @tparam WordType_ The atomic word type for bucket storage (uint32_t or uint64_t).
 */
template <
    typename T,
    size_t bitsPerTag_,
    size_t maxEvictions_ = 500,
    size_t blockSize_ = 256,
    size_t bucketSize_ = 16,
    template <typename, typename, size_t, size_t> class AltBucketPolicy_ = XorAltBucketPolicy,
    EvictionPolicy evictionPolicy_ = EvictionPolicy::BFS,
    typename WordType_ = uint64_t>
struct Config {
    using KeyType = T;
    static constexpr size_t bitsPerTag = bitsPerTag_;
    static constexpr size_t maxEvictions = maxEvictions_;
    static constexpr size_t blockSize = blockSize_;
    static constexpr size_t bucketSize = bucketSize_;
    static constexpr EvictionPolicy evictionPolicy = evictionPolicy_;

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::type;

    using WordType = WordType_;
    static_assert(
        std::is_same_v<WordType, uint32_t> || std::is_same_v<WordType, uint64_t>,
        "WordType must be uint32_t or uint64_t"
    );
    static_assert(sizeof(TagType) <= sizeof(WordType), "TagType must fit within WordType");

    /// The top bit of every slot holds the bucket's overflow mask, so fingerprints use the rest.
    using AltBucketPolicy = AltBucketPolicy_<KeyType, TagType, bitsPerTag - 1, bucketSize_>;
};

template <typename Config>
class Filter;

namespace detail {

/**
 * @brief Kernel for inserting keys into the filter.
 */
template <typename Config>
__global__ void insertKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    Filter<Config>* filter,
    uint32_t* evictionAttempts
);

/**
 * @brief Kernel for inserting pre-sorted keys into the filter.
 */
template <typename Config>
__global__ void insertKernelSorted(
    const typename Filter<Config>::PackedTagType* packedTags,
    bool* output,
    size_t n,
    Filter<Config>* filter,
    uint32_t* evictionAttempts
);

/**
 * @brief Kernel for computing packed tags for sorting.
 */
template <typename Config>
__global__ void computePackedTagsKernel(
    const typename Config::KeyType* keys,
    typename Filter<Config>::PackedTagType* packedTags,
    size_t n,
    size_t numBuckets
);

/**
 * @brief Kernel for checking existence of keys.
 */
template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    Filter<Config>* filter
);

/**
 * @brief Kernel for deleting keys.
 */
template <typename Config>
__global__ void
deleteKernel(const typename Config::KeyType* keys, bool* output, size_t n, Filter<Config>* filter);

/// Keys handled per thread by the region scatter kernel. Larger tiles mean shorter per-tile run
/// tables for the gather; sixteen keys still leave enough loads in flight at three blocks per SM.
inline constexpr size_t partitionItemsPerThread = 16;

/// Keys handled per thread by `gatherBitmapKernel`, whose blocks each cover one scatter tile.
inline constexpr size_t gatherItemsPerThread = 8;

/// Upper bound on the number of regions a batch is split into (sizes shared memory).
inline constexpr uint32_t maxRegions = 1024;

/// Threads per block of `regionSortedKernel`; also its number of local-bucket bins.
inline constexpr uint32_t sortedThreads = 512;

/// Resident `regionSortedKernel` blocks per SM. Two blocks let one block's DRAM-bound tag load
/// and sort overlap the other's L2-bound bucket visits; more would spill registers.
inline constexpr uint32_t sortedBlocksPerSm = 2;

/// What `regionSortedKernel` does with each entry of a region-grouped batch.
enum class RegionOp {
    Contains,
    Insert,
    Delete
};

/**
 * Region-grouped batch entries one `regionSortedKernel` block sorts and visits: 80 bytes of
 * entries per thread, the most for which two blocks' chunks still fit in an SM's shared memory.
 * Larger chunks are denser, so a warp's sorted entries touch fewer cache lines.
 */
template <typename RegionTagType>
inline constexpr uint32_t sortedChunk = sortedThreads * (80 / sizeof(RegionTagType));

/// Dynamic shared memory of `regionSortedKernel`: the sorted chunk plus its result bitmap.
template <typename RegionTagType>
inline constexpr size_t sortedSharedBytes =
    sortedChunk<RegionTagType> * sizeof(RegionTagType) + sortedChunk<RegionTagType> / 8;

/**
 * @brief Layout of a region-grouped batch: region r owns slots [r * capacity, (r + 1) * capacity)
 * and covers the buckets whose index has r as its top bits.
 */
struct RegionLayout {
    uint32_t localBits;   ///< Bucket-index bits below the region index
    uint32_t numRegions;  ///< Number of regions (power of two)
    uint32_t capacity;    ///< Slots per region
};

/**
 * @brief Where a region-grouped batch put each input key, in 16 bits per key instead of a 32-bit
 * slot. The scatter kernel claims, for each tile of keys and each region, a run of consecutive
 * slots starting at runSlot[t * numRegions + b]. Key i's code keyRun[i] holds its region b in
 * the top log2(numRegions) bits and its rank within the run in the rest, so it lands in slot
 * runSlot[...] + rank. A run too long for its rank bits counts as a region overflow.
 */
struct KeyPlacement {
    uint16_t* keyRun;   ///< Per key: region (high bits) and rank within its tile's run (low bits)
    uint32_t* runSlot;  ///< Per tile and region: slot of the run's first key
};

/// Bits of a `KeyPlacement` code that hold the rank within a run.
__device__ inline uint32_t runRankBits(uint32_t numRegions) {
    return 16 - (31 - __clz(numRegions));
}

/**
 * @brief Hashes keys and scatters region-local tags into their region's slots. Sets *overflow if
 * any region runs out of capacity. Fills `placement` unless its pointers are null.
 */
template <typename Config>
__global__ void regionScatterKernel(
    const typename Config::KeyType* keys,
    size_t n,
    size_t numBuckets,
    RegionLayout layout,
    uint32_t* regionCounts,
    typename Filter<Config>::RegionTagType* tags,
    KeyPlacement placement,
    uint32_t* overflow
);

/**
 * @brief Applies `Op` to every occupied slot of a region-grouped batch, one chunk of a region per
 * block. If `results` is non-null, writes a bitmap with one bit per slot. Mutations update the
 * occupancy counter.
 */
template <typename Config, RegionOp Op>
__global__ void regionSortedKernel(
    const typename Filter<Config>::RegionTagType* tags,
    const uint32_t* regionCounts,
    RegionLayout layout,
    uint32_t* results,
    Filter<Config>* filter
);

/**
 * @brief output[i] = the `results` bit of the slot key i was scattered to, one scatter tile per
 * block.
 */
template <typename Config>
__global__ void gatherBitmapKernel(
    const uint32_t* results,
    KeyPlacement placement,
    uint32_t numRegions,
    const uint32_t* overflow,
    bool* output,
    size_t n
);

}  // namespace detail

/**
 * @brief A CUDA-accelerated Cuckoo Filter implementation.
 *
 * This class implements a Cuckoo Filter using CUDA for high-throughput
 * insertion, lookup, and deletion. It supports concurrent operations
 * and uses atomic operations for thread safety within buckets.
 *
 * @tparam Config The configuration structure defining filter parameters.
 */
template <typename Config>
struct Filter {
    using T = typename Config::KeyType;
    static constexpr size_t bitsPerTag = Config::bitsPerTag;

    using TagType = typename Config::TagType;
    using AltBucketPolicy = typename Config::AltBucketPolicy;

    static constexpr size_t tagEntryBytes = sizeof(TagType);
    static constexpr size_t bucketSize = Config::bucketSize;

    static constexpr size_t maxEvictions = Config::maxEvictions;
    static constexpr size_t blockSize = Config::blockSize;
    static_assert(
        bitsPerTag == 8 || bitsPerTag == 16 || bitsPerTag == 32,
        "The tag must be 8, 16 or 32 bits"
    );

    static_assert(detail::powerOfTwo(bucketSize), "Bucket size must be a power of 2");

    using PackedTagType = typename std::conditional<bitsPerTag <= 8, uint32_t, uint64_t>::type;

    /// Region-local tag: the primary bucket's index bits below its region, then the fingerprint.
    using RegionTagType = std::conditional_t<bitsPerTag <= 16, uint32_t, uint64_t>;

    /**
     * @brief This is used by the sorted insert kernel to store the fingerprint and primary bucket
     * index in a compact format that allows you to sort them directly since the bucket index lives
     * in the upper bits.
     *
     */
    struct PackedTag {
        PackedTagType value;

        // Lower bits = fingerprint
        // Upper bits = bucket index
        static constexpr size_t fpBits = bitsPerTag;
        static constexpr size_t totalBits = sizeof(PackedTagType) * 8;
        static constexpr size_t bucketIdxBits = totalBits - fpBits;

        static_assert(fpBits < totalBits, "fpBits must leave at least some bits for bucketIdx");

        static constexpr PackedTagType fpMask = PackedTagType((1ULL << fpBits) - 1ULL);

        static constexpr PackedTagType bucketIdxMask =
            PackedTagType(((1ULL << bucketIdxBits) - 1ULL) << fpBits);

        __host__ __device__ PackedTag() : value(0) {
        }

        __host__ __device__ PackedTag(TagType fp, uint64_t bucketIdx) : value(0) {
            setFingerprint(fp);
            setBucketIdx(bucketIdx);
        }

        __host__ __device__ TagType getFingerprint() const {
            return static_cast<TagType>(value & fpMask);
        }

        __host__ __device__ uint64_t getBucketIndex() const {
            return uint64_t((value & bucketIdxMask) >> fpBits);
        }

        __host__ __device__ void setFingerprint(TagType fp) {
            value = (value & ~fpMask) | (static_cast<PackedTagType>(fp) & fpMask);
        }

        __host__ __device__ void setBucketIdx(size_t bucketIdx) {
            PackedTagType v = static_cast<PackedTagType>(bucketIdx) << fpBits;
            value = (value & ~bucketIdxMask) | v;
        }
    };

    static constexpr TagType EMPTY = 0;
    /// Fingerprint bits of a slot. The top bit of every slot instead belongs to its bucket's
    /// overflow mask (see `Bucket::overflowBit`).
    static constexpr size_t fpMask = (1ULL << (bitsPerTag - 1)) - 1;

    /**
     * @brief Bucket structure that holds the fingerprint and tags for a given bucket.
     * The bucket is divided into words, where each word contains one or more fingerprints
     * depending on tag size.
     *
     * This optimisation allows us to avoid having to perform atomic operations on every fingerprint
     * in the bucket, the extra computational overhead is negligible.
     *
     * For efficiency reasons, the number of fingerprints per word is enforced to be a power of 2,
     * same goes for the total number of fingerprints in the bucket.
     *
     */
    struct Bucket {
        static_assert(detail::powerOfTwo(bitsPerTag), "bitsPerTag must be a power of 2");

        using WordType = typename Config::WordType;

        static constexpr size_t tagsPerWord = sizeof(WordType) / sizeof(TagType);
        static_assert(tagsPerWord >= 1, "TagType must fit within WordType");
        static_assert(bucketSize % tagsPerWord == 0, "bucketSize must be divisible by tagsPerWord");
        static_assert(detail::powerOfTwo(tagsPerWord), "tagsPerWord must be a power of 2");

        static constexpr size_t wordCount = bucketSize / tagsPerWord;
        static_assert(detail::powerOfTwo(wordCount), "wordCount must be a power of 2");

        cuda::std::atomic<WordType> packedTags[wordCount];

        /**
         * @brief Relaxed, device-scope atomic view of one bucket word.
         *
         * The filter is only mutated by kernels on its own device, so system scope is not needed.
         * Raw intrinsics keep `expected` in registers; libcu++ atomics spill it to local memory.
         */
        struct WordRef {
            WordType* ptr;

            __device__ __forceinline__ WordType load(cuda::memory_order) const {
                WordType v;
                if constexpr (sizeof(WordType) == 8) {
                    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(ptr));
                } else {
                    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(ptr));
                }
                return v;
            }

            __device__ __forceinline__ bool compare_exchange_strong(
                WordType& expected,
                WordType desired,
                cuda::memory_order,
                cuda::memory_order
            ) const {
                WordType old;
                if constexpr (sizeof(WordType) == 8) {
                    asm volatile("atom.relaxed.gpu.global.cas.b64 %0, [%1], %2, %3;"
                                 : "=l"(old)
                                 : "l"(ptr), "l"(expected), "l"(desired)
                                 : "memory");
                } else {
                    asm volatile("atom.relaxed.gpu.global.cas.b32 %0, [%1], %2, %3;"
                                 : "=r"(old)
                                 : "l"(ptr), "r"(expected), "r"(desired)
                                 : "memory");
                }
                const bool success = old == expected;
                expected = old;
                return success;
            }

            __device__ __forceinline__ bool compare_exchange_weak(
                WordType& expected,
                WordType desired,
                cuda::memory_order success,
                cuda::memory_order failure
            ) const {
                return compare_exchange_strong(expected, desired, success, failure);
            }

            __device__ __forceinline__ void fetchOr(WordType bits) const {
                if constexpr (sizeof(WordType) == 8) {
                    asm volatile("red.relaxed.gpu.global.or.b64 [%0], %1;"
                                 :
                                 : "l"(ptr), "l"(bits)
                                 : "memory");
                } else {
                    asm volatile("red.relaxed.gpu.global.or.b32 [%0], %1;"
                                 :
                                 : "l"(ptr), "r"(bits)
                                 : "memory");
                }
            }
        };

        __device__ __forceinline__ WordRef word(size_t idx) {
            return {reinterpret_cast<WordType*>(&packedTags[idx])};
        }

        __host__ __device__ __forceinline__ TagType
        extractTag(WordType packed, size_t tagIdx) const {
            return static_cast<TagType>((packed >> (tagIdx * bitsPerTag)) & fpMask);
        }

        __host__ __device__ __forceinline__ WordType
        replaceTag(WordType packed, size_t tagIdx, TagType newTag) const {
            size_t shift = tagIdx * bitsPerTag;
            WordType cleared = packed & ~(static_cast<WordType>(fpMask) << shift);
            return cleared | (static_cast<WordType>(newTag) << shift);
        }

        /// Fingerprint bits of every slot in a word.
        static constexpr WordType fpLanes =
            detail::replicateTag<TagType, WordType>(static_cast<TagType>(fpMask));

        /// Marks slots whose fingerprint is zero (empty), ignoring overflow bits.
        __host__ __device__ __forceinline__ static constexpr WordType fpZeroMask(WordType v) {
            return detail::getZeroMask<TagType, WordType>(v & fpLanes);
        }

        /**
         * Overflow mask: a tag with fingerprint `fp` that leaves this bucket for its other
         * candidate (or skips it) sets this bit first. Lookups that miss here only need the
         * other bucket if it is set. Bits are only cleared by `clear()`.
         */
        __host__ __device__ static constexpr size_t overflowWord(TagType fp) {
            return (fp & (bucketSize - 1)) / tagsPerWord;
        }

        __host__ __device__ static constexpr WordType overflowBit(TagType fp) {
            return WordType(1) << ((fp & (tagsPerWord - 1)) * bitsPerTag + bitsPerTag - 1);
        }

        /**
         * @brief Checks an array of loaded words for a matching tag using SWAR.
         */
        template <size_t N>
        __device__ __forceinline__ static bool
        checkWords(const WordType (&loaded)[N], WordType replicatedTag) {
            _Pragma("unroll")
            for (size_t j = 0; j < N; ++j) {
                if (fpZeroMask(loaded[j] ^ replicatedTag)) {
                    return true;
                }
            }
            return false;
        }

        /// Ors this tag's overflow bit from the words loaded at [startIdx, startIdx + N).
        template <size_t N>
        __device__ __forceinline__ static void
        noteOverflow(const WordType (&loaded)[N], size_t startIdx, TagType tag, bool* overflow) {
            if (overflow == nullptr) {
                return;
            }
            const size_t word = overflowWord(tag);
            _Pragma("unroll")
            for (size_t j = 0; j < N; ++j) {
                if (startIdx + j == word && (loaded[j] & overflowBit(tag))) {
                    *overflow = true;
                }
            }
        }

        /**
         * @brief Loads words using 128-bit vectorized loads into a fixed-size array.
         */
        template <size_t N>
        __device__ __forceinline__ void load128Bit(size_t startIdx, WordType (&out)[N]) const {
            static_assert(N == 2 || N == 4, "128-bit loads support 2 or 4 words");
            if constexpr (sizeof(WordType) == 4) {
                auto vec = __ldg(reinterpret_cast<const uint4*>(&packedTags[startIdx]));
                out[0] = vec.x;
                out[1] = vec.y;
                out[2] = vec.z;
                out[3] = vec.w;
            } else {
                auto vec = __ldg(reinterpret_cast<const ulonglong2*>(&packedTags[startIdx]));
                out[0] = vec.x;
                out[1] = vec.y;
            }
        }

        /**
         * @brief Checks if a tag is present in the bucket using vectorized loads.
         *        Automatically selects the best load width based on bucket size and
         * architecture.
         *
         * @param overflow If non-null and the tag is absent, set to true when the tag's overflow
         * bit is set, i.e. when the other candidate bucket must be checked too.
         */
        __device__ bool contains(TagType tag, bool* overflow = nullptr) const {
            const WordType replicatedTag = detail::replicateTag<TagType, WordType>(tag);

            // Scalar path
            if constexpr (wordCount == 1) {
                const WordType loaded[1] = {reinterpret_cast<const WordType&>(packedTags[0])};
                noteOverflow(loaded, 0, tag, overflow);
                return checkWords(loaded, replicatedTag);
            }

            const uint32_t startSlot = tag & (bucketSize - 1);
            const size_t startWordIdx = startSlot / tagsPerWord;

#if __CUDA_ARCH__ >= 1000 && !defined(CUCKOO_FILTER_DISABLE_256BIT_LOADS)
            // 256-bit load path
            constexpr size_t wordsPerLoad256 = (sizeof(WordType) == 4) ? 8 : 4;
            if constexpr (wordCount >= wordsPerLoad256) {
                constexpr size_t alignMask = wordsPerLoad256 - 1;
                const size_t startAlignedIdx = startWordIdx & ~alignMask;

                WordType loaded[wordsPerLoad256];
                for (size_t i = 0; i < wordCount / wordsPerLoad256; i++) {
                    const size_t idx = (startAlignedIdx + i * wordsPerLoad256) & (wordCount - 1);
                    detail::load256BitGlobalNC(
                        reinterpret_cast<const WordType*>(&packedTags[idx]), loaded
                    );
                    if (checkWords(loaded, replicatedTag)) {
                        return true;
                    }
                    noteOverflow(loaded, idx, tag, overflow);
                }
                return false;
            }
#endif
            // 128-bit load path
            constexpr size_t wordsPerLoad128 = (sizeof(WordType) == 4) ? 4 : 2;
            if constexpr (wordCount >= wordsPerLoad128) {
                constexpr size_t alignMask = wordsPerLoad128 - 1;
                const size_t startAlignedIdx = startWordIdx & ~alignMask;

                WordType loaded[wordsPerLoad128];
                for (size_t i = 0; i < wordCount / wordsPerLoad128; i++) {
                    const size_t idx = (startAlignedIdx + i * wordsPerLoad128) & (wordCount - 1);
                    load128Bit(idx, loaded);
                    if (checkWords(loaded, replicatedTag)) {
                        return true;
                    }
                    noteOverflow(loaded, idx, tag, overflow);
                }
                return false;
            }

            return false;
        }
    };

    size_t numBuckets;  ///< Number of buckets in the filter
    Bucket* d_buckets;  ///< Pointer to the device memory for the buckets
    cuda::std::atomic<size_t>*
        d_numOccupied{};  ///< Pointer to the device memory for the occupancy counter

#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
    cuda::std::atomic<size_t>*
        d_deleteCasAttempts{};  ///< Pointer to device memory for the delete CAS attempt counter
    cuda::std::atomic<size_t>*
        d_deleteCasFailures{};  ///< Pointer to device memory for the failed delete CAS counter
#endif

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    cuda::std::atomic<size_t>* d_numEvictions{};  ///< Pointer to device memory for eviction counter
#endif

    size_t h_numOccupied = 0;   ///< Number of occupied buckets in the filter
    uint32_t h_overflow = 0;    ///< Host copy of the region-overflow flag
    size_t l2CacheBytes = 0;    ///< L2 size of the device the filter lives on
    void* d_scratch{};          ///< Reusable device scratch for batch preprocessing
    size_t d_scratchBytes = 0;  ///< Size of d_scratch in bytes
    std::mutex scratchMutex;    ///< Serialises host calls that use d_scratch
    /// Device-memory copy of this object, passed to kernels instead of `this` so that device
    /// code reads `d_buckets` and `numBuckets` from GPU memory, not from host memory over the
    /// bus. Valid because every field device code reads is fixed once the constructor finishes.
    Filter* d_self{};

    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return AltBucketPolicy::hash64(key);
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType, TagType>
    getCandidateBucketsAndFPs(const T& key, size_t numBuckets) {
        return AltBucketPolicy::getCandidateBucketsAndFPs(key, numBuckets);
    }

    /**
     * @brief Computes the alternate bucket for a fingerprint.
     */
    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return AltBucketPolicy::getAlternateBucket(bucket, fp, numBuckets);
    }

    /**
     * @brief Computes alternate bucket AND updated fingerprint for choice bit policies.
     * For non-choice-bit policies, returns the original fingerprint unchanged.
     */
    static __host__ __device__ cuda::std::tuple<size_t, TagType>
    getAlternateBucketWithNewFp(size_t bucket, TagType fp, size_t numBuckets) {
        if constexpr (AltBucketPolicy::usesChoiceBit) {
            return AltBucketPolicy::getAlternateBucketWithNewFp(bucket, fp, numBuckets);
        } else {
            return {AltBucketPolicy::getAlternateBucket(bucket, fp, numBuckets), fp};
        }
    }

    /**
     * @brief The number of buckets is enforced to be a power of two in order to allow for
     * efficient modulo on the bucket indices
     */
    static size_t calculateNumBuckets(size_t capacity) {
        return AltBucketPolicy::calculateNumBuckets(capacity);
    }

    Filter(const Filter&) = delete;
    Filter& operator=(const Filter&) = delete;

    /**
     * @brief Constructs a new Cuckoo Filter.
     *
     * Allocates device memory for buckets and occupancy counters.
     *
     * @param capacity Desired capacity (number of items) for the filter.
     */
    explicit Filter(size_t capacity) : numBuckets(calculateNumBuckets(capacity)) {
        CUCKOO_CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUCKOO_CUDA_CALL(cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>)));
#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
        CUCKOO_CUDA_CALL(cudaMalloc(&d_deleteCasAttempts, sizeof(cuda::std::atomic<size_t>)));
        CUCKOO_CUDA_CALL(cudaMalloc(&d_deleteCasFailures, sizeof(cuda::std::atomic<size_t>)));
#endif
#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
        CUCKOO_CUDA_CALL(cudaMalloc(&d_numEvictions, sizeof(cuda::std::atomic<size_t>)));
#endif

        clear();

        int device = 0;
        int l2Bytes = 0;
        CUCKOO_CUDA_CALL(cudaGetDevice(&device));
        CUCKOO_CUDA_CALL(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device));
        l2CacheBytes = static_cast<size_t>(l2Bytes);
        setSortedSharedMemory<detail::RegionOp::Contains>();
        setSortedSharedMemory<detail::RegionOp::Insert>();
        setSortedSharedMemory<detail::RegionOp::Delete>();
        CUCKOO_CUDA_CALL(cudaMalloc(&d_self, sizeof(Filter)));
        CUCKOO_CUDA_CALL(cudaMemcpy(d_self, this, sizeof(Filter), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Destroys the Cuckoo Filter.
     *
     * Frees allocated device memory.
     */
    ~Filter() {
        if (d_buckets) {
            CUCKOO_CUDA_CALL(cudaFree(d_buckets));
        }
        if (d_scratch) {
            CUCKOO_CUDA_CALL(cudaFree(d_scratch));
        }
        if (d_self) {
            CUCKOO_CUDA_CALL(cudaFree(d_self));
        }
        if (d_numOccupied) {
            CUCKOO_CUDA_CALL(cudaFree(d_numOccupied));
        }
#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
        if (d_deleteCasAttempts) {
            CUCKOO_CUDA_CALL(cudaFree(d_deleteCasAttempts));
        }
        if (d_deleteCasFailures) {
            CUCKOO_CUDA_CALL(cudaFree(d_deleteCasFailures));
        }
#endif
#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
        if (d_numEvictions) {
            CUCKOO_CUDA_CALL(cudaFree(d_numEvictions));
        }
#endif
    }

    /**
     * @brief Inserts a batch of keys into the filter.
     *
     * @param d_keys Pointer to device memory containing keys to insert.
     * @param n Number of keys to insert.
     * @param d_output Optional pointer to an output array indicating the success of each key
     * insertion.
     * @param stream CUDA stream to use for the operation.
     * @return size_t Total number of occupied slots after insertion.
     */
    size_t insertMany(
        const T* d_keys,
        const size_t n,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        if (usePartitioning(n) &&
            applyByRegion<detail::RegionOp::Insert>(d_keys, n, d_output, stream)) {
            return syncOccupiedSlots(stream);
        }

        size_t numBlocks = SDIV(n, blockSize);
        detail::insertKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, d_self, nullptr);

        return syncOccupiedSlots(stream);
    }

    /// Batch grouped by primary-bucket region, plus what is needed to restore input order.
    struct RegionPartition {
        RegionTagType* tags;             ///< Region-local tags, grouped by region
        uint32_t* counts;                ///< Occupied slots per region
        detail::KeyPlacement placement;  ///< Where each input key went (optional)
        uint32_t* results;               ///< Result bitmap, one bit per slot (optional)
        detail::RegionLayout layout;

        [[nodiscard]] size_t slots() const {
            return static_cast<size_t>(layout.numRegions) * layout.capacity;
        }
    };

    /**
     * @brief log2 of the number of regions batches are grouped into: at least 256 regions, and
     * enough that a bucket's index below its region fits next to its fingerprint in a
     * RegionTagType.
     */
    [[nodiscard]] size_t regionBits() const {
        constexpr size_t maxLocalBits = sizeof(RegionTagType) * 8 - bitsPerTag;
        const size_t bucketBits = bucketIndexBits();
        return std::max<size_t>(
            std::min<size_t>(8, bucketBits),
            bucketBits > maxLocalBits ? bucketBits - maxLocalBits : 0
        );
    }

    [[nodiscard]] detail::RegionLayout regionLayout(size_t n) const {
        const size_t bits = regionBits();
        const size_t numRegions = size_t{1} << bits;
        // Hashing spreads keys evenly over regions, so 25% headroom plus one scatter tile absorbs
        // normal variance; heavily skewed batches overflow and take the unpartitioned path.
        constexpr size_t tile = blockSize * detail::partitionItemsPerThread;
        const size_t capacity = SDIV(SDIV(n, numRegions) * 5 / 4 + tile, 32) * 32;
        return {
            static_cast<uint32_t>(bucketIndexBits() - bits),
            static_cast<uint32_t>(numRegions),
            static_cast<uint32_t>(capacity),
        };
    }

    /**
     * @brief Whether a batch should be grouped by primary-bucket region before processing.
     *
     * For filters larger than L2, random keys turn almost every bucket access into a DRAM access
     * and every update into a write-back. Grouping keys by the top bits of their primary bucket
     * makes consecutive work hit a small region of the table that stays in L2 (the alternate
     * bucket is in the same 2^16-bucket block), so table traffic drops to roughly one pass. The
     * grouping costs a few streaming passes over the batch, far less than the traffic it removes.
     */
    [[nodiscard]] bool usePartitioning(size_t n) const {
        // Region decoding splits the bucket index into fixed-width bit fields.
        if (!std::has_single_bit(numBuckets) || sizeInBytes() <= l2CacheBytes ||
            n > UINT32_MAX / 2 || (size_t{1} << regionBits()) > detail::maxRegions) {
            return false;
        }
        const auto layout = regionLayout(n);
        return static_cast<size_t>(layout.numRegions) * layout.capacity <= UINT32_MAX;
    }

    template <detail::RegionOp Op>
    static void setSortedSharedMemory() {
        CUCKOO_CUDA_CALL(cudaFuncSetAttribute(
            detail::regionSortedKernel<Config, Op>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(detail::sortedSharedBytes<RegionTagType>)
        ));
    }

    /**
     * @brief Groups the batch by region and applies `Op` to it, writing per-key results to
     * `d_output` if non-null. Returns false, having changed nothing, if a region overflowed.
     */
    template <detail::RegionOp Op>
    bool applyByRegion(const T* d_keys, size_t n, bool* d_output, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(scratchMutex);
        const auto part = partitionByRegion(d_keys, n, d_output != nullptr, stream);
        // The kernels below read the overflow flag themselves and do nothing if it is set, so the
        // host checks it only once, after the stream finishes.
        const uint32_t* overflow = part.counts + part.layout.numRegions;
        constexpr uint32_t chunk = detail::sortedChunk<RegionTagType>;
        detail::regionSortedKernel<Config, Op>
            <<<part.layout.numRegions * SDIV(part.layout.capacity, chunk),
               detail::sortedThreads,
               detail::sortedSharedBytes<RegionTagType>,
               stream>>>(part.tags, part.counts, part.layout, part.results, d_self);
        if (d_output != nullptr) {
            constexpr size_t tile = blockSize * detail::partitionItemsPerThread;
            const uint32_t numRegions = part.layout.numRegions;
            detail::gatherBitmapKernel<Config>
                <<<SDIV(n, tile),
                   tile / detail::gatherItemsPerThread,
                   numRegions * sizeof(uint32_t),
                   stream>>>(part.results, part.placement, numRegions, overflow, d_output, n);
        }
        CUCKOO_CUDA_CALL(
            cudaMemcpyAsync(&h_overflow, overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream)
        );
        // The partition lives in scratch, which the next batch may reuse once the lock is gone.
        CUCKOO_CUDA_CALL(cudaStreamSynchronize(stream));
        return h_overflow == 0;
    }

    /**
     * @brief Hashes `d_keys` and scatters them into fixed-capacity regions by primary bucket.
     *
     * With `withResults`, also records where each key went and allocates the result bitmap. If a
     * region overflowed, the scatter sets `counts[layout.numRegions]`, and kernels consuming the
     * partition must check it. Results live in the filter's scratch buffer, so the caller must
     * hold scratchMutex until the stream has consumed them.
     */
    RegionPartition
    partitionByRegion(const T* d_keys, size_t n, bool withResults, cudaStream_t stream) {
        const auto layout = regionLayout(n);
        const size_t slots = static_cast<size_t>(layout.numRegions) * layout.capacity;

        auto aligned = [](size_t bytes) { return SDIV(bytes, 256) * 256; };
        constexpr size_t tile = blockSize * detail::partitionItemsPerThread;
        const size_t tileBins = SDIV(n, tile) * layout.numRegions;
        const size_t tagBytes = aligned(slots * sizeof(RegionTagType));
        const size_t countBytes = aligned((layout.numRegions + 1) * sizeof(uint32_t));
        const size_t keyRunBytes = withResults ? aligned(n * sizeof(uint16_t)) : 0;
        const size_t runSlotBytes = withResults ? aligned(tileBins * sizeof(uint32_t)) : 0;
        const size_t resultBytes = withResults ? aligned(SDIV(slots, 32) * sizeof(uint32_t)) : 0;

        char* base = static_cast<char*>(
            scratch(tagBytes + countBytes + keyRunBytes + runSlotBytes + resultBytes, stream)
        );
        char* const keyRun = base + tagBytes + countBytes;
        char* const runSlot = keyRun + keyRunBytes;
        char* const results = runSlot + runSlotBytes;
        RegionPartition part{
            reinterpret_cast<RegionTagType*>(base),
            reinterpret_cast<uint32_t*>(base + tagBytes),
            withResults ? detail::KeyPlacement{
                              reinterpret_cast<uint16_t*>(keyRun),
                              reinterpret_cast<uint32_t*>(runSlot),
                          }
                        : detail::KeyPlacement{},
            withResults ? reinterpret_cast<uint32_t*>(results) : nullptr,
            layout,
        };
        uint32_t* d_overflow = part.counts + layout.numRegions;

        CUCKOO_CUDA_CALL(
            cudaMemsetAsync(part.counts, 0, (layout.numRegions + 1) * sizeof(uint32_t), stream)
        );
        detail::regionScatterKernel<Config>
            <<<SDIV(n, tile), blockSize, 3 * layout.numRegions * sizeof(uint32_t), stream>>>(
                d_keys, n, numBuckets, layout, part.counts, part.tags, part.placement, d_overflow
            );
        return part;
    }

    /**
     * @brief Returns a device scratch buffer of at least `bytes`, reused across calls.
     *
     * Kept for the filter's lifetime because allocating gigabytes per batch costs more than the
     * batch itself. Growing waits for `stream` so in-flight users of the old buffer finish first.
     */
    void* scratch(size_t bytes, cudaStream_t stream) {
        if (bytes > d_scratchBytes) {
            CUCKOO_CUDA_CALL(cudaStreamSynchronize(stream));
            if (d_scratch) {
                CUCKOO_CUDA_CALL(cudaFree(d_scratch));
                d_scratch = nullptr;
                d_scratchBytes = 0;
            }
            CUCKOO_CUDA_CALL(cudaMalloc(&d_scratch, bytes));
            d_scratchBytes = bytes;
        }
        return d_scratch;
    }

    /// log2(numBuckets); numBuckets is always a power of two.
    [[nodiscard]] size_t bucketIndexBits() const {
        return static_cast<size_t>(std::countr_zero(numBuckets));
    }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    /**
     * @brief Inserts a batch of keys and records per-attempt eviction counts.
     *
     * @param d_keys Pointer to device memory containing keys to insert.
     * @param n Number of keys to insert.
     * @param d_evictionAttempts Device output array of size n. Each entry stores the number of
     * evictions performed by the corresponding insertion attempt.
     * @param d_output Optional pointer to an output array indicating per-key insertion success.
     * @param stream CUDA stream to use for the operation.
     * @return size_t Total number of occupied slots after insertion.
     */
    size_t insertManyWithEvictionCounts(
        const T* d_keys,
        const size_t n,
        uint32_t* d_evictionAttempts,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        size_t numBlocks = SDIV(n, blockSize);
        detail::insertKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, d_self, d_evictionAttempts);

        return syncOccupiedSlots(stream);
    }
#endif

    /**
     * @brief This pre-sorts the input keys based on the primary bucket index to allow for
     * coalesced memory access when you later insert them into the filter.
     *
     * @param d_keys Pointer to device memory array of keys to insert
     * @param n Number of keys to insert
     * @param d_output Optional pointer to an output array indicating the success of each key
     * insertion.
     * @param stream CUDA stream to use for the operation.
     * @return size_t Updated number of occupied slots in the filter
     */
    size_t insertManySorted(
        const T* d_keys,
        const size_t n,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        PackedTagType* d_packedTags;

        CUCKOO_CUDA_CALL(cudaMallocAsync(&d_packedTags, n * sizeof(PackedTagType), stream));

        size_t numBlocks = SDIV(n, blockSize);

        detail::computePackedTagsKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_packedTags, n, numBuckets);

        void* d_tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage,
            tempStorageBytes,
            d_packedTags,
            d_packedTags,
            n,
            0,
            sizeof(PackedTagType) * 8,
            stream
        );

        CUCKOO_CUDA_CALL(cudaMallocAsync(&d_tempStorage, tempStorageBytes, stream));

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage,
            tempStorageBytes,
            d_packedTags,
            d_packedTags,
            n,
            0,
            sizeof(PackedTagType) * 8,
            stream
        );

        CUCKOO_CUDA_CALL(cudaFreeAsync(d_tempStorage, stream));

        detail::insertKernelSorted<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_packedTags, d_output, n, d_self, nullptr);

        CUCKOO_CUDA_CALL(cudaFreeAsync(d_packedTags, stream));
        return syncOccupiedSlots(stream);
    }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    /**
     * @brief Inserts a pre-sorted batch of keys and records per-attempt eviction counts.
     *
     * @param d_keys Pointer to device memory containing keys to insert.
     * @param n Number of keys to insert.
     * @param d_evictionAttempts Device output array of size n. Each entry stores the number of
     * evictions performed by the corresponding insertion attempt.
     * @param d_output Optional pointer to an output array indicating per-key insertion success.
     * @param stream CUDA stream to use for the operation.
     * @return size_t Updated number of occupied slots in the filter.
     */
    size_t insertManySortedWithEvictionCounts(
        const T* d_keys,
        const size_t n,
        uint32_t* d_evictionAttempts,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        PackedTagType* d_packedTags;

        CUCKOO_CUDA_CALL(cudaMallocAsync(&d_packedTags, n * sizeof(PackedTagType), stream));

        size_t numBlocks = SDIV(n, blockSize);

        detail::computePackedTagsKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_packedTags, n, numBuckets);

        void* d_tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage,
            tempStorageBytes,
            d_packedTags,
            d_packedTags,
            n,
            0,
            sizeof(PackedTagType) * 8,
            stream
        );

        CUCKOO_CUDA_CALL(cudaMallocAsync(&d_tempStorage, tempStorageBytes, stream));

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage,
            tempStorageBytes,
            d_packedTags,
            d_packedTags,
            n,
            0,
            sizeof(PackedTagType) * 8,
            stream
        );

        CUCKOO_CUDA_CALL(cudaFreeAsync(d_tempStorage, stream));

        detail::insertKernelSorted<Config><<<numBlocks, blockSize, 0, stream>>>(
            d_packedTags, d_output, n, d_self, d_evictionAttempts
        );

        CUCKOO_CUDA_CALL(cudaFreeAsync(d_packedTags, stream));
        return syncOccupiedSlots(stream);
    }
#endif

    /**
     * @brief Checks for the existence of a batch of keys.
     *
     * @param d_keys Pointer to device memory containing keys to check.
     * @param n Number of keys to check.
     * @param d_output Pointer to device memory to store results (true/false).
     * @param stream CUDA stream to use for the operation.
     */
    void containsMany(const T* d_keys, const size_t n, bool* d_output, cudaStream_t stream = {}) {
        size_t numBlocks = SDIV(n, blockSize);
        if (usePartitioning(n) &&
            applyByRegion<detail::RegionOp::Contains>(d_keys, n, d_output, stream)) {
            return;
        }

        detail::containsKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, d_self);

        CUCKOO_CUDA_CALL(cudaStreamSynchronize(stream));
    }

    /**
     * @brief Tries to remove a set of keys from the filter. If the key is not present, it is
     * ignored.
     *
     * @param d_keys Pointer to the array of keys to remove
     * @param n Number of keys to remove
     * @param d_output Optional pointer to an output array indicating the success of each key
     * removal
     * @param stream CUDA stream to use for the operation
     * @return size_t Updated number of occupied slots in the filter
     */
    size_t deleteMany(
        const T* d_keys,
        const size_t n,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        if (n == 0) {
            return syncOccupiedSlots(stream);
        }
        size_t numBlocks = SDIV(n, blockSize);
        if (n >= numBuckets && usePartitioning(n) &&
            applyByRegion<detail::RegionOp::Delete>(d_keys, n, d_output, stream)) {
            return syncOccupiedSlots(stream);
        }
        detail::deleteKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, d_self);

        return syncOccupiedSlots(stream);
    }

    /**
     * @brief Inserts keys from a Thrust device vector.
     * @param d_keys Vector of keys to insert.
     * @param d_output Vector to store results (bool). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return insertMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    /**
     * @brief Inserts keys from a Thrust device vector (uint8_t output).
     * @param d_keys Vector of keys to insert.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return insertMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    /**
     * @brief Inserts keys from a Thrust device vector without outputting results.
     * @param d_keys Vector of keys to insert.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertMany(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return insertMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr, stream);
    }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    /**
     * @brief Inserts keys from a Thrust device vector and records per-attempt eviction counts.
     * @param d_keys Vector of keys to insert.
     * @param d_evictionAttempts Vector to store per-key eviction counts. Resized if necessary.
     * @param d_output Optional vector to store per-key insertion success. Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertManyWithEvictionCounts(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint32_t>& d_evictionAttempts,
        thrust::device_vector<uint8_t>* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        if (d_evictionAttempts.size() != d_keys.size()) {
            d_evictionAttempts.resize(d_keys.size());
        }

        bool* d_outputPtr = nullptr;
        if (d_output != nullptr) {
            if (d_output->size() != d_keys.size()) {
                d_output->resize(d_keys.size());
            }
            d_outputPtr = reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output->data()));
        }

        return insertManyWithEvictionCounts(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_evictionAttempts.data()),
            d_outputPtr,
            stream
        );
    }
#endif

    /**
     * @brief Inserts keys from a Thrust device vector, sorting them first.
     * @param d_keys Vector of keys to insert.
     * @param d_output Vector to store results (bool). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertManySorted(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return insertManySorted(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    /**
     * @brief Inserts keys from a Thrust device vector, sorting them first (uint8_t output).
     * @param d_keys Vector of keys to insert.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertManySorted(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return insertManySorted(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    /**
     * @brief Inserts keys from a Thrust device vector, sorting them first, without outputting
     * results.
     * @param d_keys Vector of keys to insert.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertManySorted(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return insertManySorted(
            thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr, stream
        );
    }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    /**
     * @brief Inserts keys from a Thrust device vector, sorting first, and records per-attempt
     * eviction counts.
     * @param d_keys Vector of keys to insert.
     * @param d_evictionAttempts Vector to store per-key eviction counts. Resized if necessary.
     * @param d_output Optional vector to store per-key insertion success. Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t insertManySortedWithEvictionCounts(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint32_t>& d_evictionAttempts,
        thrust::device_vector<uint8_t>* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        if (d_evictionAttempts.size() != d_keys.size()) {
            d_evictionAttempts.resize(d_keys.size());
        }

        bool* d_outputPtr = nullptr;
        if (d_output != nullptr) {
            if (d_output->size() != d_keys.size()) {
                d_output->resize(d_keys.size());
            }
            d_outputPtr = reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output->data()));
        }

        return insertManySortedWithEvictionCounts(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_evictionAttempts.data()),
            d_outputPtr,
            stream
        );
    }
#endif

    /**
     * @brief Checks for existence of keys in a Thrust device vector.
     * @param d_keys Vector of keys to check.
     * @param d_output Vector to store results (bool). Resized if necessary.
     * @param stream CUDA stream.
     */
    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    /**
     * @brief Checks for existence of keys in a Thrust device vector (uint8_t output).
     * @param d_keys Vector of keys to check.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     * @param stream CUDA stream.
     */
    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    /**
     * @brief Deletes keys in a Thrust device vector.
     * @param d_keys Vector of keys to delete.
     * @param d_output Vector to store results (bool). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    /**
     * @brief Deletes keys in a Thrust device vector (uint8_t output).
     * @param d_keys Vector of keys to delete.
     * @param d_output Vector to store results (uint8_t). Resized if necessary.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    /**
     * @brief Deletes keys in a Thrust device vector without outputting results.
     * @param d_keys Vector of keys to delete.
     * @param stream CUDA stream.
     * @return size_t Total number of occupied slots.
     */
    size_t deleteMany(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return deleteMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr, stream);
    }

    /**
     * @brief Clears the filter, removing all items.
     */
    void clear() {
        CUCKOO_CUDA_CALL(cudaMemset(d_buckets, 0, numBuckets * sizeof(Bucket)));
        CUCKOO_CUDA_CALL(cudaMemset(d_numOccupied, 0, sizeof(cuda::std::atomic<size_t>)));
#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
        CUCKOO_CUDA_CALL(cudaMemset(d_deleteCasAttempts, 0, sizeof(cuda::std::atomic<size_t>)));
        CUCKOO_CUDA_CALL(cudaMemset(d_deleteCasFailures, 0, sizeof(cuda::std::atomic<size_t>)));
#endif
#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
        CUCKOO_CUDA_CALL(cudaMemset(d_numEvictions, 0, sizeof(cuda::std::atomic<size_t>)));
#endif
        h_numOccupied = 0;
    }

    /**
     * @brief Calculates the current load factor of the filter.
     * @return float Load factor (occupied slots / total capacity).
     */
    [[nodiscard]] float loadFactor() {
        return static_cast<float>(occupiedSlots()) / (numBuckets * bucketSize);
    }

    /**
     * @brief Returns the total number of occupied slots.
     *
     * Retrieves the value from the device counter.
     *
     * @return size_t Number of occupied slots.
     */
    size_t occupiedSlots() {
        CUCKOO_CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return h_numOccupied;
    }

    /**
     * @brief Waits for `stream` and returns the occupied-slot count, reading the counter back in
     * the same stream so only one host round trip is needed.
     */
    size_t syncOccupiedSlots(cudaStream_t stream) {
        CUCKOO_CUDA_CALL(cudaMemcpyAsync(
            &h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost, stream
        ));
        CUCKOO_CUDA_CALL(cudaStreamSynchronize(stream));
        return h_numOccupied;
    }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
    /**
     * @brief Returns the total number of evictions performed.
     *
     * Only available when CUCKOO_FILTER_COUNT_EVICTIONS is defined.
     *
     * @return size_t Number of evictions.
     */
    size_t evictionCount() {
        size_t count;
        CUCKOO_CUDA_CALL(
            cudaMemcpy(&count, d_numEvictions, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return count;
    }

    /**
     * @brief Resets the eviction counter to zero.
     */
    void resetEvictionCount() {
        CUCKOO_CUDA_CALL(cudaMemset(d_numEvictions, 0, sizeof(cuda::std::atomic<size_t>)));
    }
#endif

#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
    /**
     * @brief Returns the total number of delete CAS attempts (compare_exchange calls).
     *
     * Only available when CUCKOO_FILTER_COUNT_DELETE_CAS is defined.
     *
     * @return size_t Number of delete CAS attempts.
     */
    size_t deleteCasAttempts() {
        size_t count;
        CUCKOO_CUDA_CALL(
            cudaMemcpy(&count, d_deleteCasAttempts, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return count;
    }

    /**
     * @brief Returns the total number of failed delete CAS attempts.
     *
     * Only available when CUCKOO_FILTER_COUNT_DELETE_CAS is defined.
     *
     * @return size_t Number of failed delete CAS attempts.
     */
    size_t deleteCasFailures() {
        size_t count;
        CUCKOO_CUDA_CALL(
            cudaMemcpy(&count, d_deleteCasFailures, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return count;
    }

    /**
     * @brief Resets the delete CAS counters to zero.
     */
    void resetDeleteCasCounters() {
        CUCKOO_CUDA_CALL(cudaMemset(d_deleteCasAttempts, 0, sizeof(cuda::std::atomic<size_t>)));
        CUCKOO_CUDA_CALL(cudaMemset(d_deleteCasFailures, 0, sizeof(cuda::std::atomic<size_t>)));
    }
#endif

    /**
     * @brief Returns the total capacity of the filter.
     * @return size_t Total number of slots.
     */
    size_t capacity() {
        return numBuckets * bucketSize;
    }

    /**
     * @brief Returns the number of buckets in the filter.
     * @return size_t Number of buckets.
     */
    [[nodiscard]] size_t getNumBuckets() const {
        return numBuckets;
    }

    /**
     * @brief Returns the size of the filter in bytes.
     * @return size_t Size in bytes.
     */
    [[nodiscard]] size_t sizeInBytes() const {
        return numBuckets * sizeof(Bucket);
    }

    /**
     * @brief Counts occupied slots by iterating over all buckets on the host.
     *
     * This is a slow operation used for verification/debugging.
     *
     * @return size_t Actual number of occupied slots.
     */
    size_t countOccupiedSlots() {
        std::vector<Bucket> h_buckets(numBuckets);

        CUCKOO_CUDA_CALL(cudaMemcpy(
            h_buckets.data(), d_buckets, numBuckets * sizeof(Bucket), cudaMemcpyDeviceToHost
        ));

        size_t occupiedCount = 0;

        for (size_t bucketIdx = 0; bucketIdx < numBuckets; ++bucketIdx) {
            const Bucket& bucket = h_buckets[bucketIdx];

            for (size_t atomicIdx = 0; atomicIdx < Bucket::wordCount; ++atomicIdx) {
                uint64_t packed = reinterpret_cast<const uint64_t&>(bucket.packedTags[atomicIdx]);

                for (size_t tagIdx = 0; tagIdx < Bucket::tagsPerWord; ++tagIdx) {
                    auto tag = bucket.extractTag(packed, tagIdx);

                    if (tag != EMPTY) {
                        occupiedCount++;
                    }
                }
            }
        }

        return occupiedCount;
    }

    template <typename TryWord>
    __device__ __forceinline__ static bool scanBucket(
        Bucket& bucket,
        size_t startWord,
        typename Bucket::WordType replicated,
        TryWord tryWord
    ) {
        using WordType = typename Bucket::WordType;
#if __CUDA_ARCH__ >= 1000 && !defined(CUCKOO_FILTER_DISABLE_256BIT_LOADS)
        constexpr bool wide = sizeof(Bucket) == 32 && sizeof(WordType) == 8;
#else
        constexpr bool wide = false;
#endif
        if constexpr (wide) {
            // Snapshot one sector. Every write still validates its word by CAS.
            WordType words[Bucket::wordCount];
            asm volatile("ld.relaxed.gpu.global.v4.b64 {%0, %1, %2, %3}, [%4];"
                         : "=l"(words[0]), "=l"(words[1]), "=l"(words[2]), "=l"(words[3])
                         : "l"(&bucket.packedTags[0]));
            uint32_t matches = 0;
            _Pragma("unroll")
            for (uint32_t k = 0; k < Bucket::wordCount; ++k) {
                matches |= uint32_t(Bucket::fpZeroMask(words[k] ^ replicated) != 0) << k;
            }
            constexpr uint32_t wordMask = Bucket::wordCount - 1;
            // Rotate the mask to keep the tag-dependent probe order.
            matches = ((matches | (matches << Bucket::wordCount)) >> startWord) &
                      ((1u << Bucket::wordCount) - 1);
            while (matches != 0) {
                const auto step = static_cast<uint32_t>(__ffs(matches) - 1);
                const auto idx = (startWord + step) & wordMask;
                WordType expected = words[0];  // Constant indices keep the snapshot in registers.
                _Pragma("unroll")
                for (uint32_t k = 1; k < Bucket::wordCount; ++k) {
                    expected = idx == k ? words[k] : expected;
                }
                if (tryWord(idx, expected)) {
                    return true;
                }
                matches &= matches - 1;
            }
        } else {
            for (size_t i = 0; i < Bucket::wordCount; ++i) {
                const size_t idx = (startWord + i) & (Bucket::wordCount - 1);
                if (tryWord(idx, bucket.word(idx).load(cuda::memory_order_relaxed))) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Removes one matching fingerprint without updating the occupancy counter. */
    __device__ bool tryRemoveAtBucket(size_t bucketIdx, TagType tag) {
        Bucket& bucket = d_buckets[bucketIdx];

        const uint32_t startSlot = tag & (bucketSize - 1);
        const size_t startWord = startSlot / Bucket::tagsPerWord;

        using WordType = typename Bucket::WordType;

        const auto replicated = detail::replicateTag<TagType, WordType>(tag);
        auto tryWord = [&](size_t currIdx, WordType expected) {
            while (const auto matchMask = Bucket::fpZeroMask(expected ^ replicated)) {
                // Find position of first matching tag
                int bitPos;
                if constexpr (sizeof(WordType) == 4) {
                    bitPos = __ffs(static_cast<int>(matchMask)) - 1;
                } else {
                    bitPos = __ffsll(static_cast<long long>(matchMask)) - 1;
                }
                size_t tagIdx = bitPos / bitsPerTag;

                auto desired = bucket.replaceTag(expected, tagIdx, EMPTY);

#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
                d_deleteCasAttempts->fetch_add(1, cuda::memory_order_relaxed);
#endif
                bool casSuccess = bucket.word(currIdx).compare_exchange_weak(
                    expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
                );
#ifdef CUCKOO_FILTER_COUNT_DELETE_CAS
                if (!casSuccess) {
                    d_deleteCasFailures->fetch_add(1, cuda::memory_order_relaxed);
                }
#endif
                if (casSuccess) {
                    return true;
                }
                // CAS failed, retry with updated expected value
            }
            return false;
        };
        return scanBucket(bucket, startWord, replicated, tryWord);
    }

    /**
     * @brief Attempts to insert a tag into a specific bucket.
     *
     * Scans the bucket for an empty slot and attempts to atomically place the tag.
     *
     * @param bucketIdx Index of the bucket.
     * @param tag Tag to insert.
     * @return true if insertion succeeded, false if the bucket is full.
     */
    __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
        Bucket& bucket = d_buckets[bucketIdx];
        const uint32_t startIdx = tag & (bucketSize - 1);
        const size_t startWord = startIdx / Bucket::tagsPerWord;

        using WordType = typename Bucket::WordType;
        auto tryWord = [&](size_t idx, WordType expected) {
            while (const WordType zeros = Bucket::fpZeroMask(expected)) {
                int bitPos;
                if constexpr (sizeof(WordType) == 4) {
                    bitPos = __ffs(static_cast<int>(zeros)) - 1;
                } else {
                    bitPos = __ffsll(static_cast<long long>(zeros)) - 1;
                }
                const auto desired = bucket.replaceTag(expected, bitPos / bitsPerTag, tag);
                if (bucket.word(idx).compare_exchange_strong(
                        expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
                    )) {
                    return true;
                }
            }
            return false;
        };

        return scanBucket(bucket, startWord, WordType{0}, tryWord);
    }

    /// Records in `bucketIdx` that a tag with fingerprint `fp` lives in its other bucket.
    __device__ void markOverflow(size_t bucketIdx, TagType fp) {
        d_buckets[bucketIdx].word(Bucket::overflowWord(fp)).fetchOr(Bucket::overflowBit(fp));
    }

    /**
     * @brief Inserts a fingerprint into the filter by evicting existing fingerprints. The
     * thread first picks a pseudo-random target to replace with the new fingerprint. Then it
     * tries to insert the evicted fingerprint into its alternate bucket. This process is
     * repeated until either a fingerprint is inserted into an empty slot or the maximum number
     * of evictions is reached.
     *
     * @param fp Fingerprint to insert
     * @param startBucket Index of the bucket to start the search from
     * @param evictionAttempts Optional pointer to a counter for eviction attempts
     * @return true if the insertion was successful, false otherwise
     */
    __device__ bool
    insertWithEvictionDFS(TagType fp, size_t startBucket, uint32_t* evictionAttempts = nullptr) {
        TagType currentFp = fp;
        size_t currentBucket = startBucket;

        for (size_t evictions = 0; evictions < maxEvictions; ++evictions) {
            auto evictSlot = (currentFp + (evictions + 1) * 0x9E3779B1UL) & (bucketSize - 1);

            size_t evictWord = evictSlot / Bucket::tagsPerWord;
            size_t evictTagIdx = evictSlot & (Bucket::tagsPerWord - 1);

            Bucket& bucket = d_buckets[currentBucket];
            auto expected = bucket.word(evictWord).load(cuda::memory_order_relaxed);
            typename Bucket::WordType desired;
            TagType evictedFp;

            do {
                evictedFp = bucket.extractTag(expected, evictTagIdx);
                desired = bucket.replaceTag(expected, evictTagIdx, currentFp);
            } while (!bucket.word(evictWord).compare_exchange_strong(
                expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
            ));

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
            d_numEvictions->fetch_add(1, cuda::memory_order_relaxed);
            if (evictionAttempts != nullptr) {
                (*evictionAttempts)++;
            }
#endif

            if (evictedFp != EMPTY) {
                markOverflow(currentBucket, evictedFp);
            }
            auto [altBucket, newFp] =
                getAlternateBucketWithNewFp(currentBucket, evictedFp, numBuckets);
            currentBucket = altBucket;
            currentFp = newFp;

            if (tryInsertAtBucket(currentBucket, currentFp)) {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Inserts a fingerprint using repeated shallow breadth-first attempts.
     *
     * Each round scans a handful of candidate eviction slots in the current bucket for one whose
     * alternate bucket has a free slot. The thread swaps itself into that slot and then places the
     * evicted tag, which it now holds, in the alternate bucket. If no candidate qualifies, it
     * evicts the last scanned slot. Either way, an evicted tag that cannot be placed continues the
     * chain from its alternate bucket. Tags only ever move by swap, so concurrent chains cannot
     * leave duplicate copies. This repeats until insertion succeeds or maxEvictions is reached.
     *
     * @param fp Fingerprint to insert
     * @param evictionAttempts Optional pointer to a counter for eviction attempts
     * @param startBucket Index of the bucket to start the search from
     * @return true if the insertion was successful, false otherwise
     */
    __device__ bool
    insertWithEvictionBFS(TagType fp, size_t startBucket, uint32_t* evictionAttempts = nullptr) {
        constexpr size_t numCandidates = std::max(1UL, bucketSize / 2);

        TagType currentFp = fp;
        size_t currentBucket = startBucket;

        size_t evictions = 0;
        while (evictions < maxEvictions) {
            Bucket& bucket = d_buckets[currentBucket];
            size_t evictWord = 0;
            size_t evictTagIdx = 0;
            bool lookahead = false;

            for (size_t i = 0; i < numCandidates && !lookahead; ++i) {
                size_t evictSlot = (currentFp + i * 0x9E3779B1UL + (evictions + 1) * 0x85EBCA77) &
                                   (bucketSize - 1);
                evictWord = evictSlot / Bucket::tagsPerWord;
                evictTagIdx = evictSlot & (Bucket::tagsPerWord - 1);

                auto packed = bucket.word(evictWord).load(cuda::memory_order_relaxed);
                TagType candidateFp = bucket.extractTag(packed, evictTagIdx);

                if (candidateFp == EMPTY) {
                    if (tryInsertAtBucket(currentBucket, currentFp)) {
                        return true;
                    }
                    continue;
                }

                auto [altBucket, altFp] =
                    getAlternateBucketWithNewFp(currentBucket, candidateFp, numBuckets);
                using WordType = typename Bucket::WordType;
                lookahead =
                    scanBucket(d_buckets[altBucket], 0, WordType{0}, [](size_t, WordType word) {
                        return Bucket::fpZeroMask(word) != 0;
                    });
            }

            // Swap our tag into the chosen slot; the displaced tag is now held by this thread.
            auto expected = bucket.word(evictWord).load(cuda::memory_order_relaxed);
            typename Bucket::WordType desired;
            TagType evictedFp;

            do {
                evictedFp = bucket.extractTag(expected, evictTagIdx);
                desired = bucket.replaceTag(expected, evictTagIdx, currentFp);
            } while (!bucket.word(evictWord).compare_exchange_strong(
                expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
            ));

            if (evictedFp == EMPTY) {
                return true;
            }

#ifdef CUCKOO_FILTER_COUNT_EVICTIONS
            d_numEvictions->fetch_add(1, cuda::memory_order_relaxed);
            if (evictionAttempts != nullptr) {
                (*evictionAttempts)++;
            }
#endif

            evictions++;
            markOverflow(currentBucket, evictedFp);
            auto [altBucket, altFp] =
                getAlternateBucketWithNewFp(currentBucket, evictedFp, numBuckets);
            if (tryInsertAtBucket(altBucket, altFp)) {
                return true;
            }
            currentBucket = altBucket;
            currentFp = altFp;
        }

        return false;
    }

    /**
     * @brief Inserts a single key into the filter.
     *
     * Computes candidate buckets and attempts insertion, performing eviction if necessary.
     *
     * @param key The key to insert.
     * @param evictionAttempts Optional pointer to a counter for eviction attempts
     * @return true if insertion succeeded, false if the filter is too full (max evictions
     * reached).
     */
    __device__ bool insert(const T& key, uint32_t* evictionAttempts = nullptr) {
        auto [i1, i2, fp1, fp2] = getCandidateBucketsAndFPs(key, numBuckets);
        return insertCandidates(i1, i2, fp1, fp2, evictionAttempts);
    }

    /**
     * @brief Inserts a fingerprint given both candidate buckets, evicting if necessary.
     *
     * fp1 is the fingerprint for bucket i1 and fp2 for bucket i2 (equal for non-choice-bit
     * policies).
     */
    __device__ bool insertCandidates(
        size_t i1,
        size_t i2,
        TagType fp1,
        TagType fp2,
        uint32_t* evictionAttempts = nullptr
    ) {
        return tryInsertAtBucket(i1, fp1) ||
               insertOutsidePrimary(i1, i2, fp1, fp2, evictionAttempts);
    }

    /**
     * @brief The rest of `insertCandidates` once bucket i1 had no room: place the fingerprint in
     * i2, evicting if necessary.
     */
    __device__ bool insertOutsidePrimary(
        size_t i1,
        size_t i2,
        TagType fp1,
        TagType fp2,
        uint32_t* evictionAttempts = nullptr
    ) {
        // Whichever way the key ends up outside i1, lookups must learn to check i2.
        markOverflow(i1, fp1);
        return tryInsertAtBucket(i2, fp2) || insertByEviction(i1, i2, fp1, fp2, evictionAttempts);
    }

    /// The last resort of `insertOutsidePrimary`, once neither bucket had room.
    __device__ bool insertByEviction(
        size_t i1,
        size_t i2,
        TagType fp1,
        TagType fp2,
        uint32_t* evictionAttempts = nullptr
    ) {
        // For eviction, use correct fingerprint for the starting bucket
        auto startBucket = (fp1 & 1) == 0 ? i1 : i2;
        TagType evictFp;

        if constexpr (AltBucketPolicy::usesChoiceBit) {
            evictFp = (fp1 & 1) == 0 ? fp1 : fp2;
        } else {
            evictFp = fp1;
        }

        if constexpr (Config::evictionPolicy == EvictionPolicy::BFS) {
            return insertWithEvictionBFS(evictFp, startBucket, evictionAttempts);
        } else if constexpr (Config::evictionPolicy == EvictionPolicy::DFS) {
            return insertWithEvictionDFS(evictFp, startBucket, evictionAttempts);
        } else {
            static_assert(
                Config::evictionPolicy == EvictionPolicy::DFS ||
                    Config::evictionPolicy == EvictionPolicy::BFS,
                "Unhandled eviction policy"
            );
        }
    }

    /**
     * @brief Checks if a key exists in the filter.
     *
     * @param key The key to check.
     * @return true if the key is found, false otherwise.
     */
    __device__ bool contains(const T& key) const {
        auto [i1, i2, fp1, fp2] = getCandidateBucketsAndFPs(key, numBuckets);

        // fp1 is for bucket i1, fp2 is for bucket i2
        // For non-choice-bit policies, fp1 == fp2
        bool overflow = false;
        return d_buckets[i1].contains(fp1, &overflow) || (overflow && d_buckets[i2].contains(fp2));
    }

    /**
     * @brief Removes a key from the filter.
     *
     * @param key The key to remove.
     * @return true if the key was found and removed, false otherwise.
     */
    __device__ bool remove(const T& key) {
        auto [i1, i2, fp1, fp2] = getCandidateBucketsAndFPs(key, numBuckets);

        // fp1 is for bucket i1, fp2 is for bucket i2
        // For non-choice-bit policies, fp1 == fp2
        return tryRemoveAtBucket(i1, fp1) || tryRemoveAtBucket(i2, fp2);
    }
};

namespace detail {

// Without a bound the inlined eviction code takes 54 registers, which leaves room for only four
// 256-thread blocks of the six that fill a 1536-thread SM.
template <typename Config>
__global__ void __launch_bounds__(Config::blockSize, 1536 / Config::blockSize) insertKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    Filter<Config>* filter,
    uint32_t* evictionAttempts
) {
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;

    if (idx < n) {
        uint32_t threadEvictions = 0;
        success = filter->insert(keys[idx], &threadEvictions);

        if (output != nullptr) {
            output[idx] = success;
        }

        if (evictionAttempts != nullptr) {
            evictionAttempts[idx] = threadEvictions;
        }
    }

    int32_t blockSuccessSum = BlockReduce(tempStorage).Sum(success);
    __syncthreads();

    if (threadIdx.x == 0) {
        if (blockSuccessSum > 0) {
            filter->d_numOccupied->fetch_add(blockSuccessSum, cuda::memory_order_relaxed);
        }
    }
}

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    Filter<Config>* filter
) {
    auto idx = globalThreadId();

    if (idx < n) {
        output[idx] = filter->contains(keys[idx]);
    }
}

template <typename Config>
__global__ void
deleteKernel(const typename Config::KeyType* keys, bool* output, size_t n, Filter<Config>* filter) {
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;
    if (idx < n) {
        success = filter->remove(keys[idx]);

        if (output != nullptr) {
            output[idx] = success;
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        filter->d_numOccupied->fetch_sub(blockSum, cuda::memory_order_relaxed);
    }
}

template <typename Config>
__global__ void computePackedTagsKernel(
    const typename Config::KeyType* keys,
    typename Filter<Config>::PackedTagType* packedTags,
    size_t n,
    size_t numBuckets
) {
    size_t idx = globalThreadId();

    if (idx >= n) {
        return;
    }

    using FilterType = Filter<Config>;
    using PackedTagType = typename FilterType::PackedTagType;
    constexpr size_t bitsPerTag = Config::bitsPerTag;

    typename Config::KeyType key = keys[idx];
    auto [i1, i2, fp1, fp2] = FilterType::getCandidateBucketsAndFPs(key, numBuckets);

    packedTags[idx] =
        (static_cast<PackedTagType>(i1) << bitsPerTag) | static_cast<PackedTagType>(fp1);
}

template <typename Config>
__global__ void insertKernelSorted(
    const typename Filter<Config>::PackedTagType* packedTags,
    bool* output,
    size_t n,
    Filter<Config>* filter,
    uint32_t* evictionAttempts
) {
    using BlockReduce = cub::BlockReduce<int, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    size_t idx = globalThreadId();

    using FilterType = Filter<Config>;
    using TagType = typename FilterType::TagType;
    using PackedTagType = typename FilterType::PackedTagType;

    constexpr size_t bitsPerTag = Config::bitsPerTag;
    constexpr TagType fpMask = (1ULL << bitsPerTag) - 1;

    int32_t success = 0;
    uint32_t threadEvictions = 0;
    if (idx < n) {
        PackedTagType packedTag = packedTags[idx];
        size_t primaryBucket = packedTag >> bitsPerTag;
        auto fp = static_cast<TagType>(packedTag & fpMask);

        auto [i2, fp2] =
            FilterType::getAlternateBucketWithNewFp(primaryBucket, fp, filter->numBuckets);
        success = filter->insertCandidates(primaryBucket, i2, fp, fp2, &threadEvictions);

        if (output != nullptr) {
            output[idx] = success;
        }

        if (evictionAttempts != nullptr) {
            evictionAttempts[idx] = threadEvictions;
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        filter->d_numOccupied->fetch_add(blockSum, cuda::memory_order_relaxed);
    }
}

// Bins are sized by the launch's region count and each key's region and rank share a register;
// the staged tile limits the kernel to three blocks per SM.
template <typename Config>
__global__ void __launch_bounds__(Config::blockSize, 768 / Config::blockSize) regionScatterKernel(
    const typename Config::KeyType* keys,
    size_t n,
    size_t numBuckets,
    RegionLayout layout,
    uint32_t* regionCounts,
    typename Filter<Config>::RegionTagType* tags,
    KeyPlacement placement,
    uint32_t* overflow
) {
    using RegionTagType = typename Filter<Config>::RegionTagType;
    constexpr uint32_t blockSize = Config::blockSize;
    constexpr uint32_t tileSize = blockSize * partitionItemsPerThread;
    constexpr uint32_t binsPerThread = (maxRegions + blockSize - 1) / blockSize;
    constexpr uint32_t noSlot = UINT32_MAX;
    using BlockScan = cub::BlockScan<uint32_t, blockSize>;

    // binCount, binStart and regionBase, numRegions entries each.
    extern __shared__ uint32_t binShared[];
    const uint32_t numRegions = layout.numRegions;
    uint32_t* const binCount = binShared;
    uint32_t* const binStart = binShared + numRegions;
    uint32_t* const regionBase = binShared + 2 * numRegions;
    __shared__ RegionTagType staged[tileSize];
    __shared__ uint16_t stagedBin[tileSize];
    __shared__ typename BlockScan::TempStorage scanStorage;

    static_assert(tileSize <= 1u << 16, "ranks share a register with the region");
    for (uint32_t b = threadIdx.x; b < numRegions; b += blockSize) {
        binCount[b] = 0;
    }
    __syncthreads();

    // Rank every key within its region for this tile.
    const size_t base = static_cast<size_t>(blockIdx.x) * tileSize;
    const size_t localMask = (size_t{1} << layout.localBits) - 1;
    RegionTagType tag[partitionItemsPerThread];
    // Region in the high half, rank within the tile's run in the low half.
    uint32_t binRank[partitionItemsPerThread];
    _Pragma("unroll")
    for (uint32_t k = 0; k < partitionItemsPerThread; ++k) {
        const size_t i = base + k * blockSize + threadIdx.x;
        if (i < n) {
            auto [i1, i2, fp1, fp2] =
                Filter<Config>::getCandidateBucketsAndFPs(keys[i], numBuckets);
            tag[k] = (static_cast<RegionTagType>(i1 & localMask) << Config::bitsPerTag) |
                     static_cast<RegionTagType>(fp1);
            const auto b = static_cast<uint32_t>(i1 >> layout.localBits);
            binRank[k] = b << 16 | atomicAdd(&binCount[b], 1u);
        }
    }
    __syncthreads();

    uint32_t counts[binsPerThread];
    _Pragma("unroll")
    for (uint32_t j = 0; j < binsPerThread; ++j) {
        const uint32_t b = threadIdx.x * binsPerThread + j;
        counts[j] = b < numRegions ? binCount[b] : 0;
    }
    BlockScan(scanStorage).ExclusiveSum(counts, counts);
    _Pragma("unroll")
    for (uint32_t j = 0; j < binsPerThread; ++j) {
        const uint32_t b = threadIdx.x * binsPerThread + j;
        if (b < numRegions) {
            binStart[b] = counts[j];
        }
    }

    // Claim this tile's share of each region.
    const uint32_t rankBits = runRankBits(numRegions);
    for (uint32_t b = threadIdx.x; b < numRegions; b += blockSize) {
        const uint32_t count = binCount[b];
        uint32_t slot = noSlot;
        if (count > 0) {
            const uint32_t offset = atomicAdd(&regionCounts[b], count);
            const bool rankFits = placement.keyRun == nullptr || count <= (1u << rankBits);
            if (offset + count <= layout.capacity && rankFits) {
                slot = b * layout.capacity + offset;
            } else {
                atomicOr(overflow, 1u);
            }
        }
        regionBase[b] = slot;
        if (placement.keyRun != nullptr) {
            placement.runSlot[static_cast<size_t>(blockIdx.x) * numRegions + b] = slot;
        }
    }
    __syncthreads();

    // Stage the tile grouped by region, then write each region's run contiguously.
    _Pragma("unroll")
    for (uint32_t k = 0; k < partitionItemsPerThread; ++k) {
        const size_t i = base + k * blockSize + threadIdx.x;
        if (i < n) {
            const uint32_t b = binRank[k] >> 16;
            const uint32_t r = binRank[k] & 0xffffu;
            const uint32_t s = binStart[b] + r;
            staged[s] = tag[k];
            stagedBin[s] = static_cast<uint16_t>(b);
            if (placement.keyRun != nullptr) {
                placement.keyRun[i] = static_cast<uint16_t>(b << rankBits | r);
            }
        }
    }
    __syncthreads();

    const auto tileCount = static_cast<uint32_t>(min(static_cast<size_t>(tileSize), n - base));
    for (uint32_t j = threadIdx.x; j < tileCount; j += blockSize) {
        const uint32_t b = stagedBin[j];
        if (regionBase[b] != noSlot) {
            tags[regionBase[b] + (j - binStart[b])] = staged[j];
        }
    }
}

template <typename Config, RegionOp Op>
__global__ void __launch_bounds__(sortedThreads, sortedBlocksPerSm) regionSortedKernel(
    const typename Filter<Config>::RegionTagType* tags,
    const uint32_t* regionCounts,
    RegionLayout layout,
    uint32_t* results,
    Filter<Config>* filter
) {
    using FilterType = Filter<Config>;
    using RegionTagType = typename FilterType::RegionTagType;
    using TagType = typename FilterType::TagType;
    using BlockScan = cub::BlockScan<uint32_t, sortedThreads>;
    using BlockReduce = cub::BlockReduce<int32_t, sortedThreads>;
    constexpr uint32_t threads = sortedThreads;
    constexpr uint32_t chunk = sortedChunk<RegionTagType>;
    constexpr uint32_t items = chunk / threads;

    // Lanes of a warp whose buckets share a 128-byte line share one L2 request, so the chunk is
    // counting-sorted by local bucket before it is visited. A chunk is a large share of its
    // region, so a warp's 32 sorted entries span few lines.
    extern __shared__ uint4 sharedRaw[];
    auto* sorted = reinterpret_cast<RegionTagType*>(sharedRaw);
    auto* hits = reinterpret_cast<uint32_t*>(sorted + chunk);
    __shared__ uint32_t binStart[threads];
    __shared__ union {
        typename BlockScan::TempStorage scan;
        typename BlockReduce::TempStorage reduce;
    } storage;

    // A region overflowed: the whole batch goes through the unpartitioned path instead.
    if (regionCounts[layout.numRegions] != 0) {
        return;
    }
    const uint32_t chunksPerRegion = SDIV(layout.capacity, chunk);
    const uint32_t region = blockIdx.x / chunksPerRegion;
    const uint32_t first = (blockIdx.x % chunksPerRegion) * chunk;
    const uint32_t count = min(regionCounts[region], layout.capacity);
    if (first >= count) {
        return;
    }
    const uint32_t valid = min(count - first, chunk);
    // Region capacity is a multiple of 32, so the chunk starts on a bitmap word.
    const size_t base = static_cast<size_t>(region) * layout.capacity + first;
    // One bin per thread.
    constexpr uint32_t binBits = std::bit_width(threads) - 1;
    static_assert(threads == 1u << binBits, "thread count must be a power of two");
    const uint32_t binShift = layout.localBits > binBits ? layout.localBits - binBits : 0;
    auto binOf = [&](RegionTagType tag) {
        return static_cast<uint32_t>(tag >> Config::bitsPerTag) >> binShift;
    };

    // The region's chunks run at about the same time, and each touches lines all over the
    // region. Each block prefetches its share of the region into L2 so the DRAM fetches overlap
    // the sort below and the lookups hit L2.
    {
        using Bucket = typename FilterType::Bucket;
        constexpr size_t lineBytes = 128;
        const size_t regionLines =
            ((size_t{1} << layout.localBits) * sizeof(Bucket) + lineBytes - 1) / lineBytes;
        const size_t share = SDIV(regionLines, SDIV(count, chunk));
        const size_t line = (first / chunk) * share + threadIdx.x;
        const auto* regionStart = reinterpret_cast<const char*>(
            filter->d_buckets + (static_cast<size_t>(region) << layout.localBits)
        );
        for (size_t l = line; l < min(regionLines, (first / chunk + 1) * share); l += threads) {
            asm volatile("prefetch.global.L2 [%0];" ::"l"(regionStart + l * lineBytes));
        }
    }

    binStart[threadIdx.x] = 0;
    __syncthreads();
    RegionTagType tag[items];
    // Sorted positions stay live until the results are un-permuted; two 16-bit positions per
    // register leave room for larger chunks without spilling.
    static_assert(chunk <= 1u << 16, "sorted positions are 16-bit");
    uint32_t packedPos[(items + 1) / 2] = {};
    auto posOf = [&](uint32_t k) { return (packedPos[k / 2] >> (16 * (k % 2))) & 0xffffu; };
    _Pragma("unroll")
    for (uint32_t k = 0; k < items; ++k) {
        const uint32_t j = k * threads + threadIdx.x;
        if (j < valid) {
            tag[k] = tags[base + j];
            packedPos[k / 2] |= atomicAdd(&binStart[binOf(tag[k])], 1u) << (16 * (k % 2));
        }
    }
    __syncthreads();
    uint32_t start = binStart[threadIdx.x];
    BlockScan(storage.scan).ExclusiveSum(start, start);
    binStart[threadIdx.x] = start;
    __syncthreads();
    _Pragma("unroll")
    for (uint32_t k = 0; k < items; ++k) {
        if (k * threads + threadIdx.x < valid) {
            // The rank within a bin plus the bin's start is below the chunk size, so no carry.
            packedPos[k / 2] += binStart[binOf(tag[k])] << (16 * (k % 2));
            sorted[posOf(k)] = tag[k];
        }
    }
    __syncthreads();

    const uint32_t warp = threadIdx.x / 32;
    const uint32_t lane = threadIdx.x % 32;
    auto* const buckets = filter->d_buckets;
    const size_t numBuckets = filter->numBuckets;
    auto decode = [&](uint32_t p, size_t& i1, TagType& fp1) {
        const RegionTagType entry = sorted[p];
        i1 = (static_cast<size_t>(region) << layout.localBits) |
             static_cast<size_t>(entry >> Config::bitsPerTag);
        fp1 = static_cast<TagType>(entry & FilterType::fpMask);
    };
    // Primary buckets first. Work that also needs the alternate bucket is deferred, so a warp
    // does not wait on one lane's second round trip (or eviction chain) at every step.
    static_assert(items <= 32, "deferred entries are tracked in a 32-bit mask");
    uint32_t deferred = 0;
    int32_t changed = 0;
    for (uint32_t k = 0; k < items; ++k) {
        const uint32_t p = k * threads + threadIdx.x;
        bool success = false;
        if (p < valid) {
            size_t i1;
            TagType fp1;
            decode(p, i1, fp1);
            if constexpr (Op == RegionOp::Insert) {
                success = filter->tryInsertAtBucket(i1, fp1);
                deferred |= uint32_t(!success) << k;
            } else if constexpr (Op == RegionOp::Delete) {
                success = filter->tryRemoveAtBucket(i1, fp1);
                deferred |= uint32_t(!success) << k;
            } else {
                bool overflow = false;
                success = buckets[i1].contains(fp1, &overflow);
                deferred |= uint32_t(!success && overflow) << k;
            }
            changed += success;
        }
        const uint32_t bits = __ballot_sync(0xffffffffu, success);
        if (lane == 0) {
            hits[p / 32] = bits;
        }
    }
    auto visitAlternate = [&](uint32_t p, bool evict) {
        size_t i1;
        TagType fp1;
        decode(p, i1, fp1);
        auto [i2, fp2] = FilterType::getAlternateBucketWithNewFp(i1, fp1, numBuckets);
        bool success;
        if constexpr (Op == RegionOp::Insert) {
            if (evict) {
                success = filter->insertByEviction(i1, i2, fp1, fp2);
            } else {
                // Whichever way the key ends up outside i1, lookups must learn to check i2.
                filter->markOverflow(i1, fp1);
                success = filter->tryInsertAtBucket(i2, fp2);
            }
        } else if constexpr (Op == RegionOp::Delete) {
            success = filter->tryRemoveAtBucket(i2, fp2);
        } else {
            success = buckets[i2].contains(fp2);
        }
        changed += success;
        if (success) {
            atomicOr(&hits[p / 32], 1u << (p % 32));
        }
    };
    if constexpr (Op == RegionOp::Contains) {
        __syncwarp();
        while (deferred != 0) {
            const uint32_t k = __ffs(deferred) - 1;
            deferred &= deferred - 1;
            visitAlternate(k * threads + threadIdx.x, false);
        }
    } else {
        // Mutations defer more entries, with longer work each (CAS retries, evictions), so they
        // go through a block-wide queue (reusing `binStart`): every thread takes an equal share
        // instead of each warp waiting on its busiest lane.
        static_assert(chunk <= 1u << 16, "queued entries are 16-bit chunk offsets");
        constexpr uint32_t queueSize = threads * sizeof(uint32_t) / sizeof(uint16_t);
        auto* const queue = reinterpret_cast<uint16_t*>(binStart);
        auto drain = [&](bool evict) {
            while (__syncthreads_or(deferred != 0)) {
                uint32_t offset = __popc(deferred);
                uint32_t total;
                BlockScan(storage.scan).ExclusiveSum(offset, offset, total);
                while (deferred != 0 && offset < queueSize) {
                    const uint32_t k = __ffs(deferred) - 1;
                    deferred &= deferred - 1;
                    queue[offset++] = static_cast<uint16_t>(k * threads + threadIdx.x);
                }
                __syncthreads();
                for (uint32_t q = threadIdx.x; q < min(total, queueSize); q += threads) {
                    visitAlternate(queue[q], evict);
                }
            }
        };
        drain(false);
        if constexpr (Op == RegionOp::Insert) {
            // Entries with no room in either bucket evict in a pass of their own, so warps above
            // do not wait on one lane's eviction chain at every step. The last barrier of
            // `drain` made every placement visible in `hits`.
            for (uint32_t k = 0; k < items; ++k) {
                const uint32_t p = k * threads + threadIdx.x;
                deferred |= uint32_t(p < valid && !((hits[p / 32] >> (p % 32)) & 1u)) << k;
            }
            drain(true);
        }
    }
    if constexpr (Op != RegionOp::Contains) {
        __syncthreads();  // storage.scan is reused for the reduction.
        const int32_t blockSum = BlockReduce(storage.reduce).Sum(changed);
        if (threadIdx.x == 0 && blockSum > 0) {
            if constexpr (Op == RegionOp::Insert) {
                filter->d_numOccupied->fetch_add(blockSum, cuda::memory_order_relaxed);
            } else {
                filter->d_numOccupied->fetch_sub(blockSum, cuda::memory_order_relaxed);
            }
        }
    }
    if (results == nullptr) {
        return;
    }
    __syncthreads();

    // Back to slot order. Words past `valid` may belong to the next region, whose capacity can be
    // smaller than a chunk, so they are left alone.
    _Pragma("unroll")
    for (uint32_t k = 0; k < items; ++k) {
        const uint32_t j = k * threads + threadIdx.x;
        const bool success = j < valid && ((hits[posOf(k) / 32] >> (posOf(k) % 32)) & 1u);
        const uint32_t bits = __ballot_sync(0xffffffffu, success);
        if (lane == 0 && j < valid) {
            results[base / 32 + k * (threads / 32) + warp] = bits;
        }
    }
}

template <typename Config>
__global__ void gatherBitmapKernel(
    const uint32_t* results,
    KeyPlacement placement,
    uint32_t numRegions,
    const uint32_t* overflow,
    bool* output,
    size_t n
) {
    constexpr uint32_t items = gatherItemsPerThread;
    constexpr uint32_t tileSize = Config::blockSize * partitionItemsPerThread;
    if (*overflow != 0) {
        return;
    }
    extern __shared__ uint32_t runSlot[];
    const size_t tileRuns = static_cast<size_t>(blockIdx.x) * numRegions;
    for (uint32_t b = threadIdx.x; b < numRegions; b += blockDim.x) {
        runSlot[b] = placement.runSlot[tileRuns + b];
    }
    __syncthreads();

    const uint32_t rankBits = runRankBits(numRegions);
    auto bit = [&](uint32_t code) {
        const uint32_t slot = runSlot[code >> rankBits] + (code & ((1u << rankBits) - 1));
        return (results[slot / 32] >> (slot % 32)) & 1;
    };

    // Consecutive keys per thread: one vector load of placement codes and one vector store.
    static_assert(items == 8, "the vector path moves eight 16-bit indices and eight flags");
    const size_t first = static_cast<size_t>(blockIdx.x) * tileSize + threadIdx.x * items;
    if (first + items <= n && reinterpret_cast<uintptr_t>(output) % 8 == 0) {
        const uint4 s = *reinterpret_cast<const uint4*>(placement.keyRun + first);
        const uint32_t words[4] = {s.x, s.y, s.z, s.w};
        uint32_t packed[2] = {};
        _Pragma("unroll")
        for (uint32_t q = 0; q < items; ++q) {
            packed[q / 4] |= bit((words[q / 2] >> (16 * (q % 2))) & 0xffffu) << (8 * (q % 4));
        }
        *reinterpret_cast<uint2*>(output + first) = make_uint2(packed[0], packed[1]);
    } else {
        for (size_t idx = first; idx < min(first + items, n); ++idx) {
            output[idx] = bit(placement.keyRun[idx]);
        }
    }
}

}  // namespace detail

}  // namespace cuckoogpu
