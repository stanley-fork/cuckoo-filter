#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/std/cstdint>
#include <helpers.cuh>
#include <random>

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
using Config = CuckooConfig<uint32_t, 16, 500, 128, 128>;


// Generate random keys on GPU
template <typename T>
void generateKeysGPU(thrust::device_vector<T>& d_keys, unsigned seed = 42) {
    size_t n = d_keys.size();
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        d_keys.begin(),
        [seed] __device__ (size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
            rng.discard(idx);
            return dist(rng);
        }
    );
}

template <typename Filter, size_t bitsPerTag>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;

    return (n * bitsPerTag) / (Filter::words_per_block * bitsPerWord);
}

static void BM_CuckooFilter_Insert(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(n, TARGET_LOAD_FACTOR);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(inserted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CuckooFilter_Query(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(n, TARGET_LOAD_FACTOR);
    thrust::device_vector<uint8_t> d_output(n);

    filter.insertMany(d_keys);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        filter.containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CuckooFilter_Delete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(n, TARGET_LOAD_FACTOR);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        filter.insertMany(d_keys);
        state.ResumeTiming();

        size_t remaining = filter.deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_BloomFilter_Insert(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    constexpr auto bitsPerTag = Config::bitsPerTag;

    using BloomFilter = cuco::bloom_filter<uint32_t>;

    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    BloomFilter filter(
        cuco::extent{numBlocks},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{}
    );

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        filter.add(d_keys.begin(), d_keys.end());
        cudaDeviceSynchronize();
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_BloomFilter_Query(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    constexpr auto bitsPerTag = Config::bitsPerTag;

    using BloomFilter = cuco::bloom_filter<uint32_t>;

    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    BloomFilter filter(
        cuco::extent{numBlocks},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{}
    );

    thrust::device_vector<uint8_t> d_output(n);

    filter.add(d_keys.begin(), d_keys.end());

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    for (auto _ : state) {
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CuckooFilter_InsertAndQuery(bm::State& state) {
    const size_t n = state.range(0);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(n, TARGET_LOAD_FACTOR);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        filter.containsMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_CuckooFilter_InsertQueryDelete(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(n, TARGET_LOAD_FACTOR);

    size_t filterMemory = filter.sizeInBytes();
    size_t capacity = filter.capacity();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        filter.containsMany(d_keys, d_output);
        size_t remaining = filter.deleteMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(capacity),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void BM_BloomFilter_InsertAndQuery(bm::State& state) {
    const size_t n = state.range(0) * TARGET_LOAD_FACTOR;

    constexpr auto bitsPerTag = Config::bitsPerTag;

    using BloomFilter = cuco::bloom_filter<uint32_t>;
    const size_t numBlocks = cucoNumBlocks<BloomFilter, bitsPerTag>(n);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);

    BloomFilter filter(
        cuco::extent{numBlocks},
        cuco::cuda_thread_scope<cuda::thread_scope_device>{}
    );

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        filter.add(d_keys.begin(), d_keys.end());
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );

        cudaDeviceSynchronize();

        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

BENCHMARK(BM_CuckooFilter_Insert)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Insert)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_Query)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_Delete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_BloomFilter_InsertAndQuery)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(BM_CuckooFilter_InsertQueryDelete)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
