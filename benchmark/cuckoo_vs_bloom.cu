#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;

    return (n * Config::bitsPerTag) / (Filter::words_per_block * bitsPerWord);
}

static void CF_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(filter, d_keys);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    adaptiveInsert(filter, d_keys);

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        timer.start();
        filter.containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        adaptiveInsert(filter, d_keys);
        cudaDeviceSynchronize();

        timer.start();
        size_t remaining = filter.deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void BBF_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    using BloomFilter = cuco::bloom_filter<uint64_t>;

    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    BloomFilter filter(numBlocks);

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        cudaDeviceSynchronize();

        timer.start();
        filter.add(d_keys.begin(), d_keys.end());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
    }

    setCommonCounters(state, filterMemory, n);
}

static void BBF_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    using BloomFilter = cuco::bloom_filter<uint64_t>;
    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    BloomFilter filter(numBlocks);

    thrust::device_vector<uint8_t> d_output(n);

    filter.add(d_keys.begin(), d_keys.end());

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    Timer timer;

    for (auto _ : state) {
        timer.start();
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(filter, d_keys);
        filter.containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_InsertQueryDelete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(filter, d_keys);
        filter.containsMany(d_keys, d_output);
        size_t remaining = filter.deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void BBF_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    using BloomFilter = cuco::bloom_filter<uint64_t>;
    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);

    BloomFilter filter(numBlocks);

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    Timer timer;

    for (auto _ : state) {
        filter.clear();
        cudaDeviceSynchronize();

        timer.start();
        filter.add(d_keys.begin(), d_keys.end());
        filter.contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CF_FPR(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    CuckooFilter<Config> filter(capacity);
    adaptiveInsert(filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = filter.sizeInBytes();

    Timer timer;

    for (auto _ : state) {
        timer.start();
        filter.containsMany(d_neverInserted, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

static void BBF_FPR(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    using BloomFilter = cuco::bloom_filter<uint64_t>;
    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    BloomFilter filter(numBlocks);
    filter.add(d_keys.begin(), d_keys.end());

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = filter.block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);

    Timer timer;

    for (auto _ : state) {
        timer.start();
        filter.contains(d_neverInserted.begin(), d_neverInserted.end(), d_output.begin());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

BENCHMARK(CF_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);
BENCHMARK(BBF_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK(CF_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);
BENCHMARK(BBF_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK(CF_Delete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK(CF_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);
BENCHMARK(BBF_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK(CF_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK(CF_FPR)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);
BENCHMARK(BBF_FPR)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1ULL << 28)
    ->Unit(bm::kMillisecond)
    ->UseManualTime()
    ->MinTime(0.5)
    ->Repetitions(5)
    ->ReportAggregatesOnly(true);

BENCHMARK_MAIN();
