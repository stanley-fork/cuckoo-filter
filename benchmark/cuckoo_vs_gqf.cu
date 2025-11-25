#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <quotientFilter.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr unsigned int QF_RBITS = 13;
using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

size_t calcQuotientFilterMemory(unsigned int q, unsigned int r) {
    size_t tableBits = (1ULL << q) * (r + 3);
    size_t tableSlots = tableBits / 8;
    return static_cast<size_t>(tableSlots * 1.1);  // 10% overflow allowance
}

using CFFixture = CuckooFilterFixture<Config>;

void transformQFResults(
    const thrust::device_vector<unsigned int>& d_results,
    thrust::device_vector<unsigned int>& d_found
) {
    thrust::transform(
        d_results.begin(), d_results.end(), d_found.begin(), [] __device__(unsigned int val) {
            return (val != UINT_MAX) ? 1u : 0u;
        }
    );
}

class QFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    static constexpr double TARGET_LOAD_FACTOR = 0.95;

    void SetUp(const benchmark::State& state) override {
        q = static_cast<uint32_t>(std::log2(state.range(0)));
        capacity = 1ULL << q;
        n = capacity * TARGET_LOAD_FACTOR;

        d_keys.resize(n);
        d_results.resize(n);
        generateKeysGPU(d_keys);

        initFilterGPU(&qf, q, QF_RBITS);
        filterMemory = calcQuotientFilterMemory(q, QF_RBITS);
    }

    void TearDown(const benchmark::State&) override {
        if (qf.table != nullptr) {
            cudaFree(qf.table);
            qf.table = nullptr;
        }
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_results.clear();
        d_results.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    uint32_t q;
    size_t capacity;
    size_t n;
    size_t filterMemory;
    struct quotient_filter qf;
    thrust::device_vector<uint32_t> d_keys;
    thrust::device_vector<unsigned int> d_results;
    Timer timer;
};

static void CF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    using FPRConfig = CuckooConfig<uint32_t, 16, 500, 128, 16, XorAltBucketPolicy>;

    auto filter = std::make_unique<CuckooFilter<FPRConfig>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint32_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint32_t>(UINT16_MAX) + 1, UINT32_MAX
    );

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
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

BENCHMARK_DEFINE_F(QFFixture, BulkBuild)(bm::State& state) {
    for (auto _ : state) {
        cudaMemset(qf.table, 0, filterMemory);
        cudaDeviceSynchronize();

        timer.start();
        float time = bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
    }
    setCounters(state);
}
BENCHMARK_DEFINE_F(QFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        cudaMemset(qf.table, 0, filterMemory);
        cudaDeviceSynchronize();

        timer.start();
        float time = insert(qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()));
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
    }
    setCounters(state);
}
BENCHMARK_DEFINE_F(QFFixture, QuerySorted)(bm::State& state) {
    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    for (auto _ : state) {
        timer.start();
        float time = launchSortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }
    setCounters(state);
}
BENCHMARK_DEFINE_F(QFFixture, QueryUnsorted)(bm::State& state) {
    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    for (auto _ : state) {
        timer.start();
        float time = launchUnsortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }
    setCounters(state);
}
BENCHMARK_DEFINE_F(QFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        cudaMemset(qf.table, 0, filterMemory);
        bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        cudaDeviceSynchronize();

        timer.start();
        float time =
            superclusterDeletes(qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()));
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
    }
    setCounters(state);
}

static void QF_FPR(bm::State& state) {
    Timer timer;
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * 0.95;
    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU<uint32_t>(d_keys, UINT16_MAX);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);
    cudaMemset(qf.table, 0, filterMemory);
    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint32_t> d_neverInserted(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint32_t>(UINT16_MAX) + 1, UINT32_MAX
    );

    thrust::device_vector<unsigned int> d_results(fprTestSize);

    for (auto _ : state) {
        timer.start();
        float time = launchSortedLookups(
            qf,
            static_cast<int>(fprTestSize),
            thrust::raw_pointer_cast(d_neverInserted.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }

    thrust::device_vector<unsigned int> d_found(fprTestSize);
    transformQFResults(d_results, d_found);

    size_t falsePositives =
        thrust::reduce(d_found.begin(), d_found.end(), 0ULL, thrust::plus<size_t>());
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

    cudaFree(qf.table);
}

#define BENCHMARK_CONFIG_QF             \
    ->RangeMultiplier(2)                \
        ->Range(1 << 10, 1ULL << 18)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->MinTime(0.5)                  \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define REGISTER_CF_BENCHMARK(BenchName)       \
    BENCHMARK_REGISTER_F(CFFixture, BenchName) \
    BENCHMARK_CONFIG_QF

// I'm lazy, this is an easy way to replace the config
#undef BENCHMARK_CONFIG
#define BENCHMARK_CONFIG BENCHMARK_CONFIG_QF

DEFINE_AND_REGISTER_CORE_BENCHMARKS(CFFixture)

REGISTER_BENCHMARK(QFFixture, BulkBuild);
REGISTER_BENCHMARK(QFFixture, Insert);
REGISTER_BENCHMARK(QFFixture, QuerySorted);
REGISTER_BENCHMARK(QFFixture, QueryUnsorted);
REGISTER_BENCHMARK(QFFixture, Delete);

REGISTER_FUNCTION_BENCHMARK(CF_FPR);
REGISTER_FUNCTION_BENCHMARK(QF_FPR);

STANDARD_BENCHMARK_MAIN();
