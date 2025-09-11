#include <chrono>
#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "BucketsTableCpu.cuh"
#include "BucketsTableGpu.cuh"
#include "common.cuh"

template <typename T>
size_t countOnes(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

struct BucketSizeBenchmarkResult {
    int exponent{};
    size_t n{};
    size_t bucketSize{};
    std::string tableType;
    double avgsInsertTimeMs{};
    double avgQueryTimeMs{};
    double avgTotalTimeMs{};
    double minTotalTimeMs{};
    double maxTotalTimeMs{};
    size_t itemsInserted{};
    size_t itemsFound{};
    double loadFactor{};
    double falsePositiveRate{};
    size_t memoryUsageBytes{};
    double insertThroughputMops{};
    double queryThroughputMops{};
};

template <typename Func>
std::vector<double> benchmarkFunction(Func func, int iterations = 3) {
    std::vector<double> times(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double timeMs = static_cast<double>(duration.count()) / 1000.0 / 1000.0;
        times[i] = timeMs;
    }
    return times;
}

template <size_t bucketSize>
BucketSizeBenchmarkResult
benchmarkCpuTableWithBucketSize(uint32_t* input, size_t n, int exponent) {
    const int num_runs = 3;
    BucketSizeBenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.bucketSize = bucketSize;
    result.tableType = "BucketsTableCpu";

    double totalInsertTime = 0.0;
    double totalQueryTime = 0.0;
    size_t totalItemsInserted = 0;
    size_t totalItemsFound = 0;
    double totalLoadFactor = 0.0;

    auto benchmarkFunc = [&]() {
        auto table =
            BucketsTableCpu<uint32_t, 32, bucketSize, 1000>(n / bucketSize);

        auto insert_start = std::chrono::high_resolution_clock::now();
        size_t current_inserts = 0;
        for (size_t i = 0; i < n; ++i) {
            if (table.insert(input[i])) {
                current_inserts++;
            }
        }
        auto insert_end = std::chrono::high_resolution_clock::now();
        totalInsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               insert_end - insert_start
                           )
                               .count() /
                           1000.0 / 1000.0;
        totalItemsInserted += current_inserts;

        auto query_start = std::chrono::high_resolution_clock::now();
        bool* output = table.containsMany(input, n);
        auto query_end = std::chrono::high_resolution_clock::now();
        totalQueryTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              query_end - query_start
                          )
                              .count() /
                          1000.0 / 1000.0;

        totalItemsFound += countOnes(output, n);
        std::free(output);

        totalLoadFactor += table.loadFactor();
        result.falsePositiveRate = table.expectedFalsePositiveRate();
        size_t numBuckets = n / bucketSize;
        result.memoryUsageBytes = numBuckets * (bucketSize * sizeof(uint32_t) +
                                                sizeof(int) + sizeof(size_t));
    };

    auto times = benchmarkFunction(benchmarkFunc, num_runs);

    result.avgsInsertTimeMs = totalInsertTime / num_runs;
    result.avgQueryTimeMs = totalQueryTime / num_runs;
    result.itemsInserted = totalItemsInserted / num_runs;
    result.itemsFound = totalItemsFound / num_runs;
    result.loadFactor = totalLoadFactor / num_runs;

    result.minTotalTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTotalTimeMs = *std::max_element(times.begin(), times.end());
    result.avgTotalTimeMs =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    result.insertThroughputMops = (static_cast<double>(n) / 1000000.0) /
                                  (result.avgsInsertTimeMs / 1000.0);
    result.queryThroughputMops =
        (static_cast<double>(n) / 1000000.0) / (result.avgQueryTimeMs / 1000.0);

    return result;
}

template <size_t bucketSize>
BucketSizeBenchmarkResult
benchmarkGpuTableWithBucketSize(uint32_t* input, size_t n, int exponent) {
    const int num_runs = 3;
    BucketSizeBenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.bucketSize = bucketSize;
    result.tableType = "BucketsTableGpu";

    bool* output = nullptr;
    CUDA_CALL(cudaMallocHost(&output, sizeof(bool) * n));

    double totalInsertTime = 0.0;
    double totalQueryTime = 0.0;
    size_t totalItemsInserted = 0;
    size_t totalItemsFound = 0;
    double totalLoadFactor = 0.0;

    auto benchmarkFunc = [&]() {
        auto table =
            BucketsTableGpu<uint32_t, 32, bucketSize, 1000>(n / bucketSize);

        auto insert_start = std::chrono::high_resolution_clock::now();
        size_t count = table.insertMany(input, n);
        auto insert_end = std::chrono::high_resolution_clock::now();
        totalInsertTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                               insert_end - insert_start
                           )
                               .count() /
                           1000.0 / 1000.0;
        totalItemsInserted += count;

        auto query_start = std::chrono::high_resolution_clock::now();
        table.containsMany(input, n, output);
        auto query_end = std::chrono::high_resolution_clock::now();
        totalQueryTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                              query_end - query_start
                          )
                              .count() /
                          1000.0 / 1000.0;

        totalItemsFound += countOnes(output, n);
        totalLoadFactor += table.loadFactor();
        result.falsePositiveRate = table.expectedFalsePositiveRate();
        size_t numBuckets = n / bucketSize;
        result.memoryUsageBytes = numBuckets * (bucketSize * sizeof(uint32_t) +
                                                sizeof(int) + sizeof(size_t));
    };

    auto times = benchmarkFunction(benchmarkFunc, num_runs);

    result.avgsInsertTimeMs = totalInsertTime / num_runs;
    result.avgQueryTimeMs = totalQueryTime / num_runs;
    result.itemsInserted = totalItemsInserted / num_runs;
    result.itemsFound = totalItemsFound / num_runs;
    result.loadFactor = totalLoadFactor / num_runs;

    result.minTotalTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTotalTimeMs = *std::max_element(times.begin(), times.end());
    result.avgTotalTimeMs =
        std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    result.insertThroughputMops = (static_cast<double>(n) / 1000000.0) /
                                  (result.avgsInsertTimeMs / 1000.0);
    result.queryThroughputMops =
        (static_cast<double>(n) / 1000000.0) / (result.avgQueryTimeMs / 1000.0);

    CUDA_CALL(cudaFreeHost(output));
    return result;
}

void writeCsvHeader(const std::string& filename) {
    std::ofstream file(filename, std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing"
                  << std::endl;
        return;
    }
    file << "exponent,n,bucketSize,tableType,avgsInsertTimeMs,avgQueryTimeMs,"
         << "avgTotalTimeMs,minTotalTimeMs,maxTotalTimeMs,itemsInserted,"
         << "itemsFound,loadFactor,falsePositiveRate,memoryUsageMB,"
         << "insertThroughputMops,queryThroughputMops\n";
    file.close();
}

void appendResultToCsv(
    const BucketSizeBenchmarkResult& result,
    const std::string& filename
) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename
                  << " for appending" << std::endl;
        return;
    }
    file << result.exponent << "," << result.n << "," << result.bucketSize
         << "," << result.tableType << "," << std::fixed << std::setprecision(6)
         << result.avgsInsertTimeMs << "," << std::fixed << std::setprecision(6)
         << result.avgQueryTimeMs << "," << std::fixed << std::setprecision(6)
         << result.avgTotalTimeMs << "," << std::fixed << std::setprecision(6)
         << result.minTotalTimeMs << "," << std::fixed << std::setprecision(6)
         << result.maxTotalTimeMs << "," << result.itemsInserted << ","
         << result.itemsFound << "," << std::fixed << std::setprecision(4)
         << result.loadFactor << "," << std::fixed << std::setprecision(8)
         << result.falsePositiveRate << "," << std::fixed
         << std::setprecision(2)
         << (result.memoryUsageBytes / (1024.0 * 1024.0)) << "," << std::fixed
         << std::setprecision(2) << result.insertThroughputMops << ","
         << std::fixed << std::setprecision(2) << result.queryThroughputMops
         << "\n";
    file.close();
}

void printSummaryTable(const std::vector<BucketSizeBenchmarkResult>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "BUCKET SIZE ANALYSIS SUMMARY" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    std::map<
        std::string,
        std::map<size_t, std::vector<BucketSizeBenchmarkResult>>>
        grouped_results;

    for (const auto& result : results) {
        grouped_results[result.tableType][result.n].push_back(result);
    }

    for (const auto& [tableType, size_map] : grouped_results) {
        std::cout << "\n" << tableType << " Performance Analysis:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::cout << std::left << std::setw(12) << "Data Size" << std::setw(12)
                  << "Bucket Size" << std::setw(15) << "Insert (MOPS)"
                  << std::setw(15) << "Query (MOPS)" << std::setw(12)
                  << "Load Factor" << std::setw(12) << "Memory (MB)"
                  << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        for (const auto& [data_size, results_vec] : size_map) {
            for (const auto& result : results_vec) {
                std::cout << std::left << std::setw(12)
                          << ("2^" + std::to_string(result.exponent))
                          << std::setw(12) << result.bucketSize << std::setw(15)
                          << std::fixed << std::setprecision(2)
                          << result.insertThroughputMops << std::setw(15)
                          << std::fixed << std::setprecision(2)
                          << result.queryThroughputMops << std::setw(12)
                          << std::fixed << std::setprecision(3)
                          << result.loadFactor << std::setw(12) << std::fixed
                          << std::setprecision(1)
                          << (result.memoryUsageBytes / (1024.0 * 1024.0))
                          << std::endl;
            }

            auto best_insert = std::max_element(
                results_vec.begin(),
                results_vec.end(),
                [](const auto& a, const auto& b) {
                    return a.insertThroughputMops < b.insertThroughputMops;
                }
            );
            auto best_query = std::max_element(
                results_vec.begin(),
                results_vec.end(),
                [](const auto& a, const auto& b) {
                    return a.queryThroughputMops < b.queryThroughputMops;
                }
            );

            std::cout << "    → Best insert performance: bucket size "
                      << best_insert->bucketSize << " (" << std::fixed
                      << std::setprecision(2)
                      << best_insert->insertThroughputMops << " MOPS)"
                      << std::endl;
            std::cout << "    → Best query performance: bucket size "
                      << best_query->bucketSize << " (" << std::fixed
                      << std::setprecision(2) << best_query->queryThroughputMops
                      << " MOPS)" << std::endl;
            std::cout << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::string output_file = "benchmark_results_bucketSize.csv";

    if (argc > 1) {
        output_file = argv[1];
    }

    const int min_exponent = 10;
    const int max_exponent = 30;

    std::cout << "Generating test data..." << std::endl;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    size_t max_n = 1ULL << max_exponent;
    uint32_t* input = nullptr;
    CUDA_CALL(cudaMallocHost(&input, sizeof(uint32_t) * max_n));

    for (size_t i = 0; i < max_n; ++i) {
        input[i] = dist(rng);
    }

    std::vector<BucketSizeBenchmarkResult> results;

    writeCsvHeader(output_file);
    std::cout << "Results will be written to " << output_file << std::endl;

    std::cout << "\nRunning bucket size benchmarks (averaging over 3 runs)..."
              << std::endl;
    std::cout << "Testing bucket sizes: 4, 8, 16, 32, 64, 128, 256"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int exponent = min_exponent; exponent <= max_exponent; ++exponent) {
        size_t n = 1ULL << exponent;

        std::cout << "\nTesting size 2^" << exponent << " (" << n
                  << " elements):" << std::endl;

        size_t starting_index = results.size();
        const int num_bucket_sizes = 7;

        auto run_and_log = [&](auto benchmark_func) {
            auto result = benchmark_func(input, n, exponent);
            results.push_back(result);
            appendResultToCsv(result, output_file);
        };

        // GPU Benchmarks
        run_and_log(benchmarkGpuTableWithBucketSize<4>);
        run_and_log(benchmarkGpuTableWithBucketSize<8>);
        run_and_log(benchmarkGpuTableWithBucketSize<16>);
        run_and_log(benchmarkGpuTableWithBucketSize<32>);
        run_and_log(benchmarkGpuTableWithBucketSize<64>);
        run_and_log(benchmarkGpuTableWithBucketSize<128>);
        run_and_log(benchmarkGpuTableWithBucketSize<256>);

        // CPU Benchmarks
        run_and_log(benchmarkCpuTableWithBucketSize<4>);
        run_and_log(benchmarkCpuTableWithBucketSize<8>);
        run_and_log(benchmarkCpuTableWithBucketSize<16>);
        run_and_log(benchmarkCpuTableWithBucketSize<32>);
        run_and_log(benchmarkCpuTableWithBucketSize<64>);
        run_and_log(benchmarkCpuTableWithBucketSize<128>);
        run_and_log(benchmarkCpuTableWithBucketSize<256>);

        for (size_t i = starting_index; i < results.size(); ++i) {
            const auto& result = results[i];
            std::cout << "    " << std::setw(15) << std::left
                      << result.tableType << " Bucket " << std::setw(3)
                      << std::right << result.bucketSize << ": " << std::setw(7)
                      << std::fixed << std::setprecision(2)
                      << result.insertThroughputMops << " MOPS (insert), "
                      << std::setw(7) << result.queryThroughputMops
                      << " MOPS (query)" << std::endl;

            if (i == starting_index + num_bucket_sizes - 1) {
                std::cout << std::endl;
            }
        }
    }

    printSummaryTable(results);

    CUDA_CALL(cudaFreeHost(input));

    std::cout << "\nBucket size benchmark completed successfully!" << std::endl;

    return 0;
}