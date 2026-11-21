#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <cuckoogpu/CuckooFilter.cuh>
#include <cuckoogpu/helpers.cuh>
#include <random>
#include <vector>

class CuckooFilterTest : public ::testing::Test {
   protected:
    using Config = cuckoogpu::Config<uint32_t, 16, 500, 256, 16>;

    const double TARGET_LOAD_FACTOR = 0.95;

    void SetUp() override {
        rng.seed(42);
    }

    template <typename T>
    std::vector<T> generateRandomKeys(size_t n) {
        std::vector<T> keys(n);
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
        std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });

        return keys;
    }

    std::mt19937 rng;
};

TEST_F(CuckooFilterTest, BasicInsertAndQuery) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted successfully";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, EmptyFilter) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> queryKeys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(queryKeys.begin(), queryKeys.end());
    thrust::device_vector<uint8_t> d_output(queryKeys.size());

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(queryKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "Empty filter should not find any keys";
}

TEST_F(CuckooFilterTest, ClearOperation) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    filter.insertMany(d_keys);

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key should be found before clear";
    }

    filter.clear();
    filter.containsMany(d_keys, d_output);

    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "After clear, no keys should be found";
    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f) << "Load factor should be 0 after clear";
}

TEST_F(CuckooFilterTest, LoadFactor) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f);

    const size_t numKeys = 5000;

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    float loadFactor = filter.loadFactor();
    EXPECT_GT(loadFactor, 0.0f) << "Load factor should be positive after insertions";
    EXPECT_LE(loadFactor, 1.0f) << "Load factor should not exceed 1.0";

    const auto totalCapacity = static_cast<float>(filter.capacity());

    EXPECT_FLOAT_EQ(loadFactor, inserted / totalCapacity)
        << "Load factor should be approximately inserted / totalCapacity";

    EXPECT_GT(inserted, 0) << "Some keys should be inserted";
}

TEST_F(CuckooFilterTest, NearCapacityInsertion) {
    const size_t numKeys = (1 << 20) * TARGET_LOAD_FACTOR;
    cuckoogpu::Filter<Config> filter(numKeys);

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(numKeys);

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_GT(inserted, numKeys * 0.99) << "Should insert at least 99% of keys";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(numKeys);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_GT(found, inserted * 0.99) << "Should find at least 99% of inserted keys";
}

TEST_F(CuckooFilterTest, DuplicateInsertions) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    EXPECT_EQ(inserted, keys.size());

    std::vector<uint32_t> uniqueKeys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_unique_keys(uniqueKeys.begin(), uniqueKeys.end());
    thrust::device_vector<uint8_t> d_output(uniqueKeys.size());

    filter.containsMany(d_unique_keys, d_output);

    std::vector<uint8_t> output(uniqueKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Unique key " << uniqueKeys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, BasicDeletion) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted";

    filter.containsMany(d_keys, d_output);
    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found before deletion";
    }

    thrust::device_vector<uint8_t> d_deleteOutput(keys.size());
    size_t remainingAfterDelete = filter.deleteMany(d_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(keys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(deleteOutput[i]) << "Key " << keys[i] << " should be successfully deleted";
    }

    EXPECT_EQ(remainingAfterDelete, 0) << "No keys should remain after deletion";

    filter.containsMany(d_keys, d_output);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "No keys should be found after deletion";
}

TEST_F(CuckooFilterTest, DeleteNonExistentKeys) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(keys.size());

    filter.deleteMany(d_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(keys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    size_t deleted = std::count(deleteOutput.begin(), deleteOutput.end(), true);
    EXPECT_EQ(deleted, 0) << "Should not delete any non-existent keys";
}

TEST_F(CuckooFilterTest, PartialDeletion) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> insertKeys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    thrust::device_vector<uint32_t> d_insertKeys(insertKeys.begin(), insertKeys.end());

    size_t inserted = filter.insertMany(d_insertKeys);
    EXPECT_EQ(inserted, insertKeys.size()) << "All keys should be inserted";

    std::vector<uint32_t> deleteKeys = {2, 4, 6, 8, 10};
    thrust::device_vector<uint32_t> d_delete_keys(deleteKeys.begin(), deleteKeys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(deleteKeys.size());

    size_t remainingAfterDelete = filter.deleteMany(d_delete_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(deleteKeys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    for (size_t i = 0; i < deleteKeys.size(); ++i) {
        EXPECT_TRUE(deleteOutput[i]) << "Key " << deleteKeys[i] << " should be deleted";
    }

    EXPECT_EQ(remainingAfterDelete, 5) << "5 keys should remain after deleting 5";

    thrust::device_vector<uint8_t> d_output(deleteKeys.size());
    filter.containsMany(d_delete_keys, d_output);

    std::vector<uint8_t> output(deleteKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "Deleted keys should not be found";

    std::vector<uint32_t> remainingKeys = {1, 3, 5, 7, 9};
    thrust::device_vector<uint32_t> d_remainingKeys(remainingKeys.begin(), remainingKeys.end());
    thrust::device_vector<uint8_t> d_remainingOutput(remainingKeys.size());

    filter.containsMany(d_remainingKeys, d_remainingOutput);

    std::vector<uint8_t> remainingOutput(remainingKeys.size());
    thrust::copy(d_remainingOutput.begin(), d_remainingOutput.end(), remainingOutput.begin());

    for (size_t i = 0; i < remainingKeys.size(); ++i) {
        EXPECT_TRUE(remainingOutput[i]) << "Key " << remainingKeys[i] << " should still be found";
    }
}

TEST_F(CuckooFilterTest, DeleteAndReinsert) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    filter.insertMany(d_keys);
    filter.deleteMany(d_keys);

    filter.containsMany(d_keys, d_output);
    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "Keys should not be found after deletion";

    size_t reinserted = filter.insertMany(d_keys);
    EXPECT_EQ(reinserted, keys.size()) << "Should be able to reinsert deleted keys";

    filter.containsMany(d_keys, d_output);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found after reinsertion";
    }
}

TEST_F(CuckooFilterTest, LoadFactorAfterDeletion) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    const size_t numKeys = 1000;
    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);
    float loadFactorBefore = filter.loadFactor();

    EXPECT_GT(loadFactorBefore, 0.0f) << "Load factor should be positive after insertion";

    size_t remaining = filter.deleteMany(d_keys);
    float loadFactorAfter = filter.loadFactor();

    EXPECT_LT(loadFactorAfter, loadFactorBefore) << "Load factor should decrease after deletion";
    EXPECT_FLOAT_EQ(loadFactorAfter, static_cast<float>(remaining) / filter.capacity());
}

TEST_F(CuckooFilterTest, DeleteDuplicates) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 1, 1, 2, 2, 3};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    filter.insertMany(d_keys);

    std::vector<uint32_t> deleteKeys = {1, 1, 1};
    thrust::device_vector<uint32_t> d_deleteKeys(deleteKeys.begin(), deleteKeys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(deleteKeys.size());

    filter.deleteMany(d_deleteKeys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(deleteKeys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    size_t successfulDeletions = std::count(deleteOutput.begin(), deleteOutput.end(), true);
    EXPECT_GE(successfulDeletions, 1) << "At least one deletion should succeed";
}

TEST_F(CuckooFilterTest, InsertAndVerifyOutput) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_insertOutput(keys.size());

    size_t inserted = filter.insertMany(d_keys, d_insertOutput);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted";

    std::vector<uint8_t> insertOutput(keys.size());
    thrust::copy(d_insertOutput.begin(), d_insertOutput.end(), insertOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(insertOutput[i]) << "Key " << keys[i] << " should be successfully inserted";
    }

    // Verify keys are actually in the filter
    thrust::device_vector<uint8_t> d_containsOutput(keys.size());
    filter.containsMany(d_keys, d_containsOutput);

    std::vector<uint8_t> containsOutput(keys.size());
    thrust::copy(d_containsOutput.begin(), d_containsOutput.end(), containsOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(containsOutput[i]) << "Key " << keys[i] << " should be found after insertion";
    }
}

TEST_F(CuckooFilterTest, InsertFullFilterReportsFailures) {
    // Use a small capacity to easily fill the filter
    const size_t capacity = 100;
    cuckoogpu::Filter<Config> filter(capacity);

    // Generate more keys than the filter can hold
    const size_t numKeys = capacity * 2;
    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_insertOutput(numKeys);

    size_t inserted = filter.insertMany(d_keys, d_insertOutput);

    std::vector<uint8_t> insertOutput(numKeys);
    thrust::copy(d_insertOutput.begin(), d_insertOutput.end(), insertOutput.begin());

    size_t successCount = std::count(insertOutput.begin(), insertOutput.end(), true);
    size_t failureCount = std::count(insertOutput.begin(), insertOutput.end(), false);

    // The output array should report the correct number of successes
    EXPECT_EQ(successCount, inserted) << "Success count should match inserted count";
    EXPECT_GT(failureCount, 0) << "Some insertions should fail when filter is full";
    EXPECT_EQ(successCount + failureCount, numKeys) << "All keys should have a result";
}

// Concurrent eviction chains in a nearly full filter must move fingerprints, never leave an
// uncounted copy behind: the occupancy counter has to match the table after every batch.
template <template <typename, typename, size_t, size_t> class Policy>
void expectEvictionsKeepOneCopyPerKey() {
    using Config = cuckoogpu::Config<uint32_t, 16, 500, 256, 16, Policy>;
    const char* policy = Config::AltBucketPolicy::name;
    cuckoogpu::Filter<Config> filter(1 << 12);
    const size_t prefill = filter.capacity() * 96 / 100;
    const size_t n = filter.capacity() * 35 / 1000;
    thrust::device_vector<uint32_t> keys(prefill + n);
    for (uint32_t batch = 0; batch < 500; ++batch) {
        thrust::sequence(keys.begin(), keys.end(), batch * static_cast<uint32_t>(keys.size()) + 1);
        filter.clear();
        filter.insertMany(thrust::raw_pointer_cast(keys.data()), prefill);
        const size_t counted =
            filter.insertMany(thrust::raw_pointer_cast(keys.data()) + prefill, n);
        ASSERT_EQ(filter.countOccupiedSlots(), counted) << policy << " batch " << batch;
    }
}

TEST_F(CuckooFilterTest, ConcurrentEvictionsKeepOneCopyPerKey) {
    expectEvictionsKeepOneCopyPerKey<cuckoogpu::XorAltBucketPolicy>();
    expectEvictionsKeepOneCopyPerKey<cuckoogpu::OffsetAltBucketPolicy>();
}

// Keys moved to their alternate bucket by evictions must stay visible to lookups, which skip
// the alternate bucket unless the primary bucket says something overflowed.
template <template <typename, typename, size_t, size_t> class Policy>
void expectEvictedKeysFound() {
    using Config = cuckoogpu::Config<uint32_t, 16, 500, 256, 16, Policy>;
    const char* policy = Config::AltBucketPolicy::name;
    cuckoogpu::Filter<Config> filter(1 << 16);
    const size_t n = filter.capacity() * 99 / 100;
    thrust::device_vector<uint32_t> keys(n);
    thrust::sequence(keys.begin(), keys.end(), 1u);
    const uint32_t* d_keys = thrust::raw_pointer_cast(keys.data());
    // Several batches so later ones evict tags placed by earlier ones.
    for (size_t first = 0; first < n; first += n / 4) {
        filter.insertMany(d_keys + first, std::min(n / 4, n - first));
    }
    ASSERT_EQ(filter.occupiedSlots(), n) << policy;

    thrust::device_vector<uint8_t> found(n);
    auto* d_found = reinterpret_cast<bool*>(thrust::raw_pointer_cast(found.data()));
    filter.containsMany(d_keys, n, d_found);
    EXPECT_EQ(thrust::count(found.begin(), found.end(), 0), 0) << policy;

    const size_t half = n / 2;
    filter.deleteMany(d_keys, half);
    filter.containsMany(d_keys + half, n - half, d_found);
    EXPECT_EQ(thrust::count(found.begin(), found.begin() + (n - half), 0), 0) << policy;
}

TEST_F(CuckooFilterTest, EvictedKeysRemainVisible) {
    expectEvictedKeysFound<cuckoogpu::XorAltBucketPolicy>();
    expectEvictedKeysFound<cuckoogpu::OffsetAltBucketPolicy>();
}

TEST_F(CuckooFilterTest, InsertSortedAndVerifyOutput) {
    const size_t capacity = 10000;
    cuckoogpu::Filter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_insertOutput(keys.size());

    size_t inserted = filter.insertManySorted(d_keys, d_insertOutput);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted";

    std::vector<uint8_t> insertOutput(keys.size());
    thrust::copy(d_insertOutput.begin(), d_insertOutput.end(), insertOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(insertOutput[i]) << "Key " << keys[i] << " should be successfully inserted";
    }

    // Verify keys are actually in the filter
    thrust::device_vector<uint8_t> d_containsOutput(keys.size());
    filter.containsMany(d_keys, d_containsOutput);

    std::vector<uint8_t> containsOutput(keys.size());
    thrust::copy(d_containsOutput.begin(), d_containsOutput.end(), containsOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(containsOutput[i]) << "Key " << keys[i] << " should be found after insertion";
    }
}

// Filters larger than L2 group each batch by bucket region before processing; per-key outputs
// must still come back in input order.
TEST_F(CuckooFilterTest, LargerThanL2KeepsOutputOrder) {
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    // 16-bit tags: 2 bytes per slot, so this is about 4x the L2 size.
    const size_t capacity = 2 * static_cast<size_t>(l2Bytes);
    cuckoogpu::Filter<Config> filter(capacity);
    ASSERT_GT(filter.sizeInBytes(), static_cast<size_t>(l2Bytes));

    // Insert every odd key 1, 3, 5, ... up to ~90% load, then query 1, 2, 3, ... interleaved.
    const size_t numInserted = filter.capacity() * 9 / 10;
    thrust::device_vector<uint32_t> d_inserted(numInserted);
    thrust::sequence(d_inserted.begin(), d_inserted.end(), 1u, 2u);
    thrust::device_vector<uint8_t> d_insertOutput(numInserted);

    size_t occupied = filter.insertMany(d_inserted, d_insertOutput);
    ASSERT_EQ(occupied, numInserted);
    EXPECT_EQ(thrust::count(d_insertOutput.begin(), d_insertOutput.end(), uint8_t{1}), numInserted);

    const size_t numQueried = 2 * numInserted;
    thrust::device_vector<uint32_t> d_queried(numQueried);
    thrust::sequence(d_queried.begin(), d_queried.end(), 1u);
    thrust::device_vector<uint8_t> d_containsOutput(numQueried);
    filter.containsMany(d_queried, d_containsOutput);

    std::vector<uint8_t> found(numQueried);
    thrust::copy(d_containsOutput.begin(), d_containsOutput.end(), found.begin());

    size_t falseNegatives = 0;
    size_t falsePositives = 0;
    for (size_t i = 0; i < numQueried; ++i) {
        const bool wasInserted = i % 2 == 0;
        falseNegatives += wasInserted && !found[i];
        falsePositives += !wasInserted && found[i];
    }
    EXPECT_EQ(falseNegatives, 0u);
    // Expected FPR is about 2 * 16 * 0.9 / 2^16 = 0.04%; misordered output would give ~50%.
    EXPECT_LT(falsePositives, numInserted / 100);
}

// Batches too small to fill a sorted chunk per region give regions shorter than a chunk; each
// region's results must survive its neighbours' writes.
TEST_F(CuckooFilterTest, LargerThanL2SmallBatchKeepsAllResults) {
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    using Filter = cuckoogpu::Filter<Config>;
    Filter filter((static_cast<size_t>(l2Bytes) / sizeof(Filter::Bucket) + 1) * Config::bucketSize);
    ASSERT_GT(filter.sizeInBytes(), static_cast<size_t>(l2Bytes));
    const size_t n = filter.getNumBuckets();
    thrust::device_vector<uint32_t> keys(n);
    thrust::sequence(keys.begin(), keys.end(), 1u);

    thrust::device_vector<uint8_t> output(n);
    ASSERT_EQ(filter.insertMany(keys, output), n);
    EXPECT_EQ(thrust::count(output.begin(), output.end(), uint8_t{1}), n);

    filter.containsMany(keys, output);
    EXPECT_EQ(thrust::count(output.begin(), output.end(), uint8_t{1}), n);

    ASSERT_EQ(filter.deleteMany(keys, output), 0u);
    EXPECT_EQ(thrust::count(output.begin(), output.end(), uint8_t{1}), n);
}

// Runs of one key longer than a per-key placement code can rank must still be answered in input
// order.
TEST_F(CuckooFilterTest, LargerThanL2LongDuplicateRunsKeepOutputOrder) {
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    using Filter = cuckoogpu::Filter<Config>;
    Filter filter(2 * static_cast<size_t>(l2Bytes) / sizeof(Filter::Bucket) * Config::bucketSize);
    ASSERT_GT(filter.sizeInBytes(), static_cast<size_t>(l2Bytes));
    const size_t n = filter.capacity() / 2;
    thrust::device_vector<uint32_t> inserted(n);
    thrust::sequence(inserted.begin(), inserted.end(), 1u);
    ASSERT_EQ(filter.insertMany(inserted), n);

    // Every key repeats `run` times in a row, alternating inserted and absent keys.
    constexpr uint32_t run = 512;
    auto wasInserted = [](size_t i) { return (i / run + 1) % 2 == 0; };
    std::vector<uint32_t> h_queried(n);
    for (size_t i = 0; i < n; ++i) {
        const auto key = static_cast<uint32_t>(i / run + 1);
        h_queried[i] = wasInserted(i) ? key : key + 0x80000000u;
    }
    thrust::device_vector<uint32_t> queried(h_queried);
    thrust::device_vector<uint8_t> output(n);
    filter.containsMany(queried, output);

    std::vector<uint8_t> found(n);
    thrust::copy(output.begin(), output.end(), found.begin());
    size_t falseNegatives = 0;
    size_t falsePositives = 0;
    for (size_t i = 0; i < n; ++i) {
        falseNegatives += wasInserted(i) && !found[i];
        falsePositives += !wasInserted(i) && found[i];
    }
    EXPECT_EQ(falseNegatives, 0u);
    // A false positive costs a whole run, but at ~0.06% FPR only a few runs are hit.
    EXPECT_LT(falsePositives, n / 50);
}

// A batch whose keys all land in one bucket region cannot be region-grouped; it must still be
// answered correctly.
TEST_F(CuckooFilterTest, LargerThanL2SkewedBatch) {
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    cuckoogpu::Filter<Config> filter(2 * static_cast<size_t>(l2Bytes));
    ASSERT_GT(filter.sizeInBytes(), static_cast<size_t>(l2Bytes));

    thrust::device_vector<uint32_t> d_key(1, 42u);
    ASSERT_EQ(filter.insertMany(d_key), 1u);

    const size_t n = 1 << 20;
    thrust::device_vector<uint32_t> d_queried(n, 42u);
    thrust::device_vector<uint8_t> d_found(n);
    filter.containsMany(d_queried, d_found);
    EXPECT_EQ(thrust::count(d_found.begin(), d_found.end(), uint8_t{1}), n);
}

TEST_F(CuckooFilterTest, LargerThanL2WithOddBucketCount) {
    using OffsetConfig =
        cuckoogpu::Config<uint32_t, 16, 500, 256, 16, cuckoogpu::OffsetAltBucketPolicy>;
    using Filter = cuckoogpu::Filter<OffsetConfig>;
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    const size_t buckets = (static_cast<size_t>(l2Bytes) / sizeof(Filter::Bucket) + 1) | 1;
    Filter filter(buckets * OffsetConfig::bucketSize);
    constexpr size_t n = 16387;
    thrust::device_vector<uint32_t> keys(n);
    thrust::sequence(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> found(n);

    ASSERT_EQ(filter.insertMany(keys), n);
    filter.containsMany(keys, found);
    EXPECT_EQ(thrust::count(found.begin(), found.end(), uint8_t{1}), n);
}

TEST_F(CuckooFilterTest, LargerThanL2DeletePreservesOutputOrder) {
    int device = 0;
    int l2Bytes = 0;
    ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device), cudaSuccess);

    using Filter = cuckoogpu::Filter<Config>;
    Filter filter((static_cast<size_t>(l2Bytes) / sizeof(Filter::Bucket) + 1) * Config::bucketSize);
    const size_t n = filter.getNumBuckets() + 3;
    thrust::device_vector<uint32_t> keys(n);
    thrust::sequence(keys.begin(), keys.end(), 1u);
    const size_t first = n / 3;
    const size_t second = 2 * first;
    const uint32_t key = keys[first];
    keys[second] = key;
    thrust::device_vector<uint32_t> duplicates(2, key);
    ASSERT_EQ(filter.insertMany(duplicates), 2u);

    thrust::device_vector<uint8_t> output(n);
    filter.containsMany(keys, output);
    // Exclude keys that already collide before testing delete output order.
    std::vector<uint8_t> found(n);
    thrust::copy(output.begin(), output.end(), found.begin());
    const auto miss = std::find(found.begin(), found.end(), uint8_t{0});
    ASSERT_NE(miss, found.end());
    const uint32_t missKey = std::distance(found.begin(), miss) + 1;
    for (size_t i = 0; i < n; ++i) {
        if (i != first && i != second && found[i]) {
            keys[i] = missKey;
        }
    }
    ASSERT_EQ(filter.deleteMany(keys, output), 0u);
    EXPECT_EQ(thrust::count(output.begin(), output.end(), uint8_t{1}), 2);
    EXPECT_EQ(static_cast<uint8_t>(output[first]), 1u);
    EXPECT_EQ(static_cast<uint8_t>(output[second]), 1u);
    thrust::device_vector<uint8_t> remaining(2);
    filter.containsMany(duplicates, remaining);
    EXPECT_EQ(thrust::count(remaining.begin(), remaining.end(), uint8_t{1}), 0);
}
