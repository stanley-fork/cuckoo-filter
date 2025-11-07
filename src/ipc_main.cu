#include <thrust/host_vector.h>
#include <chrono>
#include <CLI/CLI.hpp>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "CuckooFilterIPC.cuh"

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

void runServer(const std::string& name, size_t capacity, bool forceShutdown = false) {
    std::cout << "Starting server with capacity: " << capacity << std::endl;
    if (forceShutdown) {
        std::cout << "Force shutdown mode enabled (pending requests will be cancelled)"
                  << std::endl;
    }

    try {
        CuckooFilterIPCServer<Config> server(name, capacity);
        server.start();

        std::cout << "Server running. Press Enter to stop..." << std::endl;
        std::cin.get();

        server.stop(forceShutdown);
        std::cout << "Server stopped." << std::endl;

        auto filter = server.getFilter();
        std::cout << "Final load factor: " << filter->loadFactor() << std::endl;
        std::cout << "Occupied slots: " << filter->occupiedSlots() << std::endl;
        std::cout << "Capacity: " << filter->capacity() << std::endl;
        std::cout << "Size in bytes: " << filter->sizeInBytes() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void runClient(const std::string& name, int clientId, size_t numKeys) {
    std::cout << "Client " << clientId << " starting..." << std::endl;

    try {
        // Give the server some time to initialise
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        CuckooFilterIPCClient<Config> client(name);

        std::vector<uint64_t> h_keys(numKeys);
        std::random_device rd;
        std::mt19937_64 gen(rd() + clientId);
        std::uniform_int_distribution<uint64_t> dis(1, UINT32_MAX);
        for (size_t i = 0; i < numKeys; i++) {
            h_keys[i] = dis(gen);
        }

        uint64_t* d_keys;
        bool* d_results;
        CUDA_CALL(cudaMalloc(&d_keys, numKeys * sizeof(uint64_t)));
        CUDA_CALL(cudaMalloc(&d_results, numKeys * sizeof(bool)));
        CUDA_CALL(
            cudaMemcpy(d_keys, h_keys.data(), numKeys * sizeof(uint64_t), cudaMemcpyHostToDevice)
        );

        auto start = std::chrono::high_resolution_clock::now();

        size_t occupiedAfterInsert = client.insertMany(d_keys, numKeys);
        std::cout << "Client " << clientId << " inserted " << numKeys << " keys (filter now has "
                  << occupiedAfterInsert << " occupied slots)" << std::endl;

        client.containsMany(d_keys, numKeys, d_results);

        std::vector<uint8_t> h_results(numKeys);
        CUDA_CALL(
            cudaMemcpy(h_results.data(), d_results, numKeys * sizeof(bool), cudaMemcpyDeviceToHost)
        );

        size_t found = 0;
        for (bool result : h_results) {
            if (result) {
                found++;
            }
        }
        std::cout << "Client " << clientId << " found " << found << "/" << numKeys << " keys"
                  << std::endl;

        size_t deleteCount = numKeys / 2;
        size_t occupiedAfterDelete = client.deleteMany(d_keys, deleteCount, d_results);
        std::cout << "Client " << clientId << " deleted " << deleteCount << " keys (filter now has "
                  << occupiedAfterDelete << " occupied slots)" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Client " << clientId << " completed in " << duration.count() << "ms"
                  << std::endl;

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_results));

    } catch (const std::exception& e) {
        std::cerr << "Client " << clientId << " error: " << e.what() << std::endl;
    }
}

void runClientThrust(const std::string& name, int clientId, size_t numKeys) {
    std::cout << "Thrust Client " << clientId << " starting..." << std::endl;

    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        CuckooFilterIPCClientThrust<Config> client(name);

        thrust::host_vector<uint64_t> h_keys(numKeys);
        std::random_device rd;
        std::mt19937_64 gen(rd() + clientId);
        std::uniform_int_distribution<uint64_t> dis(1, UINT32_MAX);
        for (size_t i = 0; i < numKeys; i++) {
            h_keys[i] = dis(gen);
        }

        thrust::device_vector<uint64_t> d_keys = h_keys;
        thrust::device_vector<bool> d_results(numKeys);

        auto start = std::chrono::high_resolution_clock::now();

        size_t occupiedAfterInsert = client.insertMany(d_keys);
        std::cout << "Thrust Client " << clientId << " inserted " << numKeys
                  << " keys (filter now has " << occupiedAfterInsert << " occupied slots)"
                  << std::endl;

        client.containsMany(d_keys, d_results);

        thrust::host_vector<bool> h_results = d_results;
        size_t found = 0;
        for (bool result : h_results) {
            if (result) {
                found++;
            }
        }
        std::cout << "Thrust Client " << clientId << " found " << found << "/" << numKeys << " keys"
                  << std::endl;

        size_t deleteCount = numKeys / 2;
        thrust::device_vector<uint64_t> d_keysToDelete(
            d_keys.begin(), d_keys.begin() + deleteCount
        );
        size_t occupiedAfterDelete = client.deleteMany(d_keysToDelete);
        std::cout << "Thrust Client " << clientId << " deleted " << deleteCount
                  << " keys (filter now has " << occupiedAfterDelete << " occupied slots)"
                  << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Thrust Client " << clientId << " completed in " << duration.count() << "ms"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Thrust Client " << clientId << " error: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    CLI::App app{"Cuckoo Filter IPC Server/Client"};
    app.require_subcommand(1);

    auto* serverCmd = app.add_subcommand("server", "Run the IPC server");

    std::string serverName;
    int serverCapacityExp = 25;
    bool forceShutdown = false;

    serverCmd->add_option("name", serverName, "Server name for IPC")->required();

    serverCmd
        ->add_option("-c,--capacity", serverCapacityExp, "Capacity exponent (capacity = 2^x)")
        ->default_val(25)
        ->check(CLI::PositiveNumber);

    serverCmd->add_flag("-f,--force", forceShutdown, "Force shutdown (cancel pending requests)");

    auto* clientCmd = app.add_subcommand("client", "Run IPC client(s)");

    std::string clientType = "normal";
    int numClients = 1;
    int clientCapacityExp = 25;
    double targetLoadFactor = 0.95;

    clientCmd->add_option("name", serverName, "Server name to connect to")->required();

    clientCmd->add_option("-t,--type", clientType, "Client type")
        ->default_val("normal")
        ->check(CLI::IsMember({"normal", "thrust"}));

    clientCmd->add_option("-n,--num-clients", numClients, "Number of concurrent clients")
        ->default_val(1)
        ->check(CLI::PositiveNumber);

    clientCmd
        ->add_option(
            "-c,--capacity",
            clientCapacityExp,
            "Capacity exponent (must match server, keys = 2^x * loadFactor)"
        )
        ->default_val(25)
        ->check(CLI::PositiveNumber);

    clientCmd->add_option("-l,--load-factor", targetLoadFactor, "Target load factor")
        ->check(CLI::Range(0.0, 1.0))
        ->default_val(0.95);

    CLI11_PARSE(app, argc, argv);

    if (*serverCmd) {
        size_t capacity = 1ULL << serverCapacityExp;
        runServer(serverName, capacity, forceShutdown);
    } else if (*clientCmd) {
        size_t numKeys = (1ULL << clientCapacityExp) * targetLoadFactor;
        auto clientFunc = (clientType == "normal") ? runClient : runClientThrust;

        std::vector<std::thread> clientThreads;
        for (int i = 0; i < numClients; i++) {
            clientThreads.emplace_back(clientFunc, serverName, i, numKeys);
        }

        for (auto& thread : clientThreads) {
            thread.join();
        }
    }

    return 0;
}