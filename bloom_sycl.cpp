#include "bloom.h"
#include <CLI/CLI.hpp>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <string.h>
#include <unistd.h>
#include <bitset>
#include <limits>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <fstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>

#include <CL/sycl.hpp>

using namespace std;
using namespace cl::sycl;

namespace WordCountBloomFilter {

class BloomFilter {
public:
    BloomFilter(size_t numberOfBits, size_t numberOfHashFunctions)
        : numBits(numberOfBits)
        , numHashFuncs(numberOfHashFunctions)
        , numInserts(0)
        , collisions(0) {
            data = make_unique<bitset<numeric_limits<size_t>::max()>>(numBits);
    }

    void insert(const string& element) {
        vector<size_t> hashes = getHashes(element);

        queue myQueue;
        buffer<size_t, 1> hashes_buffer(hashes.data(), range<1>(hashes.size()));
        buffer<bool, 1> data_buffer(data->size());
        std::copy(data->begin(), data->end(), data_buffer.get_host_access().begin());
        myQueue.submit([&](handler& cgh) {
            auto hashes_acc = hashes_buffer.get_access<access::mode::read>(cgh);
            auto data_acc = data_buffer.get_access<access::mode::read_write>(cgh);
            cgh.parallel_for<class insert_kernel>(range<1>(hashes.size()), [=](id<1> idx) {
                size_t hash = hashes_acc[idx];
                if (data_acc[hash]) {
                    atomic_ref<int, memory_order::relaxed, memory_scope::device, access::address_space::global_space> collisions_atomic(collisions);
                    collisions_atomic++;
                }
                data_acc[hash] = true;
            });
        });
        myQueue.wait_and_throw();
        std::copy(data_buffer.get_host_access().begin(), data_buffer.get_host_access().end(), data->begin());
        numInserts++;
    }

    double search(const string& element) {
        vector<size_t> hashes = getHashes(element);

        bool all_found = true;
        queue myQueue;
        buffer<size_t, 1> hashes_buffer(hashes.data(), range<1>(hashes.size()));
        buffer<bool, 1> data_buffer(data->size());
        std::copy(data->begin(), data->end(), data_buffer.get_host_access().begin());
        myQueue.submit([&](handler& cgh) {
            auto hashes_acc = hashes_buffer.get_access<access::mode::read>(cgh);
            auto data_acc = data_buffer.get_access<access::mode::read>(cgh);
            cgh.parallel_for<class search_kernel>(range<1>(hashes.size()), [=](id<1> idx) mutable {
            size_t hash = hashes_acc[idx];
                if (!data_acc[hash]) {
                    all_found = false;
                }
            });
        });
        myQueue.wait_and_throw();

        if (!all_found) {
            return -1.0;
        }

        double prob = pow(1.0 - pow(1.0 - 1.0 / numBits, numHashFuncs * numInserts), numHashFuncs);
        return prob;
    }


    int get_collisions() { return collisions; }

private:
    size_t numBits;
    size_t numHashFuncs;
    unique_ptr<bitset<numeric_limits<size_t>::max()>> data;
    int numInserts;
    int collisions;

    int hashF1(const string& word) {
        unsigned char md[MD5_DIGEST_LENGTH];
        MD5(reinterpret_cast<const unsigned char*>(word.c_str()), word.size(), md);
        hash<string> strHash;
        string str_md(reinterpret_cast<const char*>(md), MD5_DIGEST_LENGTH);
        return strHash(str_md.substr(0, 6)) % numBits;
    }

    int hashF2(const string& word) {
        unsigned char md[SHA256_DIGEST_LENGTH];
        SHA256(reinterpret_cast<const unsigned char*>(word.c_str()), word.size(), md);
        hash<string> strHash;
        string str_md(reinterpret_cast<const char*>(md), SHA256_DIGEST_LENGTH);
        return strHash(str_md.substr(0, 6)) % numBits;
    }

    vector<size_t> getHashes(const string& word) {
        vector<size_t> hashes(numHashFuncs);

        int hash1 = hashF1(word);
        int hash2 = hashF2(word);

        for (size_t i = 0; i < numHashFuncs; i++) {
            hashes[i] = (hash1 + i * hash2) % numBits;
        }

        return hashes;
    }
};

template <typename Container>
void loadContainer(const string& path, size_t minimumWordLength, Container& data) {
    string line;
    ifstream inputFile;

    inputFile.open(path);

    if (inputFile.is_open()) {
        while (getline(inputFile, line)) {
            transform(line.begin(), line.end(), line.begin(), ::toupper);
            if (line.length() >= minimumWordLength && line.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == string::npos) {
                data.insert(data.end(), line);
            }
        }
        inputFile.close();
    } else {
        cerr << "Error: Could not open file '" << path << "'." << endl;
        exit(EXIT_FAILURE);
    }
}

}  // namespace WordCountBloomFilter

int main(int argc, char **argv) {
    int numberOfBits = 10;
    int numberOfHashFunctions = 1;
    string dictionaryPath = "wordlist.txt";
    string hamletPath = "hamlet_test.txt";
    size_t wordSize = 1;

    CLI::App app{"Bloom Filter Implementation"};
    app.option_defaults()->always_capture_default(true);

    app.add_option("-b,--bits", numberOfBits,
                    "Number of bits to allocate to the bit vector (default = 10)")
        ->check(CLI::PositiveNumber.description(
            "Number of bits to allocate to the bit vector (default = 10)"));

    app.add_option("-f,--hashf", numberOfHashFunctions,
                    "Number of functions to hash the data (default = 1)")
        ->check(CLI::PositiveNumber.description(
            "Number of functions to hash the data (default = 1)"));

    app.add_option("-d,--dict", dictionaryPath,
                    "Path to dictionary (default = wordlist.txt)");

    app.add_option("--hamlet", hamletPath,
                    "Path to hamlet (default = hamlet_test.txt)");

    app.add_option("--wordSize", wordSize, "Minimum word size (default = 1)");

    CLI11_PARSE(app, argc, argv);

    set<string> dictionary;
    set<string> hamletSet;
    vector<string> hamletVector;
    WordCountBloomFilter::loadContainer(dictionaryPath, wordSize, dictionary);
    WordCountBloomFilter::loadContainer(hamletPath, wordSize, hamletSet);
    WordCountBloomFilter::loadContainer(hamletPath, wordSize, hamletVector);

    WordCountBloomFilter::BloomFilter bf(numberOfBits, numberOfHashFunctions);

    unordered_map<string, int> wordCount;

    for (const auto &word : hamletVector) {
        if (bf.search(word) != -1.0) {
            wordCount[word]++;
        }
    }

    for (const auto &[word, count] : wordCount) {
        cout << word << " : " << count << endl;
    }

    return 0;
}
