#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <cctype>

using namespace sycl;
using namespace std;


class BloomFilter {
public:
    BloomFilter(size_t size) : size(size), bit_vector(size) {}

    void insert(const string &word) {
        size_t index = hash_function(word) % size;
        bit_vector[index] = true;
    }

    buffer<bool, 1> get_buffer() {
        return buffer<bool, 1>(bit_vector.data(), range<1>(bit_vector.size()));
    }

private:
    size_t size;
    vector<bool> bit_vector;

    size_t hash_function(const string &str) {
        size_t hash = 0;
        for (char c : str) {
            hash = hash * 31 + c;
        }
        return hash;
    }
};

template <typename Container>
void loadContainer(const string &path, size_t minimumWordLength, Container &data) {
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
    } else {
        cerr << "Error: Could not open file '" << path << "'." << endl;
        exit(EXIT_FAILURE);
    }

    inputFile.close();
}


int main() {
    string dictionaryPath = "wordlist.txt";
    string hamletPath = "hamlet_test.txt";
    size_t wordSize = 1;

    set<string> dictionary;
    vector<string> hamletVector;
    loadContainer(dictionaryPath, wordSize, dictionary);
    loadContainer(hamletPath, wordSize, hamletVector);

    BloomFilter bf(12400001);

    for (const auto &word : dictionary) {
        bf.insert(word);
    }

    queue q;

    vector<int> wordCount(hamletVector.size(), 0);

    buffer<int, 1> countBuf(wordCount.data(), range<1>(wordCount.size()));
    buffer<bool, 1> bfBuf = bf.get_buffer();

    q.submit([&](handler &h) {
        auto count = countBuf.get_access<access::mode::read_write>(h);
        auto bf_data = bfBuf.get_access<access::mode::read>(h);

        auto hash_function = [](const string &str) {
            size_t hash = 0;
            for (char c : str) {
                hash = hash * 31 + c;
            }
            return hash;
        };

        h.parallel_for<class word_count_kernel>(range<1>(hamletVector.size()), [=](id<1> i) {
            const auto &word = hamletVector[i];
            size_t index = hash_function(word) % bfBuf.get_count();
            if (bf_data[index]) {
                count[i]++;
            }
        });
    });

    q.wait();

    for (size_t i = 0; i < hamletVector.size(); i++) {
        cout << hamletVector[i] << " : " << wordCount[i] << endl;
    }

    return 0;
}