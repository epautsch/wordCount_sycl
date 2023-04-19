#include "bloom.h"
#include <CLI/CLI.hpp>
#include <openssl/md5.h>
#include <openssl/sha.h>
#include <string.h>
#include <unistd.h>
#include <bitset>
#include <limits>


using namespace std;

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

        for (size_t hash : hashes) {
            if ((*data)[hash]) {
                collisions++;
            }
            (*data)[hash] = true;
        }

        numInserts++;
    }

    double search(const string& element) {
        vector<size_t> hashes = getHashes(element);

        for (size_t hash : hashes) {
            if (!(*data)[hash]) {
                return -1.0;
            }
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
    } else {
        cerr << "Error: Could not open file '" << path << "'." << endl;
        exit(EXIT_FAILURE);
    }

    inputFile.close();
}

}  // namespace WordCountBloomFilter

#include <iostream>
#include <unordered_map>
#include <vector>
#include <set>
#include <fstream>
#include <algorithm>
#include <string>
#include <iterator>
#include <cctype>

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

  for (const auto &word : dictionary) {
    bf.insert(word);
  }

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
