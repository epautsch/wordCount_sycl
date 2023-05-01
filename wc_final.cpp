#include <sycl/sycl.hpp>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <utility>


using namespace sycl;
using namespace std;


// From oneAPI tutorial
class CustomDeviceSelector {
 public:
  CustomDeviceSelector(std::string vendorName) : vendorName_(vendorName){};
  int operator()(const device &dev) {
    int device_rating = 0;
    if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) !=
                        std::string::npos))
      device_rating = 3;
    else if (dev.is_gpu())
      device_rating = 2;
    else if (dev.is_cpu())
      device_rating = 1;
    return device_rating;
  };

 private:
  std::string vendorName_;
};


vector<string> readWordsFromFile(const string &path, size_t minimumWordLength) {
    vector<string> words;
    string line, word;
    ifstream inputFile;

    inputFile.open(path);

    if (inputFile.is_open()) {
        while (getline(inputFile, line)) {
            transform(line.begin(), line.end(), line.begin(), ::toupper);
            istringstream lineStream(line);

            while (lineStream >> word) {
                if (word.length() >= minimumWordLength && word.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == string::npos) {
                    words.push_back(word);
                }
            }
        }
    } else {
        cerr << "Error: Could not open file '" << path << "'." << "\n";
        exit(EXIT_FAILURE);
    }

    inputFile.close();

    return words;
}

// for non-parallelized version
// unordered_map<string, int> countWordOccurrences(const vector<string> &words) {
//     unordered_map<string, int> wordCounts;
//     for (const string &word : words) {
//         wordCounts[word]++;
//     }
//     return wordCounts;
// }

struct StringData {
    char data[32];

    StringData() {
        data[0] = '\0';
    }

    StringData(const std::string &str) {
        std::strncpy(data, str.c_str(), sizeof(data));
        data[sizeof(data) - 1] = '\0';
    }

    bool operator==(const StringData &other) const {
        return std::strncmp(data, other.data, sizeof(data)) == 0;
    }
};


void countWordOccurrences(sycl::queue &q, const std::vector<StringData> &words,
                          const std::vector<StringData> &uniqueWords,
                          std::vector<int> &wordCounts) {

    sycl::buffer inputWordsBuffer(words.data(), sycl::range<1>(words.size()));
    sycl::buffer uniqueWordsBuffer(uniqueWords.data(), sycl::range<1>(uniqueWords.size()));
    sycl::buffer wordCountsBuffer(wordCounts.data(), sycl::range<1>(wordCounts.size()));

    q.submit([&](sycl::handler &h) {
        auto inputWordsAccessor = inputWordsBuffer.get_access<sycl::access::mode::read>(h);
        auto uniqueWordsAccessor = uniqueWordsBuffer.get_access<sycl::access::mode::read>(h);
        auto wordCountsAccessor = wordCountsBuffer.get_access<sycl::access::mode::read_write>(h);

        // use accessors in the lambda, not vectors
        h.parallel_for(sycl::range<1>(words.size()), [=](sycl::id<1> i) {
            const StringData &word = inputWordsAccessor[i];
            // get_count causing error, using size
            for (size_t j = 0; j < uniqueWordsAccessor.size(); ++j) {
                if (uniqueWordsAccessor[j] == word) {
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomicCounter(wordCountsAccessor[j]);
                    atomicCounter.fetch_add(1);
                }
            }
        });
    });

    q.wait_and_throw();
}



vector<pair<string, int>> mapToVector(const unordered_map<string, int> &wordCounts) {
    vector<pair<string, int>> wordCountVector(wordCounts.begin(), wordCounts.end());
    return wordCountVector;
}


bool compareWordCounts(const pair<string, int> &a, const pair<string, int> &b) {
    return a.second > b.second;
}


int main() {
    string targetFilePath = "hamlet_manylines.txt";
    size_t minimumWordLength = 10;
    vector<string> targetWords = readWordsFromFile(targetFilePath, minimumWordLength);

    std::vector<StringData> targetWordsData;
    targetWordsData.reserve(targetWords.size());
    for (const auto &word : targetWords) {
        targetWordsData.push_back(StringData(word));
    }

    std::unordered_set<std::string> wordSet(targetWords.begin(), targetWords.end());
    std::vector<StringData> uniqueWordsData;
    uniqueWordsData.reserve(wordSet.size());
    for (const auto &word : wordSet) {
        uniqueWordsData.push_back(StringData(word));
    }

    std::vector<int> wordCounts(uniqueWordsData.size());

    try {
        std::string vendor_name = "Intel";
        // std::string vendor_name = "AMD";
        // std::string vendor_name = "Nvidia";
        CustomDeviceSelector selector(vendor_name);
        sycl::queue q(selector);
        countWordOccurrences(q, targetWordsData, uniqueWordsData, wordCounts);
    } catch (...) {
        std::cout << "Failure" << "\n";
        std::terminate();
    }
    

    std::vector<std::pair<std::string, int>> wordCountPairs(uniqueWordsData.size());
    for (size_t i = 0; i < uniqueWordsData.size(); ++i) {
        wordCountPairs[i] = std::make_pair(std::string(uniqueWordsData[i].data), wordCounts[i]);
    }
    sort(wordCountPairs.begin(), wordCountPairs.end(), compareWordCounts);

    cout << "Word counts:" << "\n";
    for (const auto &pair : wordCountPairs) {
        cout << pair.first << ": " << pair.second << "\n";
    }

    return 0;
}

