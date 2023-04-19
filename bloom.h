#ifndef BLOOM_H
#define BLOOM_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;

class Bloom_Filter_T {
public:
    void Bloom_Filter(int numberOfBits, int numberOfHashFunctions);
    void insert(string element);
    double search(string element);
    bool boolSearch(string element);
    string get_data();
    int getCollisions();

private:
    int hash_f_1(string word);
    int hash_f_2(string word);
};

set<string> loadSet(string path, int minimumWordLength);
vector<string> loadVector(string hamletPath, int minimumWordLength);

#endif
