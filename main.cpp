#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "fca_algorithms.h"
#include "fcanetwork.h"

using namespace std;
using namespace NN;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& vec) {
    os << '[';
    for (const auto& x : vec)
        os << x << ", ";
    return os << ']';
}

std::ostream& operator<<(std::ostream& os, const FCA::Concept& c) {
    for (size_t i = 0; i < c.Extent().size(); ++i)
        if (c.Extent().test(i))
            os << i << ',';
    os << " | ";
    for (size_t i = 0; i < c.Intent().size(); ++i)
        if (c.Intent().test(i))
            os << static_cast<char>('a' + i) << ',';

    return os;
}

int usage(const char* executable);

int main(int argc, char** argv) {
    if (argc < 3) {
        return usage(argv[0]);
    }

    ifstream input_stream(argv[1]);
    if (!input_stream) {
        cerr << "Cannot open file: " << argv[1] << endl;
        return 1;
    }

    auto [context, targets] = FCA::ReadContext(input_stream);

    if ("full"s == argv[2]) {
        vector<size_t> structure(argc - 1u);
        structure.front() = context.AttrSize();
        structure.back() = *max_element(targets.begin(), targets.end()) + 1;
        for (size_t i = 1; i < argc - 2u; ++i) {
            structure[i] = stoi(argv[i+2]);
        }
        NN::FCANetwork network(structure);
        // cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
        CycleTrainNetwork(network, context, targets);
        cout << 100.0 * Accuracy(network, context, targets) << endl;
    } else if ("min_supp"s == argv[2]) {
        if (argc != 5) {
            cerr << "Expected arguments: 'min_supp' min_supp max_level" << endl;
            return 3;
        }
        double min_supp = stod(argv[3]);
        size_t max_level = stoi(argv[4]);

        FCA::Predicate pred = [&](const FCA::Concept& c) {
            return c.ExtentSize() >= min_supp * c.Extent().size();
        };
        auto concepts = ThetaSophia(context, pred);
        FCA::Lattice lattice(move(concepts));
        try {
            NN::FCANetwork network(lattice, targets, max_level);
            // cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
            CycleTrainNetwork(network, context, targets);
            cout << 100.0 * Accuracy(network, context, targets) << endl;
        } catch (const out_of_range& e) {
            cerr << e.what() << endl;
            return 4;
        }
    } else {
        return usage(argv[0]);
    }
    return 0;
}

int usage(const char* executable) {
    cerr << "Usage: " << executable << " dataset ('full' | 'min_supp') number [number ...]" << endl;
    return 2;
}
