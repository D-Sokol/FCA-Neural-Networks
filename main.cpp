#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <type_traits>
#include <vector>
#include "fca_algorithms.h"
#include "fcanetwork.h"
#include "measures.h"

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

template <typename RandomIt>
typename iterator_traits<RandomIt>::value_type Average(RandomIt begin, RandomIt end) {
    using value = typename iterator_traits<RandomIt>::value_type;
    static_assert(is_arithmetic_v<value>);
    return accumulate(begin, end, static_cast<value>(0)) / distance(begin, end);
}

int usage(const char* executable);

int print_accuracy(const FCA::Lattice& lattice, const FCA::Context& context,
                   const vector<size_t>& targets, size_t max_level);
int print_accuracy(const vector<size_t>& layer_sizes, const FCA::Context& context,
                   const vector<size_t>& targets);

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
        return print_accuracy(structure, context, targets);
    } else if ("min_supp"s == argv[2]) {
        if (argc != 5) {
            cerr << "Expected arguments: 'min_supp' min_supp max_level" << endl;
            return 3;
        }
        double min_supp = stod(argv[3]);
        size_t max_level = stoi(argv[4]);

        FCA::Predicate pred = [=](const FCA::Concept& c, size_t){ return Support(c) >= min_supp; };
        auto concepts = ThetaSophia(context, pred);
        FCA::Lattice lattice(move(concepts));
        return print_accuracy(lattice, context, targets, max_level);
    } else if ("cv"s == argv[2]) {
        if (argc != 5) {
            cerr << "Expected arguments: 'cv' min_cv max_level" << endl;
            return 3;
        }
        double min_cv = stod(argv[3]);
        size_t max_level = stoi(argv[4]);

        FCA::Predicate pred = [=,&context=context](const FCA::Concept& c, size_t ac){ return CVMeasure(c, context, ac) >= min_cv; };
        auto concepts = ThetaSophia(context, pred);
        FCA::Lattice lattice(move(concepts));
        return print_accuracy(lattice, context, targets, max_level);
    } else if ("cfc"s == argv[2]) {
        if (argc != 5) {
            cerr << "Expected arguments: 'cfc' min_cfc max_level" << endl;
            return 3;
        }
        double min_cfc = stod(argv[3]);
        size_t max_level = stoi(argv[4]);

        FCA::Predicate pred = [=,&context=context](const FCA::Concept& c, size_t ac){ return CVMeasure(c, context, ac) >= min_cfc; };
        auto concepts = ThetaSophia(context, pred);
        FCA::Lattice lattice(move(concepts));
        return print_accuracy(lattice, context, targets, max_level);
    } else {
        return usage(argv[0]);
    }
    return 0;
}

int usage(const char* executable) {
    cerr << "Usage: " << executable << " dataset ('full' | 'min_supp') number [number ...]" << endl;
    return 2;
}

int print_accuracy(const FCA::Lattice& lattice, const FCA::Context& context,
                   const vector<size_t>& targets, size_t max_level) {
    try {
        auto structure = NN::FCANetwork::GetStructure(lattice, targets, 2, max_level);
        auto accuracies = CrossValidationAccuracies(structure, context, targets);
        cout << structure.size() << ' ' << 100.0 * Average(accuracies.begin(), accuracies.end()) << endl;
        return 0;
    } catch (const out_of_range& e) {
        cerr << e.what() << endl;
        return 4;
    }
}

int print_accuracy(const vector<size_t>& layer_sizes, const FCA::Context& context,
                   const vector<size_t>& targets) {
    auto structure = NetworkStructure::FullyConnected(layer_sizes);
    auto accuracies = CrossValidationAccuracies(structure, context, targets);
    cout << structure.size() << ' ' << 100.0 * Average(accuracies.begin(), accuracies.end()) << endl;
    return 0;
}
