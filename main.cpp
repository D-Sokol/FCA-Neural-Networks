#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "fca_algorithms.h"
#include "fcanetwork.h"

using namespace std;

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

size_t CycleTrainNetwork(NN::FCANetwork& network, const FCA::Context& context,
                         const vector<size_t>& targets, size_t iter_limit=100) {
    const size_t objects = context.ObjSize();

    vector<size_t> train_indexes, test_indexes;
    train_indexes.reserve((4 * objects + 4) / 5);
    test_indexes.reserve((1 * objects + 4) / 5);
    for (size_t id = 0; id < objects; ++id) {
        if (id % 5 != 4)
           train_indexes.push_back(id);
        else
            test_indexes.push_back(id);
    }
    random_shuffle(train_indexes.begin(), train_indexes.end());
    random_shuffle(test_indexes.begin(), test_indexes.end());

    const size_t EPOCH_ACCURACY_DECREASES_TO_EXIT = 5;
    size_t epoch_accuracy_decreases = 0;
    size_t last_correct_answers = 0;

    for (size_t epoch = 1; epoch < iter_limit; ++epoch) {
        for (auto id : train_indexes)
            network.FitTransform(context.Intent(id), targets[id]);

        size_t corrects = 0;

        for (auto id : test_indexes) {
            auto vec = network.FitTransform(context.Intent(id), targets[id]);
            size_t y_pred = max_element(vec.begin(), vec.end()) - vec.begin();
            if (targets[id] == y_pred)
                ++corrects;
        }

        cerr << "Accuracy after " << epoch << " iterations: "
             << 100.0 * static_cast<double>(corrects) / test_indexes.size() << '%' << endl;

        if (corrects > last_correct_answers) {
            epoch_accuracy_decreases = 0;
        } else if (epoch_accuracy_decreases++ >= EPOCH_ACCURACY_DECREASES_TO_EXIT) {
            return epoch;
        }
        last_correct_answers = corrects;
    }
    return iter_limit;
}

double Accuracy(NN::FCANetwork& network, const FCA::Context& context, const vector<size_t>& targets) {
    size_t corrects = 0;
    const size_t objects = context.ObjSize();
    for (size_t id = 0; id < objects; ++id) {
        auto vec = network.Transform(context.Intent(id));
        size_t y_pred = max_element(vec.begin(), vec.end()) - vec.begin();
        if (targets[id] == y_pred)
            ++corrects;
    }
    return static_cast<double>(corrects) / objects;
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
        cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
        cout << "Accuracy: " << 100.0 * Accuracy(network, context, targets) << '%' << endl;
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
            cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
            cout << "Accuracy: " << 100.0 * Accuracy(network, context, targets) << '%' << endl;
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
