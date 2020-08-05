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

int main() {
    ifstream example("datasets/mammographic_masses.txt");
    auto [context, targets] = FCA::ReadContext(example);
    auto concepts = ThetaSophia(context);
    FCA::Lattice lattice(move(concepts));
    for (size_t max_level = 2; max_level < 5; ++max_level) {
        NN::FCANetwork network(lattice, targets, 3);
        cout << "\nUse max_level = " << max_level << '\n';
        cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
        cout << "Accuracy: " << 100.0 * Accuracy(network, context, targets) << '%' << endl;
    }
    return 0;
}
