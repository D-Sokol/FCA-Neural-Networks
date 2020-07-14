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
    for (size_t epoch = 1; epoch < iter_limit; ++epoch) {
        for (size_t id = 0; id < objects; ++id) {
            network.FitTransform(context.Intent(id), targets[id]);
        }
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

    NN::FCANetwork network(lattice, targets, 2);
    cout << CycleTrainNetwork(network, context, targets) << " iterations passed\n";
    cout << "Accuracy: " << 100.0 * Accuracy(network, context, targets) << '%' << endl;
    return 0;
}
