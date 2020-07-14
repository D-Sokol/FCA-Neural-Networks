#include <iostream>
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

int main() {
    FCA::Context context(vector<vector<bool>>{
        { true, false, false,  true},
        { true, false,  true, false},
        {false,  true,  true, false},
        {false,  true,  true,  true}
    });

    auto concepts = ThetaSophia(context);

    FCA::Lattice lattice(move(concepts));

    NN::FCANetwork network(lattice, {0, 1, 2, 3}, 3);

    for (size_t epoch = 0; epoch < 400; ++epoch) {
        auto vec = network.FitTransform(context.Intent(epoch % 4), (epoch % 4));
        cout << "Expected " << (epoch % 4) << ", got: " << vec << endl;
    }
    return 0;
}
