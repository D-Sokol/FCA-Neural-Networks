#include <iostream>
#include <vector>
#include "fca_algorithms.h"
#include "fcanetwork.h"

using namespace std;

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

    NN::FCANetwork network(lattice, {0, 1}, 3);

    for (size_t epoch = 0; epoch < 400; ++epoch) {
        auto vec = network.FitTransform(context.Intent(epoch % 4), (epoch & 2) >> 1);
        cout << "Expected " << ((epoch&2)>>1) << ", got: " << vec[0] << ' ' << vec[1] << endl;
    }
    return 0;
}
