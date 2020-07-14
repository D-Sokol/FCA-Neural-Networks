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
    size_t i = 0;
    for (const auto& c : lattice.GetConcepts())
        cout << i++ << ' ' << c << endl;

    i = 0;
    for (const auto& vec : lattice.GetConnections()) {
        cout << i++ << ": ";
        for (size_t index : vec)
            cout << index << ',';
        cout << endl;
    }

    NN::FCANetwork network(lattice, {0, 1}, 3);
    return 0;
}
