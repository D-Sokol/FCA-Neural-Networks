#include <iostream>
#include "fcai/src/fca_concept.h"
#include "network.h"

using namespace std;

int main() {
    NN::NetworkStructure structure{{
        {},
        {},
        {0, 1},
        {0, 1},
        {},
        {2, 3, 4}
    }};
    NN::Network net(structure);
    for (size_t i = 0; i < 10; i += 1) {
        cout << net.FitTransform({2, 1}, {0.74}).back() << ' ';
    }
    cout << endl;
    return 0;
}
