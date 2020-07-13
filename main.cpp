#include <iostream>
#include "fcai/src/fca_concept.h"
#include "network.h"

using namespace std;

int main() {
    NN::NetworkStructure structure({1, 2, 2, 1});
    structure.connections[2][0] = vector<size_t>{0};
    structure.connections[2][1] = vector<size_t>{1};
    NN::Network net(structure);
    for (size_t i = 0; i < 450; i += 1) {
        cout << net.FitTransform({2}, {0.74}).front() << ' ';
    }
    cout << endl;
    return 0;
}
