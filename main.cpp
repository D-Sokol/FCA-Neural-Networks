#include <iostream>
#include "fcai/src/fca_concept.h"
#include "network.h"

using namespace std;

int main() {
    NN::Network net({1, 2, 1});
    for (size_t i = 0; i < 450; i += 1) {
        cout << net.FitTransform({2}, {0.74}).front() << ' ';
    }
    cout << endl;
    return 0;
}
