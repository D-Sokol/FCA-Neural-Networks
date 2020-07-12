#include <iostream>
#include "fcai/src/fca_concept.h"
#include "network.h"

using namespace std;

int main() {
    NN::Network net({1, 2, 1});
    for (double i = 0; i < 5; i += 1) {
        cout << net.Transform({i}).front() << ' ';
    }
    return 0;
}
