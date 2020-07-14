#include "fcanetwork.h"

using namespace std;

namespace NN {
    using namespace FCA;
    FCANetwork::FCANetwork(const Lattice& lattice, const vector<size_t>& target_classes)
      : Network(NetworkStructure(lattice.GetConnections()))
    {

    }

    Data FCANetwork::Transform(const BitSet& attributes) {
        return NN::Data();
    }

    Data FCANetwork::FitTransform(const BitSet& attributes, size_t target_class) {
        return NN::Data();
    }
}
