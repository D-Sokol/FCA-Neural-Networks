#ifndef FCANN_FCANETWORK_H
#define FCANN_FCANETWORK_H

#include "lattice.h"
#include "network.h"

namespace NN {
    class FCANetwork : protected Network {
    public:
        FCANetwork(const FCA::Lattice& lattice, const std::vector<size_t>& target_classes);
        Data Transform(const FCA::BitSet& attributes);
        Data FitTransform(const FCA::BitSet& attributes, size_t target_class);
    };
}


#endif //FCANN_FCANETWORK_H
