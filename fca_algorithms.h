#ifndef FCANN_FCA_ALGORITHMS_H
#define FCANN_FCA_ALGORITHMS_H

#include <vector>
#include "fcai/src/fca_datastructures.h"

namespace FCA {
    std::vector<FCA::Concept> ThetaSophia(const FCA::Context&, size_t min_size = 0);

    std::pair<Context, std::vector<size_t>> ReadContext(std::istream& is);
}


#endif //FCANN_FCA_ALGORITHMS_H
