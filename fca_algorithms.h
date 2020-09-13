#ifndef FCANN_FCA_ALGORITHMS_H
#define FCANN_FCA_ALGORITHMS_H

#include <functional>
#include <optional>
#include <vector>
#include "fcai/src/fca_datastructures.h"

namespace FCA {
    using Predicate = std::function<bool(const FCA::Concept&, size_t)>;

    std::vector<FCA::Concept> ThetaSophia(const FCA::Context&,
                                          Predicate keep_concept = [](const auto&, size_t){ return true; },
                                          std::optional<std::vector<bool>> mask_objects = std::nullopt);

    std::pair<Context, std::vector<size_t>> ReadContext(std::istream& is);
}


#endif //FCANN_FCA_ALGORITHMS_H
