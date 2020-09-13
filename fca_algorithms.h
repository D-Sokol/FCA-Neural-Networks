#ifndef FCANN_FCA_ALGORITHMS_H
#define FCANN_FCA_ALGORITHMS_H

#include <functional>
#include <optional>
#include <vector>
#include "fcai/src/fca_datastructures.h"

namespace FCA {
    using Predicate = std::function<bool(const FCA::Concept&, size_t)>;
    const Predicate keep_all = [](const auto&, size_t){ return true; };

    std::vector<FCA::Concept> ThetaSophia(const FCA::Context&,
                                          Predicate keep_concept=keep_all,
                                          std::optional<std::vector<bool>> mask_objects = std::nullopt);

    std::pair<Context, std::vector<size_t>> ReadContext(std::istream& is);
}


#endif //FCANN_FCA_ALGORITHMS_H
