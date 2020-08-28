#ifndef FCANN_FCA_ALGORITHMS_H
#define FCANN_FCA_ALGORITHMS_H

#include <functional>
#include <vector>
#include "fcai/src/fca_datastructures.h"

namespace FCA {
    using Predicate = std::function<bool(const FCA::Concept&)>;
    template <typename F, typename=std::void_t<decltype(std::declval<F>()(std::declval<FCA::Concept>()))>>
    Predicate Bind(F&& measure, double threshold) {
        return [=,measure=std::decay_t<F>(measure)](const Concept& c) {return measure(c) >= threshold; };
    }

    std::vector<FCA::Concept> ThetaSophia(const FCA::Context&, Predicate keep_concept = [](const auto&){ return true; });

    std::pair<Context, std::vector<size_t>> ReadContext(std::istream& is);
}


#endif //FCANN_FCA_ALGORITHMS_H
