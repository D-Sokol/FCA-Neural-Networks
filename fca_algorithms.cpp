#include <algorithm>
#include "fca_algorithms.h"

using namespace std;

namespace FCA {
    vector<Concept> ThetaSophia(const Context& context, size_t min_size) {
        // Concept candidate with label that shows if the first one is proper concept.
        using MarkedConcept = pair<Concept, bool>;
        // Projection phi_{-1} produces only one trivial concept,
        //  that contains all objects and no attributes.
        vector<MarkedConcept> result = {{{context.ObjSize(), context.AttrSize()}, true}};
        result.front().first.Extent().flip();

        for (size_t i = 0; i < context.AttrSize(); ++i) {
            // Consider projection phi_i that consider attributes from 0 to i-th, inclusively.
            const size_t concepts_number = result.size();
            for (size_t k = 0; k < concepts_number; ++k) {
                {
                    // Concept (A, B) should be kept in the result, if the set A_{-} is not empty
                    // A_{-} = {a \in A | a does not possess attribute i}
                    auto& concept_ = result[k];
                    auto ext = context.Extent(i);
                    ext.flip();
                    ext &= concept_.first.Extent();
                    if (ext.any())
                        result.push_back(concept_);
                }
                {
                    // (A, B) can produces a preimage (A_{+}, B \sup {i}),
                    // where A_{+} = {a \in A | a possess attribute i}
                    auto& concept_ = result[k];  // Reference from the previous block may be invalidated.

                    concept_.first.Extent() &= context.Extent(i);
                    concept_.first.Intent().set(i);
                    if (!concept_.first.Intent().is_prefix_equal(context.DrvtObj(concept_.first.Extent()), i)) {
                        concept_.second = false;
                    }
                }
            }
            // Filter found concepts to have at least min_size elements.
            auto it = remove_if(result.begin(), result.end(),
                                [=](const MarkedConcept& c) { return !c.second || c.first.ExtentSize() < min_size; });
            result.erase(it, result.end());
        }

        vector<Concept> return_value;
        return_value.reserve(result.size());
        for (auto& [concept_, label] : result)
            return_value.push_back(move(concept_));
        return return_value;
    }
}
