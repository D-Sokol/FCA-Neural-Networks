#include <algorithm>
#include "fca_algorithms.h"

using namespace std;

namespace FCA {
    vector<Concept> ThetaSophia(const Context& context, size_t min_size) {
        // Projection phi_{-1} produces only one trivial concept,
        //  that contains all objects and no attributes.
        vector<Concept> result = {{context.ObjSize(), context.AttrSize()}};
        result.front().Extent().flip();

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
                    ext &= concept_.Extent();
                    if (ext.any())
                        result.push_back(concept_);
                }
                {
                    // (A, B) can produces a preimage (A_{+}, B \sup {i}),
                    // where A_{+} = {a \in A | a possess attribute i}
                    auto& concept_ = result[k];  // Reference from the previous block may be invalidated.

                    concept_.Extent() &= context.Extent(i);
                    concept_.Intent().set(i);
                    if (!concept_.Intent().is_prefix_equal(context.DrvtObj(concept_.Extent()), i)) {
                        // TODO: remove false positive.
                    }
                }
            }
            // Filter found concepts to have at least min_size elements.
            auto it = remove_if(result.begin(), result.end(),
                                [=](const Concept& c) { return c.ExtentSize() < min_size; });
            result.erase(it, result.end());
        }
        return result;
    }
}
