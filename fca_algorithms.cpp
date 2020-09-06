#include <algorithm>
#include <string>
#include "fca_algorithms.h"

using namespace std;

namespace FCA {
    vector<Concept> ThetaSophia(const Context& context, Predicate keep_concept) {
        // Concept candidate with label that shows if the first one is proper concept.
        using MarkedConcept = pair<Concept, bool>;
        // Projection phi_{-1} produces only one trivial concept,
        //  that contains all objects and no attributes.
        vector<MarkedConcept> result = {{{context.ObjSize(), context.AttrSize()}, true}};
        result.front().first.Extent().flip();

        for (size_t added_attr_id = 0; added_attr_id < context.AttrSize(); ++added_attr_id) {
            // Consider projection phi_i that consider attributes from 0 to i-th, inclusively.
            const size_t concepts_number = result.size();
            for (size_t k = 0; k < concepts_number; ++k) {
                {
                    // Concept (A, B) should be kept in the result, if the set A_{-} is not empty
                    // A_{-} = {a \in A | a does not possess attribute i}
                    auto& concept_ = result[k];
                    auto ext = context.Extent(added_attr_id);
                    ext.flip();
                    ext &= concept_.first.Extent();
                    if (ext.any())
                        result.push_back(concept_);
                }
                {
                    // (A, B) can produces a preimage (A_{+}, B \sup {i}),
                    // where A_{+} = {a \in A | a possess attribute i}
                    auto& concept_ = result[k];  // Reference from the previous block may be invalidated.

                    concept_.first.Extent() &= context.Extent(added_attr_id);
                    concept_.first.Intent().set(added_attr_id);
                    if (!concept_.first.Intent().is_prefix_equal(context.DrvtObj(concept_.first.Extent()), added_attr_id)) {
                        concept_.second = false;
                    }
                }
            }
            auto it = remove_if(result.begin(), result.end(),
                                [=](const MarkedConcept& c) { return !c.second || !keep_concept(c.first, added_attr_id+1); });
            result.erase(it, result.end());
        }

        vector<Concept> return_value;
        return_value.reserve(result.size());
        for (auto& [concept_, label] : result)
            return_value.push_back(move(concept_));
        return return_value;
    }

    pair<Context, vector<size_t>> ReadContext(istream& is) {
        constexpr const auto ignore_length = numeric_limits<streamsize>::max();
        size_t extent_size, intent_size;
        is >> extent_size >> intent_size;
        is.ignore(ignore_length, '\n');
        vector<vector<bool>> data(extent_size, vector<bool>(intent_size));
        vector<size_t> targets;
        targets.reserve(extent_size);
        for (auto& row : data) {
            // Specialized version of vector<bool> does not allow to simplify this cycle.
            for (size_t i = 0; i < row.size(); ++i) {
                bool tmp;
                is >> tmp;
                row[i] = tmp;
                is.ignore(ignore_length, ',');
            }
            targets.emplace_back();
            is >> targets.back();
        }
        return {Context(data), move(targets)};
    }
}
