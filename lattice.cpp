#include <algorithm>
#include <unordered_set>
#include "lattice.h"

using namespace std;

namespace FCA {
    bool IsSubConcept(const Concept& lhs, const Concept& rhs) {
        return lhs.Extent().is_subset_of(rhs.Extent());
    }

    Lattice::Lattice(std::vector<Concept> concept_list)
      : concepts(move(concept_list)), connections(concepts.size())
    {
        sort(concepts.begin(), concepts.end(),
            [](const Concept& lhs, const Concept& rhs){
                return lhs.ExtentSize() > rhs.ExtentSize();
            }
        );
        for (size_t i = 1; i < concepts.size(); ++i) {
            unordered_set<int> ancestors;
            for (int j = i-1; j > -1; --j) {
                if (ancestors.count(j)) {
                    ancestors.insert(connections[j].begin(), connections[j].end());
                } else if (IsSubConcept(concepts[i], concepts[j])) {
                    ancestors.insert(connections[j].begin(), connections[j].end());
                    connections[i].push_back(j);
                }
            }
        }
    }
}
