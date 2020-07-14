#include <algorithm>
#include <unordered_set>
#include "lattice.h"

using namespace std;

namespace FCA {
    bool IsSubConcept(const Concept& lhs, const Concept& rhs) {
        return lhs.Extent().is_subset_of(rhs.Extent());
    }

    Lattice::Lattice(std::vector<Concept> concept_list)
    {
        sort(concept_list.begin(), concept_list.end(),
            [](const Concept& lhs, const Concept& rhs){
                return lhs.ExtentSize() > rhs.ExtentSize();
            }
        );

        vector<pair<size_t, size_t>> levels_in_list(concept_list.size());
        vector<vector<size_t>> unordered_connections(concept_list.size());

        for (size_t i = 1; i < concept_list.size(); ++i) {
            unordered_set<size_t> ancestors;
            levels_in_list[i].second = i;
            for (int j = i-1; j > -1; --j) {
                if (ancestors.count(j)) {
                    ancestors.insert(unordered_connections[j].begin(), unordered_connections[j].end());
                } else if (IsSubConcept(concept_list[i], concept_list[j])) {
                    ancestors.insert(unordered_connections[j].begin(), unordered_connections[j].end());
                    levels_in_list[i].first = max(levels_in_list[i].first, levels_in_list[j].first + 1);
                    unordered_connections[i].push_back(j);
                }
            }
        }

        sort(levels_in_list.begin(), levels_in_list.end());
        vector<size_t> new_order(levels_in_list.size());
        {
            size_t old_id = 0;
            for (auto[level, origin_id] : levels_in_list)
                new_order[origin_id] = old_id++;
        }

        concepts.reserve(concept_list.size());
        connections.reserve(unordered_connections.size());
        for (auto [level, origin_id] : levels_in_list) {
            concepts.emplace_back(move(concept_list[origin_id]));
            connections.emplace_back();
            connections.back().reserve(unordered_connections[origin_id].size());
            for (auto old_id : unordered_connections[origin_id]) {
                connections.back().push_back(new_order[old_id]);
            }
        }
    }
}
