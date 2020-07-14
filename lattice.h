#ifndef FCANN_LATTICE_H
#define FCANN_LATTICE_H

#include <vector>
#include "fcai/src/fca_datastructures.h"

namespace FCA {
    bool IsSubConcept(const Concept& lhs, const Concept& rhs);

    class Lattice {
    public:
        explicit Lattice(std::vector<Concept>);
        const auto& GetConcepts() const { return concepts; }
        const auto& GetConnections() const { return connections; }
    private:
        std::vector<Concept> concepts;
        std::vector<std::vector<size_t>> connections;
    };
}


#endif //FCANN_LATTICE_H
