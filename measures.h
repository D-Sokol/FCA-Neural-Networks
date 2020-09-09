#ifndef FCANN_MEASURES_H
#define FCANN_MEASURES_H

#include <vector>
#include "fcai/src/fca_datastructures.h"

double Support(const FCA::Concept& c);
double CVMeasure(const FCA::Concept& c, const FCA::Context& context, size_t attributes_count);
double CFCMeasure(const FCA::Concept& c, const FCA::Context& context, size_t attributes_count);

double Purity(const FCA::Concept& c, const std::vector<size_t>& targets);
#endif //FCANN_MEASURES_H
