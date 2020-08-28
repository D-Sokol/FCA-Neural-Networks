#ifndef FCANN_MEASURES_H
#define FCANN_MEASURES_H

#include "fcai/src/fca_datastructures.h"

double Support(const FCA::Concept& c);
double CVMeasure(const FCA::Concept& c, const FCA::Context& context);
double CFCMeasure(const FCA::Concept& c, const FCA::Context& context);
#endif //FCANN_MEASURES_H
