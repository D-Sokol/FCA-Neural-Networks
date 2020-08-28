#include "measures.h"

double Support(const FCA::Concept& c) {
    return static_cast<double>(c.ExtentSize()) / c.Extent().size();
}
