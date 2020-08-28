#include "measures.h"

double Support(const FCA::Concept& c) {
    return static_cast<double>(c.ExtentSize()) / c.Extent().size();
}

double CVMeasure(const FCA::Concept& c, const FCA::Context& context) {
    const auto& b = c.Intent();
    double result = 0;
    FCA::BitSet temp(context.AttrSize());
    for (size_t i = 0; i < context.AttrSize(); ++i) {
        if (b.test(i)) {
            temp.set(i);
            // May be cached for the different concepts in the real applications.
            result += context.DrvtAttr(temp).count();
            temp.reset(i);
        }
    }
    return result * c.ExtentSize();
}

double CFCMeasure(const FCA::Concept& c, const FCA::Context& context) {
    const auto& a = c.Extent();
    double result = 0;
    FCA::BitSet temp(context.AttrSize());
    for (size_t i = 0; i < context.AttrSize(); ++i) {
        temp.set(i);
        auto y_prime = context.DrvtAttr(temp);
        temp.reset(i);

        result += (a & y_prime).count() / y_prime.count();
    }
    return result / c.ExtentSize();
}
