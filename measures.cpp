#include <algorithm>
#include <unordered_map>
#include "measures.h"

using namespace std;

double Support(const FCA::Concept& c) {
    return static_cast<double>(c.ExtentSize()) / c.Extent().size();
}

double CVMeasure(const FCA::Concept& c, const FCA::Context& context, size_t attributes_count) {
    const auto& b = c.Intent();
    double result = 0;
    FCA::BitSet temp(context.AttrSize());
    for (size_t i = 0; i < min(context.AttrSize(), attributes_count); ++i) {
        if (b.test(i)) {
            temp.set(i);
            // May be cached for the different concepts in the real applications.
            result += 1.0 / context.DrvtAttr(temp).count();
            temp.reset(i);
        }
    }
    return result * c.ExtentSize();
}

double CFCMeasure(const FCA::Concept& c, const FCA::Context& context, size_t attributes_count) {
    if (size_t extent_size = c.ExtentSize()) {
        const auto& a = c.Extent();
        double result = 0;
        FCA::BitSet temp(context.AttrSize());
        for (size_t i = 0; i < min(context.AttrSize(), attributes_count); ++i) {
            temp.set(i);
            auto y_prime = context.DrvtAttr(temp);
            temp.reset(i);

            result += static_cast<double>((a & y_prime).count()) / y_prime.count();
        }
        return result / extent_size;
    } else {
        return 0.0;
    }
}

double Purity(const FCA::Concept& c, const std::vector<size_t>& targets) {
    unordered_map<size_t, size_t> frequencies;
    for (size_t index = 0; index < targets.size(); ++index) {
        if (c.Extent().test(index)) {
            frequencies[targets[index]]++;
        }
    }

    const size_t max_frequency = max_element(frequencies.begin(), frequencies.end(),
                                             [](const auto& p1, const auto& p2){ return p1.second < p2.second; }
                                             )->second;

    return static_cast<double>(max_frequency) / c.ExtentSize();
}
