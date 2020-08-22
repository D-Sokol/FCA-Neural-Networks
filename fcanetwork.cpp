#include <algorithm>
#include <numeric>
#include "fcanetwork.h"

using namespace std;

namespace NN {
    using namespace FCA;

    NetworkStructure GetStructure(const Lattice& lattice, const vector<size_t>& target_classes,
                                  size_t min_level, size_t max_level) {
        size_t input_size = lattice.GetConcepts().front().Intent().size();
        size_t output_size = *max_element(target_classes.begin(), target_classes.end()) + 1;
        size_t begin = lattice.GetLevelStarts().at(min_level);
        size_t end = lattice.GetLevelStarts().at(max_level);
        size_t first_layer_size = lattice.GetLevelStarts().at(min_level+1) - begin;
        size_t last_layer_size = end - lattice.GetLevelStarts().at(max_level-1);

        const auto& connections = lattice.GetConnections();
        vector<vector<size_t>> connections_shifted(connections.begin() + begin, connections.begin() + end);
        for (auto& vec : connections_shifted)
            for (auto& index : vec) {
                index += input_size;
                index -= begin;
            }

        for (size_t i = 0; i < first_layer_size; ++i) {
            connections_shifted[i].clear();
            const auto& concept_ = lattice.GetConcepts()[begin+i];
            for (size_t attr = 0; attr < input_size; ++attr) {
                if (concept_.Intent().test(attr))
                    connections_shifted[i].push_back(attr);
            }
        }
        connections_shifted.insert(connections_shifted.begin(), input_size, {});

        vector<size_t> all_connections_with_last(last_layer_size);
        iota(all_connections_with_last.begin(), all_connections_with_last.end(), end - begin - last_layer_size + input_size);
        connections_shifted.insert(connections_shifted.end(), output_size, all_connections_with_last);

        return NetworkStructure(move(connections_shifted));
    }

    FCANetwork::FCANetwork(const Lattice& lattice, const vector<size_t>& target_classes,
                           size_t max_level, size_t min_level)
      : Network(GetStructure(lattice, target_classes, min_level, max_level))
    {
    }

    FCANetwork::FCANetwork(const vector<size_t>& layer_sizes)
      : Network(NetworkStructure::FullyConnected(layer_sizes))
    {
    }

    Data FCANetwork::Transform(const BitSet& attributes) {
        Data tmp(attributes.size());
        for (size_t i = 0; i < attributes.size(); ++i)
            tmp[i] = attributes.test(i);
        return Network::Transform(tmp);
    }

    Data FCANetwork::FitTransform(const BitSet& attributes, size_t target_class) {
        Data tmp(attributes.size());
        for (size_t i = 0; i < attributes.size(); ++i)
            tmp[i] = attributes.test(i);
        Data tmp_output(Network::OutputSize());
        tmp_output[target_class] = 1;
        return Network::FitTransform(tmp, tmp_output);
    }
}
