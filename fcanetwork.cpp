#include <algorithm>
#include <iostream>  // Used for debug messages only
#include <numeric>
#include "fcanetwork.h"
#include "measures.h"

using namespace std;

namespace NN {
    using namespace FCA;

    NetworkStructure FCANetwork::GetStructure(const Lattice& lattice, const vector<size_t>& target_classes,
                                              size_t min_level, size_t max_level) {
        max_level = min(max_level, lattice.GetLevelStarts().size()-1);

        size_t input_size = lattice.GetConcepts().front().Intent().size();
        size_t output_size = *max_element(target_classes.begin(), target_classes.end()) + 1;
        size_t begin = lattice.GetLevelStarts().at(min_level);
        size_t end = lattice.GetLevelStarts().at(max_level);
        size_t first_layer_size = lattice.GetLevelStarts().at(min_level+1) - begin;
        size_t last_layer_unfiltered_size = (
                max_level == lattice.GetLevelStarts().size()-1
                ? lattice.GetConcepts().size()
                : lattice.GetLevelStarts().at(max_level+1)
            ) - end;

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

        const auto last_layer_size = (last_layer_unfiltered_size + 1) / 2;
        {
            vector<size_t> last_layer_indexes(last_layer_unfiltered_size);
            iota(last_layer_indexes.begin(), last_layer_indexes.end(), end);
            auto middle_it = last_layer_indexes.begin() + last_layer_size;
            partial_sort(last_layer_indexes.begin(), middle_it,
                         last_layer_indexes.end(),
                         [&concepts=lattice.GetConcepts(),&targets=target_classes](size_t i, size_t j){
                            return Purity(concepts[i], targets) > Purity(concepts[j], targets);
                         });
            for (auto it = last_layer_indexes.begin(); it != middle_it; ++it) {
                connections_shifted.push_back(connections[*it]);
                for (auto& index : connections_shifted.back()) {
                    index += input_size;
                    index -= begin;
                }
            }
        }

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

    FCANetwork::FCANetwork(const NetworkStructure& structure)
      : Network(structure)
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

    vector<double> CrossValidationAccuraciesImpl(const FCA::Context& context, function<FCANetwork(size_t, size_t)> generator,
                                                 const vector<size_t>& targets, size_t iter_limit, size_t split_number) {
        const auto objects = context.ObjSize();
        vector<size_t> shuffled_indices(objects);
        iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
        random_shuffle(shuffled_indices.begin(), shuffled_indices.end());

        vector<double> result(split_number);

        for (size_t split = 0; split < split_number; ++split) {
            FCANetwork network = generator(split, split_number);

            auto test_begin = shuffled_indices.begin() + objects * split / split_number;
            auto test_end = shuffled_indices.begin() + objects * (split+1) / split_number;

            const size_t EPOCH_ACCURACY_DECREASES_TO_EXIT = 5;
            size_t epoch_accuracy_decreases = 0;
            size_t last_correct_answers = 0;
            for (size_t epoch = 1; epoch <= iter_limit; ++epoch) {
                for (auto it = shuffled_indices.begin(); it != test_begin; ++it)
                    network.FitTransform(context.Intent(*it), targets[*it]);
                for (auto it = test_end; it != shuffled_indices.end(); ++it)
                    network.FitTransform(context.Intent(*it), targets[*it]);

                size_t corrects = 0;
                for (auto it = test_begin; it != test_end; ++it) {
                    auto vec = network.Transform(context.Intent(*it));
                    size_t y_pred = max_element(vec.begin(), vec.end()) - vec.begin();
                    if (targets[*it] == y_pred)
                        ++corrects;
                }

                cerr << "Accuracy after " << epoch << " iterations: "
                     << 100.0 * static_cast<double>(corrects) / (test_end - test_begin) << '%' << endl;

                if (corrects > last_correct_answers) {
                    epoch_accuracy_decreases = 0;
                } else if (epoch_accuracy_decreases++ >= EPOCH_ACCURACY_DECREASES_TO_EXIT) {
                    result[split] = static_cast<double>(corrects) / (test_end - test_begin);
                    break;  // go to the next split
                }
                last_correct_answers = corrects;
            }
        }

        return result;
    }

    vector<double> CrossValidationAccuracies(const NetworkStructure& structure, const FCA::Context& context,
                                             const std::vector<size_t>& targets, size_t iter_limit, size_t split_number) {
        auto generator = [&](size_t, size_t){ return FCANetwork(structure);};
        return CrossValidationAccuraciesImpl(context, generator, targets, iter_limit, split_number);
    }

    vector<double> CrossValidationAccuracies(const FCA::Context& context, const vector<size_t>& targets,
                                             FCA::Predicate predicate, size_t max_level,
                                             size_t iter_limit, size_t split_number) {
        auto generator = [&](size_t split, size_t split_number){
            const auto objects = context.ObjSize();
            vector<bool> mask(objects, true);
            vector<size_t> marked_targets;
            {
                const size_t begin = objects * split / split_number;
                const size_t end = objects * (split+1) / split_number;
                for (size_t i = begin; i < end; ++i)
                    mask[i] = false;
                marked_targets.insert(marked_targets.end(), targets.begin(), targets.begin() + begin);
                marked_targets.insert(marked_targets.end(), targets.begin() + end, targets.end());
            }
            return FCANetwork(FCA::Lattice(ThetaSophia(context, predicate, move(mask))), marked_targets, max_level);
        };
        return CrossValidationAccuraciesImpl(context, generator, targets, iter_limit, split_number);
    }
}
