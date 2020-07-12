#include <cassert>
#include <cmath>
#include "network.h"

namespace NN {
    Data ExtractOutputs(const Layer& layer) {
        Data data(layer.size());
        for (size_t i = 0; i < layer.size(); ++i)
            data[i] = layer[i].GetOutput();
        return data;
    }

    Neuron::Neuron(size_t inputs, double output)
      : output(output), input_weights(inputs, 1)
    {
        // TODO: weights generation.
    }

    double Neuron::FeedForward(const Layer & layer) const {
        assert(layer.size() + 1u == input_weights.size());
        return FeedForward(ExtractOutputs(layer));
    }

    double Neuron::FeedForward(const Data& data) const {
        assert(data.size() + 1u == input_weights.size());
        output = 0;
        for (size_t i = 0; i < data.size(); ++i) {
            output += data[i] * input_weights[i];
        }
        output += input_weights.back();
        return (output = ActivationFunction(output));
    }

    double Neuron::ActivationFunction(double x) {
        return tanh(x);
    }

    double Neuron::ActivationFunctionDerivative(double activation) {
        return 1.0 - activation * activation;
    }

    Network::Network(const Network::Structure& structure)
      : input_size(structure[0])
    {
        for (size_t i = 1; i < structure.size(); ++i) {
            layers.emplace_back(
                structure[i],
                Neuron(structure[i-1] + 1)
            );
        }
    }

    Data Network::Transform(const Data& data) const {
        assert(data.size() == input_size);
        for (const auto& neuron : layers.front())
            neuron.FeedForward(data);
        for (size_t i = 1; i < layers.size(); ++i)
            for (const auto& neuron : layers[i]) {
                neuron.FeedForward(layers[i-1]);
            }
        return ExtractOutputs(layers.back());
    }

    Data Network::FitTransform(const Data& input) {
        auto output = Transform(input);
        // TODO: backpropagation
        return output;
    }
}
