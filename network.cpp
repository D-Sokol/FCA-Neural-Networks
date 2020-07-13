#include <cassert>
#include <cmath>
#include "network.h"

template <typename T>
inline T sqr(T x) {
    return x * x;
}

namespace NN {
    Data ExtractOutputs(const Layer& layer) {
        Data data(layer.size());
        for (size_t i = 0; i < layer.size(); ++i)
            data[i] = layer[i].GetOutput();
        return data;
    }

    NetworkStructure::NetworkStructure(std::vector<size_t> layers_size_)
      : layers_size(move(layers_size_)), connections(layers_size.size())
    {
        for (size_t i = 1; i < layers_size.size(); ++i)
            connections[i].resize(layers_size[i]);
    }

    Neuron::Neuron(double output)
      : output(output)
    {
    }

    Neuron::Neuron(Layer& prev_layer)
      : inputs(prev_layer.size())
    {
        // TODO: weights generation.
        for (size_t n = 0; n < prev_layer.size(); ++n)
            inputs[n].source = &prev_layer[n];
    }

    Neuron::Neuron(Layer& prev_layer, const std::vector<size_t>& input_numbers)
      : inputs(input_numbers.size())
    {
        // TODO: weights generation.
        for (size_t n = 0; n < input_numbers.size(); ++n)
            inputs[n].source = &prev_layer[input_numbers[n]];
    }

    double Neuron::FeedForward() const {
        output = 0;
        for (const auto& input : inputs) {
            output += input.source->output * input.weight;
        }
        return (output = ActivationFunction(output));
    }

    void Neuron::CalcGradient(double target) {
        gradient = LossFunctionDerivative(output, target) * ActivationFunctionDerivative(output);
    }

    void Neuron::CalcGradient() {
        assert(!inputs.empty());
        double tmp = 0;
        for (const auto& connection : outputs)
            tmp += connection.destination->gradient * (*connection.weight);
        gradient = tmp * ActivationFunctionDerivative(output);
    }

    void Neuron::UpdateWeight() {
        for (auto& input : inputs) {
            double delta = eta * gradient * input.source->output + alpha * input.delta_weight;
            input.weight += delta;
            input.delta_weight = delta;
        }
    }

    void Neuron::Connect() {
        for (auto& connection : inputs) {
            connection.source->outputs.push_back({this, &connection.weight});
        }
    }

    double Neuron::LossFunction(double out, double target) {
        return sqr(target - out);
    }

    double Neuron::LossFunctionDerivative(double out, double target) {
        return target - out;
    }

    double Neuron::ActivationFunction(double x) {
        return tanh(x);
    }

    double Neuron::ActivationFunctionDerivative(double activation) {
        return 1.0 - activation * activation;
    }

    const double Neuron::eta = 0.15;
    const double Neuron::alpha = 0.5;

    Network::Network(const Network::Structure& structure)
      : input_size(structure.layers_size[0])
    {
        layers.reserve(structure.size());
        for (size_t i = 0; i < structure.size(); ++i) {
            if (i == 0) {
                layers.emplace_back(input_size);
            } else {
                auto& prev_layer = layers.back();
                layers.emplace_back();
                layers.back().reserve(structure.layers_size[i] + 1u);
                assert(structure.connections[i].size() == structure.layers_size[i]);
                for (const auto& connections : structure.connections[i])
                    if (connections.has_value())
                        layers.back().emplace_back(prev_layer, *connections);
                    else
                        layers.back().emplace_back(prev_layer);
            }
            // Bias neuron with output value 1.
            if (i + 1u != structure.size())
                layers.back().emplace_back(1.0);
        }

        for (auto& layer : layers)
            for (auto& neuron : layer)
                neuron.Connect();
    }

    Data Network::Transform(const Data& data) const {
        assert(data.size() == input_size);
        for (size_t n = 0; n < input_size; ++n)
            layers.front()[n].SetOutput(data[n]);

        for (size_t i = 1; i < layers.size(); ++i) {
            bool hidden_layer = (i + 1u != layers.size());
            for (size_t n = 0; n + hidden_layer < layers[i].size(); ++n) {
                layers[i][n].FeedForward();
            }
        }
        return ExtractOutputs(layers.back());
    }

    Data Network::FitTransform(const Data& input, const Data& target) {
        assert(target.size() == layers.back().size());
        auto output = Transform(input);
        for (size_t i = 0; i < target.size(); ++i) {
            layers.back()[i].CalcGradient(target[i]);
        }
        for (size_t layer_id = layers.size() - 2; layer_id > 0; --layer_id) {
            auto& layer = layers[layer_id];
            for (size_t i = 0; i + 1u < layer.size(); ++i)
                layer[i].CalcGradient();
        }

        for (size_t i = 1; i < layers.size(); ++i) {
            bool hidden_layer = (i + 1u != layers.size());
            for (size_t n = 0; n + hidden_layer < layers[i].size(); ++n)
                layers[i][n].UpdateWeight();
        }

        return output;
    }
}
