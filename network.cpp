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

    Neuron::Neuron(size_t inputs, double output)
      : output(output)
      , input_weights(inputs, 1)
      , last_delta_weights(inputs, 0.0)
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

    void Neuron::CalcGradient(double target) {
        gradient = LossFunctionDerivative(output, target) * ActivationFunctionDerivative(output);
    }

    void Neuron::CalcGradient(const Layer& next_layer, size_t neuron_id) {
        double tmp = 0;
        for (const auto& neuron : next_layer)
            tmp += neuron.gradient * neuron.input_weights[neuron_id];
        gradient = tmp * ActivationFunctionDerivative(output);
    }

    void Neuron::UpdateWeight(const Layer& prev_layer) {
        for (size_t i = 0; i + 1 < input_weights.size(); ++i) {
            double delta = eta * gradient * prev_layer[i].output + alpha * last_delta_weights[i];
            input_weights[i] += delta;
            last_delta_weights[i] = delta;
        }
        double delta = eta * gradient + alpha * last_delta_weights.back();
        input_weights.back() += delta;
        last_delta_weights.back() = delta;
    }

    void Neuron::UpdateWeight(const Data& input) {
        for (size_t i = 0; i + 1 < input_weights.size(); ++i) {
            double delta = eta * gradient * input[i] + alpha * last_delta_weights[i];
            input_weights[i] -= delta;
            last_delta_weights[i] = delta;
        }
        double delta = eta * gradient + alpha * last_delta_weights.back();
        input_weights.back() -= delta;
        last_delta_weights.back() = delta;
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

    Data Network::FitTransform(const Data& input, const Data& target) {
        auto output = Transform(input);
        for (size_t i = 0; i < target.size(); ++i) {
            layers.back()[i].CalcGradient(target[i]);
        }
        for (int layer_id = layers.size() - 2; layer_id >= 0; --layer_id) {
            auto& layer = layers[layer_id];
            for (size_t i = 0; i < layer.size(); ++i)
                layer[i].CalcGradient(layers[layer_id+1], i);
        }

        for (auto& neuron : layers.front())
            neuron.UpdateWeight(input);
        for (size_t i = 1; i < layers.size(); ++i)
            for (auto& neuron : layers[i])
                neuron.UpdateWeight(layers[i-1]);

        return output;
    }
}
