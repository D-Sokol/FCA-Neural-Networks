#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include "network.h"

template <typename T>
inline T sqr(T x) {
    return x * x;
}

namespace NN {
    NetworkStructure::NetworkStructure(std::vector<std::vector<size_t>> connections_)
      : connections(move(connections_))
    {
        for (size_t i = 0; i < connections.size(); ++i)
            for (const auto ref : connections[i])
                if (ref >= i)
                    throw std::invalid_argument("Neurons dependence order violation");
    }

    Neuron::Neuron(double output)
      : output(output)
    {
    }

    Neuron::Neuron(std::vector<Neuron>& neurons, const std::vector<size_t>& connections)
      : inputs(connections.size())
    {
        static std::mt19937 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        static std::uniform_int_distribution<int8_t> gen;
        const double xavier_multiplier = sqrt(3.0 / connections.size());
        for (size_t n = 0; n < connections.size(); ++n) {
            inputs[n].source = &neurons[connections[n]];
            // Random variable having uniform distribution U[-1, +1]
            double u = (static_cast<double>(gen(generator)) + 0.5) / 127.5;
            // Xavier initialization: weight is a random variable from
            //  a distribution with zero mean and a variance V w = 1/N.
            inputs[n].weight = u * xavier_multiplier;
        }
    }

    double Neuron::FeedForward() {
        if (inputs.empty())
            return output;
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
      : input_size(structure.size()), neurons(structure.size(), Neuron(1.0))
    {
        for (size_t i = 0; i < neurons.size(); ++i) {
            const auto& inputs = structure.connections[i];
            if (inputs.empty())
                continue;
            neurons[i] = Neuron(neurons, inputs);
            input_size = std::min(input_size, i);
        }

        for (auto& neuron : neurons)
            neuron.Connect();
        for (auto it = neurons.rbegin(); it != neurons.rend(); ++it)
            if (it->OutputConnections() == 0u)
                ++output_size;
            else
                break;
    }

    Data Network::Transform(const Data& data) {
        assert(data.size() == input_size);
        for (size_t n = 0; n < input_size; ++n)
            neurons[n].SetOutput(data[n]);

        for (size_t n = input_size; n < neurons.size(); ++n) {
            neurons[n].FeedForward();
        }
        return ExtractOutputs(neurons.end() - output_size, neurons.end());
    }

    Data Network::FitTransform(const Data& input, const Data& target) {
        assert(target.size() == output_size);
        auto output = Transform(input);
        recent_average_error = 0.0;
        for (size_t i = 0; i < output_size; ++i) {
            auto& neuron = neurons[neurons.size()-output_size+i];
            neuron.CalcGradient(target[i]);
            recent_average_error += Neuron::LossFunction(neuron.GetOutput(), target[i]);
        }
        recent_average_error = sqrt(recent_average_error / (output_size - 1));

        for (size_t n = 0; n < neurons.size()-output_size; ++n)
            neurons[n].CalcGradient();

        for (auto& neuron : neurons)
            neuron.UpdateWeight();

        return output;
    }
}
