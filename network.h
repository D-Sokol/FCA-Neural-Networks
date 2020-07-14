#ifndef FCANN_NETWORK_H
#define FCANN_NETWORK_H
#include <optional>
#include <vector>

namespace NN {
    class Neuron;

    using Layer = std::vector<Neuron>;
    using Data = std::vector<double>;

    Data ExtractOutputs(const Layer&);

    struct NetworkStructure {
        std::vector<std::vector<size_t>> connections;
        size_t size() const { return connections.size(); }
    };

    struct Connection {
        Neuron* source = nullptr;
        double weight = 1.0;
        double delta_weight = 0.0;
    };
    struct ReverseConnection {
        Neuron* destination = nullptr;
        double* weight = nullptr;
    };

    class Neuron {
    public:
        explicit Neuron(double output = 1.0);
        Neuron(std::vector<Neuron>& neurons, const std::vector<size_t>& input_numbers);

        double FeedForward() const;
        inline void SetOutput(double val) const { output = val; }
        inline double GetOutput() const { return output; }
        inline size_t InputConnections() const { return inputs.size(); }
        inline size_t OutputConnections() const { return outputs.size(); }

        void CalcGradient(double target);
        void CalcGradient();
        void UpdateWeight();
        void Connect();
    private:
        static double LossFunction(double out, double target);
        static double LossFunctionDerivative(double out, double target);
        static double ActivationFunction(double x);
        static double ActivationFunctionDerivative(double activation);
        static const double eta;
        static const double alpha;

        mutable double output = 1.0;
        double gradient = 0.0;
        std::vector<Connection> inputs;
        std::vector<ReverseConnection> outputs;
    };

    class Network {
    public:
        using Structure = NetworkStructure;

        explicit Network(const Structure&);

        Data Transform(const Data&) const;
        Data FitTransform(const Data& input, const Data& target);
    private:
        size_t input_size;
        size_t output_size;
        std::vector<Neuron> neurons;
    };
}

#endif //FCANN_NETWORK_H
