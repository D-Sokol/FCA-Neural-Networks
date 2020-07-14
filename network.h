#ifndef FCANN_NETWORK_H
#define FCANN_NETWORK_H
#include <vector>

namespace NN {
    class Neuron;

    using Data = std::vector<double>;

    template <typename Iter>
    Data ExtractOutputs(Iter begin, Iter end) {
        static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, Neuron>);
        Data result;
        result.reserve(std::distance(begin, end));
        for (; begin != end; ++begin)
            result.push_back(begin->GetOutput());
        return result;
    }

    struct NetworkStructure {
        NetworkStructure(std::vector<std::vector<size_t>>);
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

        double FeedForward();
        inline void SetOutput(double val) { output = val; }
        inline double GetOutput() const { return output; }
        inline size_t InputConnections() const { return inputs.size(); }
        inline size_t OutputConnections() const { return outputs.size(); }

        void CalcGradient(double target);
        void CalcGradient();
        void UpdateWeight();
        void Connect();
        static double LossFunction(double out, double target);
    private:
        static double LossFunctionDerivative(double out, double target);
        static double ActivationFunction(double x);
        static double ActivationFunctionDerivative(double activation);
        static const double eta;
        static const double alpha;

        double output = 1.0;
        double gradient = 0.0;
        std::vector<Connection> inputs;
        std::vector<ReverseConnection> outputs;
    };

    class Network {
    public:
        using Structure = NetworkStructure;

        explicit Network(const Structure&);

        Data Transform(const Data&);
        Data FitTransform(const Data& input, const Data& target);

        inline size_t InputSize() const { return input_size; }
        inline size_t OutputSize() const { return output_size; }
        inline double GetRecentAverageError() const { return recent_average_error; }
    private:
        size_t input_size = 0;
        size_t output_size = 0;
        std::vector<Neuron> neurons;

        static const double smoothing_factor;
        double recent_average_error = 0;
    };
}

#endif //FCANN_NETWORK_H
