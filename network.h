#ifndef FCANN_NETWORK_H
#define FCANN_NETWORK_H
#include <vector>

namespace NN {
    class Neuron;

    using Layer = std::vector<Neuron>;
    using Data = std::vector<double>;

    Data ExtractOutputs(const Layer&);

    class Neuron {
    public:
        explicit Neuron(size_t inputs, double output = 0.0);
        double FeedForward(const Layer&) const;
        double FeedForward(const Data&) const;
        inline double GetOutput() const { return output; }
    private:
        static double ActivationFunction(double x);
        static double ActivationFunctionDerivative(double activation);

        mutable double output;
        std::vector<double> input_weights;
    };

    class Network {
    public:
        using Structure = std::vector<size_t>;

        explicit Network(const Structure&);

        Data Transform(const Data&) const;
        Data FitTransform(const Data&);
    private:
        const size_t input_size;
        std::vector<Layer> layers;
    };
}

#endif //FCANN_NETWORK_H
