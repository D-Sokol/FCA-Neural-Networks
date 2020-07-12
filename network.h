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

        void CalcGradient(double target);
        void CalcGradient(const Layer& next_layer, size_t neuron_id);
        void UpdateWeight(const Layer& prev_layer);
        void UpdateWeight(const Data&);
    private:
        static double LossFunction(double out, double target);
        static double LossFunctionDerivative(double out, double target);
        static double ActivationFunction(double x);
        static double ActivationFunctionDerivative(double activation);
        static const double eta;
        static const double alpha;

        mutable double output;
        double gradient;
        std::vector<double> input_weights;
        std::vector<double> last_delta_weights;
    };

    class Network {
    public:
        using Structure = std::vector<size_t>;

        explicit Network(const Structure&);

        Data Transform(const Data&) const;
        Data FitTransform(const Data& input, const Data& target);
    private:
        const size_t input_size;
        std::vector<Layer> layers;
    };
}

#endif //FCANN_NETWORK_H
