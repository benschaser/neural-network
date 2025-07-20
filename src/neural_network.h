#include <Eigen>
#include <iostream>
#include <vector>

using RowVector = Eigen::RowVectorXf;
using ColVector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;
template <typename T>
using vec = std::vector<T>;
using ulong = unsigned long;

const double activationFunction(double x) {
    return tanhf(x);
} 
const double activationFunctionDerivative(double x) {
    return 1.0 - tanhf(x) * tanhf(x);
}

class NeuralNetwork
{
public:
    NeuralNetwork(std::string dataset_label, vec<int> topology, double learningRate = 0.005);

    void propogateForward(RowVector &input);
    void propogateBackward(RowVector &output);
    void calcErrors(RowVector &output);
    void updateWeights();
    void train(vec<RowVector *> input, vec<RowVector*> output);
    std::string get_env_label();

    std::string dataset_label = "default";
    vec<int> topology;
    vec<RowVector *> neuronLayers;
    vec<RowVector *> cacheLayers;
    vec<RowVector *> deltas;
    vec<Matrix *> weights;
    double learningRate;
    std::function<double(double)> activationFunction = activationFunction;
    std::function<double(double)> activationFunctionDerivative = activationFunctionDerivative;
};