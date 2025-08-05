#pragma once
#include "aliases.h"

const double activationFunctionX(double x);
const double activationFunctionDerivativeX(double x);

// define softmax function to apply to output for proper distribution

class NeuralNetwork
{
public:
    NeuralNetwork(std::string cli_label, vec<int> topology, double learningRate = 0.005);

    void propogateForward(RowVector &input);
    void propogateBackward(RowVector &output);
    void calcErrors(RowVector &output);
    void updateWeights();
    void train(vec<RowVector *> input, vec<RowVector *> output);

    std::string cli_label = "[default]";
    vec<int> topology;
    vec<RowVector *> neuronLayers;
    vec<RowVector *> cacheLayers;
    vec<RowVector *> deltas;
    vec<Matrix *> weights;
    double learningRate;
    std::function<double(double)> activationFunction = activationFunctionX;
    std::function<double(double)> activationFunctionDerivative = activationFunctionDerivativeX;
};