#include <Eigen>
#include <iostream>
#include <vector>

using RowVector = Eigen::RowVectorXf;
using ColVector = Eigen::VectorXf;
using Matrix = Eigen::MatrixXf;
template <typename T>
using vec = std::vector<T>;

class NeuralNetwork
{
public:
    NeuralNetwork(vec<int> topology, double learningRate = 0.005);

    void propogateForward(RowVector &input);
    void propogateBackward(RowVector &output);
    void calcErrors(RowVector &output);
    void updateWeights();
    void train(vec<RowVector *> data);

    vec<RowVector *> neuronLayers;
    vec<RowVector *> cacheLayers;
    vec<RowVector *> deltas;
    vec<Matrix *> weights;
    double learningRate;
};