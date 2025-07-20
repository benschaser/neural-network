#include "neural_network.h"

NeuralNetwork::NeuralNetwork(std::string cli_label, vec<int> topology, double learningRate)
{
    this->cli_label = cli_label;
    this->topology = topology;
    this->learningRate = learningRate;
    for (ulong i = 0; i < topology.size(); ++i)
    {
        // init neuron layers
        if (i == topology.size() - 1)
            neuronLayers.push_back(new RowVector(topology[i]));
        else
            neuronLayers.push_back(new RowVector(topology[i] + 1));

        // init cache and deltas
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        if (i != topology.size() - 1)
        {
            neuronLayers.back()->coeffRef(topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(topology[i]) = 1.0;
        }

        // init weights
        if (i > 0)
        {
            if (i != topology.size() - 1)
            {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(topology[i]).setZero();
                weights.back()->coeffRef(topology[i - 1], topology[i]) = 1.0;
            }
            else
            {
                weights.push_back(new Matrix(topology[i - 1] + 1, topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::propogateForward(RowVector& input) {
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;
    for (ulong i = 1; i < topology.size(); ++i) {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        neuronLayers[i]->block(0, 0, 1, topology[i]).unaryExpr(activationFunction);
    }
}

void NeuralNetwork::propogateBackward(RowVector& output) {
    calcErrors(output);
    updateWeights();
}

void NeuralNetwork::calcErrors(RowVector& output) {
    (*deltas.back()) = output - (*neuronLayers.back());

    for (ulong i = 0; i < topology.size() - 1; ++i) {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
    }
}

void NeuralNetwork::updateWeights() {
    for (ulong i = 0; i < topology.size() - 1; ++i) {
        if (i != topology.size() - 2) {
            for (ulong c = 0; c < weights[i]->cols() - 1; ++c) {
                for (ulong r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else {
            for (ulong c = 0; c < weights[i]->cols(); ++c) {
                for (ulong r = 0; r < weights[i]->rows(); r++) {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::train(vec<RowVector*> input, vec<RowVector*> output) {
    for (ulong i = 0; i < input.size(); ++i) {
        std::cout << cli_label << " Training on input: " << *input[i] << '/' << input.size() << '\n';
        propogateForward(*input[i]);
        std::cout << cli_label << " \33[33mExpected value: " << *output[i] << "\33[0m\n";
        if (*output[i] == *neuronLayers.back()) {
            std::cout << cli_label << " \33[32mOutput value: " << *neuronLayers.back() << "\33[0m\n";
        }
        else {
            std::cout << cli_label << " \33[31mOutput value: " << *neuronLayers.back() << "\33[0m\n";
        }
        propogateBackward(*output[i]);
        std::cout << cli_label << "MSE: " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << '\n';
    }
 }