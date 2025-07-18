#include "neural_network.h"

NeuralNetwork::NeuralNetwork(vec<int> topology, double learningRate)
{
    this->learningRate = learningRate;
    for (int i = 0; i < topology.size(); ++i)
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