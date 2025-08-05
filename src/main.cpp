#include <iostream>
#include "neural_network.h"
#include "training_sets/training_set_mnist.h"

int main()
{
    TrainingSetMNIST mnist;
    bool loaded = mnist.load_data("./mnist_train.csv", 1);
    if (!loaded)
    {
        std::cout << "File not loaded. Ending process...\n";
        return -1;
    }

    NeuralNetwork nnet('[' + mnist.label + ']', {784, 256, 128, 64, 10}, 0.01);
    nnet.train(mnist.input, mnist.output);
}