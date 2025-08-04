#include <iostream>
#include "neural_network.h"
#include "training_sets/training_set_mnist.h"

int main()
{
    TrainingSetMNIST mnist;
    bool loaded = mnist.load_data("./training_sets/mnist/mnist_train.csv");
}