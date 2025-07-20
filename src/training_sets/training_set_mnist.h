#pragma once
#include "aliases.h"
#include "dataset.h"

class TrainingSetMNIST : public Dataset {
public:
    TrainingSetMNIST(std::string label)
        : Dataset(label) {}

    bool load_data(std::string arg) override {
        std::cout << "Loading MNIST data from: " << arg << std::endl;

        
        return true;
    }
};
