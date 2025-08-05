#pragma once
#include "aliases.h"
#include "dataset.h"

class TrainingSetMNIST : public Dataset
{
public:
    TrainingSetMNIST()
        : Dataset("mnist") {}

    bool load_data(std::string arg, int max_count) override;
};
