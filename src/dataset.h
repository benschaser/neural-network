#pragma once
#include "aliases.h"
#include <any>


class Dataset {
public:
    Dataset(std::string label)
        : label(label) {}
    Dataset(std::string label, vec<RowVector*> input, vec<RowVector*> output)
        : label(label), input(input), output(output) {}

    virtual bool load_data(std::string arg) = 0;
    std::string get_env_label() { return "[" + label + "] "; }

    std::string label;
    vec<RowVector*> input;
    vec<RowVector*> output;
};