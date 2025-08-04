#include "training_set_mnist.h"
#include "aliases.h"
#include <fstream>
#include <sstream>
/**
 * @brief Gets the total line count of the file.
 *
 * This function takes in a filepath and returns the line count.
 *
 * @param filepath The absolute filepath represented as a string.
 * @return Line count. If -1 is returned, the file could not be found or opened.
 */
int count_lines(const std::string filepath)
{
    std::ifstream file(filepath);
    if (!file.is_open())
        return -1;

    int count = 0;
    std::string line;
    while (std::getline(file, line))
        ++count;

    return count;
}

bool TrainingSetMNIST::load_data(std::string arg)
{
    std::cout << "Loading MNIST data from: " << arg << std::endl;
    int line_count = count_lines(arg);
    if (line_count == -1)
        return false;

    std::ifstream file(arg);
    if (!file.is_open())
        return false;

    std::string line;
    int count = 0;

    while (std::getline(file, line))
    {
        if (count >= 100000)
            break; // shouldn't go over 100,000 samples

        std::stringstream ss(line);
        std::string cell;

        // read label
        std::getline(ss, cell, ',');
        int label = std::stoi(cell);
        // add label to expected output;
        RowVector outRow(10);
        if (label < 10 && label >= 0)
            outRow[label] = 1.0;

        // read pixels (784 count)
        RowVector inRow(784);
        for (int i = 0; i < 784; ++i)
        {
            if (!std::getline(ss, cell, ','))
                throw std::runtime_error("Malformed CSV line: " + line);

            inRow(i) = std::stoi(cell) / 255.0; // normalized
        }
        input.push_back(new RowVector(inRow));
        output.push_back(new RowVector(outRow));
        ++count;
        std::cout << "\rLoaded " << count << '/' << line_count << std::flush;
    }
    std::cout << "Done\n";

    return true;
}
