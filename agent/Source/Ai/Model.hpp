#pragma once

#include <vector>
#include <cstdint> // For uint8_t
#include <string>

// Struct to represent button presses.
struct ButtonPresses {
    bool left;
    bool right;
    bool up;
    bool down;
    bool a;
    bool b;

    // Initialize all buttons to false (not pressed)
    ButtonPresses() : left(false), right(false), up(false), down(false), a(false), b(false) {}
};

class Model {
public:
    // Constructor and Destructor
    Model();
    ~Model();

    // Disable copy and assignment.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Function to load the model from a model registry.
    // The details of this function will depend on how your model registry is structured.
    bool loadModel(const std::string& modelPath);

    // Function to process the input (uin8_t buffer) and predict the button presses.
    // The buffer is expected to be a processed image/frame from the emulator.
    ButtonPresses predict(const std::vector<uint8_t>& inputBuffer);

private:
    // Private member variables for the model.
    // Depending on the ML framework, you might have a model object here.
    // For example, for TensorFlow:
    // std::unique_ptr<tensorflow::SavedModelBundle> model;

    // Helper functions for processing the model's output.
    ButtonPresses interpretOutput(/* Output data structure from your ML model */);
};
