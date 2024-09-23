#include "Model.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <optional>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/script.h>
#include <torch/torch.h>

ButtonPresses Model::interpret_output(uint8_t *input) {
  static ButtonPresses presses;
  presses.right = true;
  return presses;
}

ButtonPresses Model::predict(uint8_t *input) { return interpret_output(input); }

std::optional<std::unique_ptr<Model>>
Model::FromFile(const std::string &model_path) {
  try {
    std::ifstream model_file("model");
    // load the module from path
    std::unique_ptr<torch::jit::script::Module> module =
        std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_file));
    // create a defaut model
    std::unique_ptr<Model> model = std::make_unique<Model>();
    // module into model :)
    model->module = std::move(module);
    return model;
  } catch (const c10::Error &e) {
    std::cerr << "Error loading the model: " << e.what() << std::endl;
    return std::nullopt;
  }
}
std::optional<std::unique_ptr<Model>>
FromWeights(const std::vector<char> &weight_data) {
  try {
    auto model = std::make_unique<Model>();

    // Create an IValue from the raw data
    auto tensor = torch::from_blob(const_cast<char *>(weight_data.data()),
                                   weight_data.size(), torch::kUInt8);
    auto ivalue = torch::jit::IValue(tensor);

    return model;
  } catch (const c10::Error &e) {
    std::cerr << "Error loading the model weights: " << e.what() << std::endl;
    return std::nullopt;
  }
}
