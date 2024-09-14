#include "Model.hpp"
#include <cstdint>
#include <iostream>
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
    // load the module from path
    std::unique_ptr<torch::jit::script::Module> module =
        std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_path));
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
Model::FromStream(const std::vector<char> &stream) {
  try {
    // load the module from path
    std::unique_ptr<torch::jit::script::Module> module =
        std::make_unique<torch::jit::script::Module>(
            torch::jit::pickle_load(stream));
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