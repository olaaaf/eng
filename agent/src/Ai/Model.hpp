#pragma once

#include <cstdint> // For uint8_t
#include <memory>
#include <optional>
#include <string>
#include <torch/torch.h>

// Struct to represent button presses.
struct ButtonPresses {
  bool left;
  bool right;
  bool up;
  bool down;
  bool a;
  bool b;
  bool enter;
  bool space;

  // Initialize all buttons to false (not pressed)
  ButtonPresses()
      : left(false), right(false), up(false), down(false), a(false), b(false),
        enter(false), space(false) {}

  int to_int() const {
    return (left & 0b1000000) | (right & 0b01000000) | (up & 0b00100000) |
           (down & 0b00010000) | (a & 0b00001000) | (b & 0b00000100) |
           (enter & 0b00000010) | (space & 0x01);
  }
};

class Model {
public:
  Model() = default;
  ~Model() = default;
  Model(const Model &) = delete;
  Model &operator=(const Model &) = delete;

  static std::optional<std::unique_ptr<Model>>
  FromFile(const std::string &model_path);
  static std::optional<std::unique_ptr<Model>>
  FromStream(const std::vector<char> &stream);
  ButtonPresses predict(uint8_t *input);

private:
  std::unique_ptr<torch::jit::script::Module> module;

  ButtonPresses interpret_output(uint8_t *input);
};
