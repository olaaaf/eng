#pragma once
#include "../Core/6502.hpp"
#include "Ai/Model.hpp"
#include <cstdint>
#include <vector>

enum GameState { RUNNING, GAME_OVER };

struct States {
  std::vector<int> x;
  std::vector<uint8_t> y;
  std::vector<uint8_t> x_speed;
  std::vector<int> actions;
};

struct Score {
  Score();
  void frame(MedNES::CPU6502 *cpu, const ButtonPresses *model_output);
  void report(int died);
  GameState check_death(MedNES::CPU6502 *cpu);
  States states;
  int heighest;
};
