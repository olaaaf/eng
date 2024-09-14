#pragma once
#include "../Core/6502.hpp"
#include <cstdint>
#include <vector>

enum GameState { RUNNING, GAME_OVER };

struct States {
  std::vector<int> x;
  std::vector<uint8_t> y;
  std::vector<uint8_t> x_speed;
};

class Score {
public:
  Score();
  void frame(MedNES::CPU6502 *cpu);
  void report(int died);
  GameState check_death(MedNES::CPU6502 *cpu);

private:
  States states;
  int heighest;
};
