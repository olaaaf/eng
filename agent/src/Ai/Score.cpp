#include "Score.hpp"
#include "Ai/Model.hpp"
#include <cstdint>

Score::Score() : heighest(0) {}

void Score::report(int died) {}

void Score::frame(MedNES::CPU6502 *cpu, const ButtonPresses *model_output) {
  int x = (static_cast<int>(cpu->read(0x006D)) << 8) | cpu->read(0x0086);
  states.x.push_back(x);
  uint8_t y = cpu->read(0x00CE);
  states.y.push_back(y);
  uint8_t x_speed = cpu->read(0x0057);
  states.x_speed.push_back(x_speed);
  states.actions.push_back(model_output->to_int());
}

GameState Score::check_death(MedNES::CPU6502 *cpu) {
  uint8_t lives = cpu->read(0x075A);
  return (lives != 2) ? GAME_OVER : RUNNING;
}