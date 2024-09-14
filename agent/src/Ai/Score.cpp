#include "Score.hpp"
#include <cstdint>
#include <iostream>

Score::Score() : heighest(0) {}

void Score::report() {}

GameState Score::frame(uint8_t horizontal, uint8_t screen, uint8_t lives) {
  int pos = (static_cast<int>(horizontal) << 8) | screen;
  heighest = (pos > heighest) ? pos : heighest;
  return (lives != 2) ? GAME_OVER : RUNNING;
}
