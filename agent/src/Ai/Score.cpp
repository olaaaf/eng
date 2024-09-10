#include "Score.hpp"
#include <cstdint>

void Score::frame(uint8_t horizontal, uint8_t screen, uint8_t lives) {
  this->heighest = (static_cast<int>(horizontal) << 8) | screen;
  if (lives < 2) {
    // game over
  }
}
