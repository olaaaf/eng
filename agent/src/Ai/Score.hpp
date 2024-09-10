#pragma once
#include <cstdint>

class Score {
public:
  Score();
  void frame(uint8_t, uint8_t, uint8_t);

private:
  int heighest;
};
