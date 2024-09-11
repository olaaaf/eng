#pragma once
#include <cstdint>

enum GameState { RUNNING, GAME_OVER };

class Score {
public:
  Score();
  GameState frame(uint8_t horizontal, uint8_t screen, uint8_t lives);
  void report();

private:
  int heighest;
};
