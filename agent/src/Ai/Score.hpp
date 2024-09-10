#pragma once
#include <cstdint>
#include <functional>

class Score {
public:
    Score(std::function<uint8_t()>, std::function<uint8_t()>, std::function<uint8_t()>);

};
