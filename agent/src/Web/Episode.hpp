#pragma once
#include "Ai/Score.hpp"
#include <json/json.h>
#include <memory>

class Episode {
public:
  std::shared_ptr<Score> score;
  int model_id;
  int final_score;
  bool died;

  Json::Value toJson() const;
  static std::unique_ptr<Episode> fromScore(std::shared_ptr<Score> score,
                                            int model_id, bool died,
                                            int final_score);
};