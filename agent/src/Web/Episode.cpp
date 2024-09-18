#include "Episode.hpp"
#include <json/json.h>
#include <memory>

Json::Value Episode::toJson() const {
  Json::Value json;
  json["model_id"] = model_id;
  json["final_score"] = final_score;
  json["died"] = died;

  Json::Value states;
  states["x"] = Json::Value(Json::arrayValue);
  states["y"] = Json::Value(Json::arrayValue);
  states["x_speed"] = Json::Value(Json::arrayValue);
  states["action"] = Json::Value(Json::arrayValue);

  for (size_t i = 0; i < score->states.x.size(); ++i) {
    states["x"].append(score->states.x[i]);
    states["y"].append(score->states.y[i]);
    states["x_speed"].append(score->states.x_speed[i]);
    states["action"].append(score->states.actions[i]);
  }

  json["states"] = states;
  return json;
}

std::unique_ptr<Episode> Episode::fromScore(std::shared_ptr<Score> score,
                                            int model_id, bool died,
                                            int final_score) {
  std::unique_ptr<Episode> episode = std::make_unique<Episode>();
  episode->score = score;
  episode->died = died;
  episode->model_id = model_id;
  episode->final_score = final_score;
  return episode;
}