#pragma once
#include "Ai/Score.hpp"
#include "Web/Episode.hpp"
#include <functional>
#include <memory>
#include <string>
#include <torch/jit.h>

class LuigiClient {
public:
  LuigiClient(const std::string &base_url);
  void
  fetchModel(int model_id,
             std::function<void(std::shared_ptr<torch::jit::Module>)> callback);
  void submitScore(std::shared_ptr<Score> score, int model_id,
                   std::function<void(bool)> callback);

private:
  const std::string base_url;

  void submitEpisode(std::unique_ptr<Episode> episode,
                     std::function<void(bool)> callback);
};