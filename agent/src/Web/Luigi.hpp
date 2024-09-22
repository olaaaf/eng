#pragma once
#include "Ai/Score.hpp"
#include "Web/Episode.hpp"
#include <functional>
#include <memory>
#include <string>
#include <torch/jit.h>

class LuigiClient {
public:
  LuigiClient() = delete;
  ~LuigiClient() = delete;
  static std::unique_ptr<torch::jit::Module>
  fetchModel(const std::string &base_url, int model_id,
             std::function<void()> callback);
  static void submitScore(const std::string &base_url, const Score *score,
                          int model_id, std::function<void(bool)> callback);

private:
  const std::string base_url;

  static void submitEpisode(const std::string &base_url,
                            std::unique_ptr<Episode> episode,
                            std::function<void(bool)> callback);
};