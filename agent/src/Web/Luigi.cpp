#include "Luigi.hpp"
#include <drogon/drogon.h>
#include <istream>
#include <json/json.h>
#include <memory>
#include <sstream>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>
#include <vector>

LuigiClient::LuigiClient(const std::string &url) : base_url(url) {}

torch::jit::Module LuigiClient::fetchModel(
    int model_id,
    std::function<void(std::shared_ptr<torch::jit::Module>)> callback) {
  auto client = drogon::HttpClient::newHttpClient(base_url);
  auto req = drogon::HttpRequest::newHttpJsonRequest(Json::Value());
  req->setPath("/get_model");
  req->setMethod(drogon::HttpMethod::Post);
  req->setParameter("model_id", std::to_string(model_id));

  client->sendRequest(req, [callback](drogon::ReqResult result,
                                      const drogon::HttpResponsePtr &response) {
    if (result != drogon::ReqResult::Ok) {
      std::cerr << "Error fetching model" << std::endl;
      callback(nullptr);
      return;
    }

    auto json = response->getJsonObject();
    if (!json || !(*json)["model_base64"].isString()) {
      std::cerr << "Invalid response format" << std::endl;
      callback(nullptr);
      return;
    }

    std::string model_base64 = (*json)["model_base64"].asString();

    std::string model_data = drogon::utils::base64Decode(model_base64);
    std::istringstream model_stream(model_data);

    try {
      auto module_ = std::make_unique<torch::jit::Module>(
          torch::jit::load(model_stream, std::nullopt));
      return module_;
    } catch (const c10::Error &e) {
      std::cerr << "Error loading the model: " << e.what() << std::endl;
      callback(nullptr);
    }
  });
}

void LuigiClient::submitScore(std::shared_ptr<Score> score, int model_id,
                              std::function<void(bool)> callback) {
  auto episode = Episode::fromScore(score, model_id, 0, 0);
  submitEpisode(std::move(episode), std::move(callback));
}

void LuigiClient::submitEpisode(std::unique_ptr<Episode> episode,
                                std::function<void(bool)> callback) {
  auto client = drogon::HttpClient::newHttpClient(base_url);
  auto req = drogon::HttpRequest::newHttpJsonRequest(episode->toJson());
  req->setPath("/submit_episode");
  req->setMethod(drogon::HttpMethod::Post);

  client->sendRequest(req, [callback](drogon::ReqResult result,
                                      const drogon::HttpResponsePtr &response) {
    if (result != drogon::ReqResult::Ok) {
      std::cerr << "Error submitting episode" << std::endl;
      callback(false);
      return;
    }

    auto json = response->getJsonObject();
    if (!json || !(*json)["status"].isString()) {
      std::cerr << "Invalid response format" << std::endl;
      callback(false);
      return;
    }

    std::string status = (*json)["status"].asString();
    callback(status == "Score submitted and training queued");
  });
}
