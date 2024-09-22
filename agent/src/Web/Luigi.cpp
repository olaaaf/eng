#include "Luigi.hpp"
#include <cstddef>
#include <drogon/HttpClient.h>
#include <drogon/HttpRequest.h>
#include <drogon/drogon.h>
#include <drogon/utils/Utilities.h>
#include <exception>
#include <fstream>
#include <istream>
#include <iterator>
#include <json/json.h>
#include <memory>
#include <sstream>
#include <string>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/script.h>
#include <vector>

std::unique_ptr<torch::jit::Module>
LuigiClient::fetchModel(const std::string &base_url, int model_id,
                        const std::function<void()> callback) {
  auto client = drogon::HttpClient::newHttpClient(base_url);
  auto req = drogon::HttpRequest::newHttpJsonRequest(Json::Value());

  req->setPath("/get_model");
  req->setMethod(drogon::HttpMethod::Post);
  req->setParameter("model_id", std::to_string(model_id));

  client->sendRequest(req, [callback](drogon::ReqResult result,
                                      const drogon::HttpResponsePtr &response) {
    if (result != drogon::ReqResult::Ok) {
      std::cerr << "error while sending request to server! result: " << result
                << std::endl;
      return;
    }

    auto json = response->getJsonObject();
    if (!json || !(*json)["model_base64"].isString()) {
      std::cerr << "error with the json" << std::endl;
      return;
    }

    auto decoded =
        drogon::utils::base64DecodeToVector((*json)["model_base64"].asString());
    try {
      std::ofstream file("model", std::ios::trunc | std::ios::binary);
      std::ostream_iterator<char> output_iterator(file);
      std::copy(decoded.begin(), decoded.end(), output_iterator);
    } catch (std::exception e) {
      std::cerr << e.what() << std::endl;
    }
    callback();
  });

  return nullptr;
}

void LuigiClient::submitScore(const std::string &base_url, const Score *score,
                              int model_id,
                              std::function<void(bool)> callback) {
  auto episode = Episode::fromScore(score, model_id, 0, 0);
  submitEpisode(base_url, std::move(episode), std::move(callback));
}

void LuigiClient::submitEpisode(const std::string &base_url,
                                std::unique_ptr<Episode> episode,
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
