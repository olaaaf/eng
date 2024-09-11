// Precompiler flag to enable or disable SDL
#include "Ai/Score.hpp"
#include <iostream>
#include <ostream>
#define USE_SDL 0

#include <cstdint>
#include <string>

#include "../Core/6502.hpp"
#include "../Core/Controller.hpp"
#include "../Core/Mapper/Mapper.hpp"
#include "../Core/PPU.hpp"
#include "../Core/ROM.hpp"
#include "Model.hpp"
#include "Utils.hpp"

uint8_t model_input[kNewHeight * kNewWidth];
constexpr std::string romPath = "mario.nes";
constexpr int frames_to_input_generation = 20;

int simulate(Score &score) {
  bool is_running = true;

  // create the model :)
  auto model_result = Model::Create("hey");
  std::unique_ptr<Model> model;
  if (model_result) {
    model = std::move(model_result.value());
  } else {
    std::cerr << "Error loading the model\n";
    model = std::make_unique<Model>();
    // return 1;
  }

  MedNES::ROM rom;
  rom.open(romPath);
  MedNES::Mapper *mapper = rom.getMapper();

  auto ppu = MedNES::PPU(mapper);
  MedNES::Controller controller;
  auto cpu = MedNES::CPU6502(mapper, &ppu, &controller);
  cpu.reset();
  skip_menu(ppu, cpu, controller);
  ButtonPresses model_output;

  int frame = 0;
  while (is_running) {
    cpu.step();

    // as downsampling and model prediction is somewhat costly
    // this happens every frames_to_input_generation
    if (frame == 0) {
      fastDownsampleToGrayscale(ppu.buffer, model_input);
      model_output = model->predict(model_input);
      controller.setButtonPressed(model_output);
    }

    // position evaluation
    if (score.frame(cpu.read(0x006d), cpu.read(0x0086), cpu.read(0x075A)) ==
        GameState::GAME_OVER) {
      return 1;
    }

    ppu.generateFrame = false;
    frame = (frame + 1) % frames_to_input_generation;
  }

  return 0;
}

int main() {
  Score score;
  simulate(score);
  score.report();
}
