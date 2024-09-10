// Precompiler flag to enable or disable SDL
#define USE_SDL 0
#if USE_SDL
#include <SDL.h>
#endif

#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <string>

#include "../Core/6502.hpp"
#include "../Core/Controller.hpp"
#include "../Core/Mapper/Mapper.hpp"
#include "../Core/PPU.hpp"
#include "../Core/ROM.hpp"
#include "Model.hpp"
#include "Utils.hpp"

uint8_t model_input[kNewHeight * kNewWidth];

int main() {
  std::string romPath = "mario.nes";
  bool is_running = true;

  MedNES::ROM rom;
  rom.open(romPath);
  rom.printHeader();
  MedNES::Mapper *mapper = rom.getMapper();

  auto ppu = MedNES::PPU(mapper);
  MedNES::Controller controller;
  auto cpu = MedNES::CPU6502(mapper, &ppu, &controller);
  cpu.reset();
  skip_menu(ppu, cpu, controller);
  while (is_running) {
    cpu.step();
    downsampleToGrayscale(ppu.buffer, model_input);

    handleCustomEvents(controller, is_running);
  }

  return 0;
}