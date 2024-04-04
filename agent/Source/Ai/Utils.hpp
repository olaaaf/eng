#pragma once

#include "../Core/PPU.hpp"
#include "../Core/6502.hpp"
#include "../Core/Controller.hpp"

constexpr int kOriginalWidth = 256;
constexpr int kOriginalHeight = 240;
constexpr int kNewWidth = 32;
constexpr int kNewHeight = 30;

constexpr int kBlockWidth = kOriginalWidth / kNewWidth;
constexpr int kBlockHeight = kOriginalHeight / kNewHeight;

void skip_menu(MedNES::PPU &ppu, MedNES::CPU6502 &cpu, MedNES::Controller &controller)
{
  uint8_t pressed = 0;
  bool game_on = false;
  while (!game_on)
  {
    cpu.step();
    if (ppu.generateFrame)
    {
      controller.setButtonPressed(SDLK_RETURN, pressed / 64 == 0);
      pressed++;
      ppu.generateFrame = false;
    }
    if (cpu.read(0x0776) != 0)
    {
      cpu.write(0x0777, 0x0000);
      cpu.write(0x0776, 0x0000);
      game_on = true;
    }
  }
}

void downsampleToGrayscale(Uint32* originalBuffer, uint8_t* newBuffer) {
    for (int newY = 0; newY < kNewHeight; ++newY) {
        for (int newX = 0; newX < kNewWidth; ++newX) {
            // Accumulate values from a 2x2 block in the original image
            Uint32 sum = 0;
            for (int dy = 0; dy < kBlockHeight; ++dy) {
                for (int dx = 0; dx < kBlockWidth; ++dx) {
                    Uint32 pixel = originalBuffer[(newY * kBlockHeight + dy) * kOriginalWidth + (newX * kBlockWidth + dx)];
                    // Convert pixel to grayscale using a simple average or a weighted method
                    uint8_t gray = ((pixel >> 16) & 0xFF) * 0.3 + ((pixel >> 8) & 0xFF) * 0.59 + (pixel & 0xFF) * 0.11;
                    sum += gray;
                }
            }

            // Average the accumulated values
            newBuffer[newY * kNewWidth + newX] = sum / (kBlockWidth * kBlockHeight);
        }
    }
}

void handleCustomEvents(MedNES::Controller &controller, bool &is_running)
{
}

void handleSDLEvents(MedNES::Controller &controller, std::map<int, int> &map, bool &is_running)
{
  SDL_Event event;
  while (SDL_PollEvent(&event))
  {
    switch (event.type)
    {
    case SDL_CONTROLLERBUTTONDOWN:
      controller.setButtonPressed(map.find(event.cbutton.button)->second, true);
      break;
    case SDL_CONTROLLERBUTTONUP:
      controller.setButtonPressed(map.find(event.cbutton.button)->second, false);
      break;
    case SDL_KEYDOWN:
      controller.setButtonPressed(event.key.keysym.sym, true);
      break;
    case SDL_KEYUP:
      controller.setButtonPressed(event.key.keysym.sym, false);
      break;
    case SDL_QUIT:
      is_running = false;
      break;
    default:
      break;
    }
  }
}
