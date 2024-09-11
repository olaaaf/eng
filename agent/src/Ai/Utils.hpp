#pragma once

#include "../Core/6502.hpp"
#include "../Core/Controller.hpp"
#include "../Core/PPU.hpp"
#include "Ai/Model.hpp"

constexpr int kOriginalWidth = 256;
constexpr int kOriginalHeight = 240;
constexpr int kNewWidth = 32;
constexpr int kNewHeight = 30;

constexpr int kBlockWidth = kOriginalWidth / kNewWidth;
constexpr int kBlockHeight = kOriginalHeight / kNewHeight;

struct SimulationSettings {};

void skip_menu(MedNES::PPU &ppu, MedNES::CPU6502 &cpu,
               MedNES::Controller &controller) {
  uint8_t current_frame = 0;
  bool game_on = false;
  while (!game_on) {
    cpu.step();
    if (ppu.generateFrame) {
      ButtonPresses p;
      p.enter = (current_frame / 64 == 0);
      controller.setButtonPressed(p);
      current_frame++;
      ppu.generateFrame = false;
    }
    if (cpu.read(0x0776) != 0) {
      cpu.write(0x0777, 0x0000);
      cpu.write(0x0776, 0x0000);
      game_on = true;
    }
    current_frame++;
  }
}

void downsampleToGrayscale(uint32_t *originalBuffer, uint8_t *newBuffer) {
  for (int newY = 0; newY < kNewHeight; ++newY) {
    for (int newX = 0; newX < kNewWidth; ++newX) {
      // Accumulate values from a 2x2 block in the original image
      uint32_t sum = 0;
      for (int dy = 0; dy < kBlockHeight; ++dy) {
        for (int dx = 0; dx < kBlockWidth; ++dx) {
          uint32_t pixel =
              originalBuffer[(newY * kBlockHeight + dy) * kOriginalWidth +
                             (newX * kBlockWidth + dx)];
          // Convert pixel to grayscale using a simple average or a weighted
          // method
          uint8_t gray = ((pixel >> 16) & 0xFF) * 0.3 +
                         ((pixel >> 8) & 0xFF) * 0.59 + (pixel & 0xFF) * 0.11;
          sum += gray;
        }
      }

      // Average the accumulated values
      newBuffer[newY * kNewWidth + newX] = sum / (kBlockWidth * kBlockHeight);
    }
  }
}

void fastDownsampleToGrayscale(uint32_t *originalBuffer, uint8_t *newBuffer,
                               int originalWidth, int originalHeight,
                               int newWidth, int newHeight) {
  int xRatio = (originalWidth << 16) / newWidth + 1;
  int yRatio = (originalHeight << 16) / newHeight + 1;

  for (int y = 0; y < newHeight; y++) {
    int sy = (y * yRatio) >> 16;
    for (int x = 0; x < newWidth; x++) {
      int sx = (x * xRatio) >> 16;
      uint32_t pixel = originalBuffer[sy * originalWidth + sx];

      // Fast grayscale conversion
      uint8_t gray =
          ((pixel & 0xFF) + ((pixel >> 8) & 0xFF) + ((pixel >> 16) & 0xFF)) / 3;

      newBuffer[y * newWidth + x] = gray;
    }
  }
}