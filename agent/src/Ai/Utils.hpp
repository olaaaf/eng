#pragma once

#include "../Core/6502.hpp"
#include "../Core/Controller.hpp"
#include "../Core/PPU.hpp"
#include "Ai/Model.hpp"

#include <cstdint>
#include <fstream>
#include <string>

constexpr int kOriginalWidth = 256;
constexpr int kOriginalHeight = 240;
constexpr int kNewWidth = 64;
constexpr int kNewHeight = 60;

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

void fastDownsampleToGrayscale(uint32_t *originalBuffer, uint8_t *newBuffer) {
  constexpr int xRatio = (kOriginalWidth << 16) / kNewWidth + 1;
  constexpr int yRatio = (kOriginalHeight << 16) / kNewHeight + 1;

  for (int y = 0; y < kNewHeight; y++) {
    int sy = (y * yRatio) >> 16;
    for (int x = 0; x < kNewWidth; x++) {
      int sx = (x * xRatio) >> 16;
      uint32_t pixel = originalBuffer[sy * kOriginalWidth + sx];

      // Fast grayscale conversion
      uint8_t gray =
          ((pixel & 0xFF) + ((pixel >> 8) & 0xFF) + ((pixel >> 16) & 0xFF)) / 3;

      newBuffer[y * kNewWidth + x] = gray;
    }
  }
}

void saveBMP(const uint8_t *model_input, int fileNumber) {
  const int width = 64;
  const int height = 60;
  const int paddingSize = (4 - (width % 4)) % 4;
  const int fileSize = 54 + (width + paddingSize) * height;

  uint8_t bmpFileHeader[14] = {'B',
                               'M',
                               (uint8_t)(fileSize),
                               (uint8_t)(fileSize >> 8),
                               (uint8_t)(fileSize >> 16),
                               (uint8_t)(fileSize >> 24),
                               0,
                               0,
                               0,
                               0,
                               54,
                               0,
                               0,
                               0};

  uint8_t bmpInfoHeader[40] = {40,
                               0,
                               0,
                               0,
                               (uint8_t)(width),
                               (uint8_t)(width >> 8),
                               (uint8_t)(width >> 16),
                               (uint8_t)(width >> 24),
                               (uint8_t)(height),
                               (uint8_t)(height >> 8),
                               (uint8_t)(height >> 16),
                               (uint8_t)(height >> 24),
                               1,
                               0,
                               8,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0};

  std::string filename = "output_" + std::to_string(fileNumber) + ".bmp";
  std::ofstream file(filename, std::ios::binary);

  if (!file) {
    return; // Error handling: unable to create file
  }

  file.write(reinterpret_cast<char *>(bmpFileHeader), 14);
  file.write(reinterpret_cast<char *>(bmpInfoHeader), 40);

  // Write color palette (grayscale)
  for (int i = 0; i < 256; i++) {
    uint8_t color[4] = {static_cast<uint8_t>(i), static_cast<uint8_t>(i),
                        static_cast<uint8_t>(i), 0};
    file.write(reinterpret_cast<char *>(color), 4);
  }

  // Write pixel data
  for (int y = height - 1; y >= 0; y--) {
    for (int x = 0; x < width; x++) {
      file.write(reinterpret_cast<const char *>(&model_input[y * width + x]),
                 1);
    }
    // Add padding
    for (int p = 0; p < paddingSize; p++) {
      uint8_t pad = 0;
      file.write(reinterpret_cast<char *>(&pad), 1);
    }
  }

  file.close();
}
