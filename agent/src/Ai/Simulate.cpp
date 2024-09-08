// Precompiler flag to enable or disable SDL
#define USE_SDL 0
#if USE_SDL
#include <SDL.h>
#endif

#include <chrono>
#include <cstdint>
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

int main()
{
  std::string romPath = "mario.nes";

#if USE_SDL
  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) < 0)
  {
    std::cout << "SDL could not initialize." << SDL_GetError() << std::endl;
  }

  SDL_GameController *con = nullptr;

  for (int i = 0; i < SDL_NumJoysticks(); i++)
  {
    if (SDL_IsGameController(i))
    {
      con = SDL_GameControllerOpen(i);
      std::cout << "Controller detected.";
      break;
    }
  }

  std::map<int, int> map;
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_A, SDLK_a));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_B, SDLK_b));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_START, SDLK_RETURN));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_DPAD_UP, SDLK_UP));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_DPAD_DOWN, SDLK_DOWN));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_DPAD_LEFT, SDLK_LEFT));
  map.insert(std::pair<int, int>(SDL_CONTROLLER_BUTTON_DPAD_RIGHT, SDLK_RIGHT));

  SDL_Window *window;
  std::string window_title = "MedNES";
  bool headlessMode = false;

  window = SDL_CreateWindow(window_title.c_str(),    // window title
                            SDL_WINDOWPOS_UNDEFINED, // initial x position
                            SDL_WINDOWPOS_UNDEFINED, // initial y position
                            512,                     // width, in pixels
                            480,                     // height, in pixels
                            SDL_WINDOW_SHOWN         // flags - see below
  );

  if (window == NULL)
  {
    printf("Could not create window: %s\n", SDL_GetError());
    return 1;
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | (headlessMode ? 0 : SDL_RENDERER_PRESENTVSYNC));
  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                           SDL_TEXTUREACCESS_STATIC, kNewWidth, kNewHeight);
#endif

  bool is_running = true;

  MedNES::ROM rom;
  rom.open(romPath);
  rom.printHeader();
  MedNES::Mapper *mapper = rom.getMapper();

  auto ppu = MedNES::PPU(mapper);
  MedNES::Controller controller;
  auto cpu = MedNES::CPU6502(mapper, &ppu, &controller);
  cpu.reset();

#if USE_SDL
  int nmiCounter = 0;
  float duration = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
#endif
  skip_menu(ppu, cpu, controller);
  while (is_running)
  {
    cpu.step();
    if (ppu.generateFrame)
    {
      ppu.generateFrame = false;
      downsampleToGrayscale(ppu.buffer, model_input);
#if USE_SDL
      handleSDLEvents(controller, map, is_running);
#else
      handleCustomEvents(controller, is_running);
#endif

#if USE_SDL
      nmiCounter++;
      auto t2 = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
      t1 = std::chrono::high_resolution_clock::now();

      if (nmiCounter == 10)
      {
        float avgFps = 1000 / (duration / nmiCounter);
        std::string fpsTitle = window_title + " (FPS: " + std::to_string((int)avgFps) + ")";
        SDL_SetWindowTitle(window, fpsTitle.c_str());
        nmiCounter = 0;
        duration = 0;
      }
      Uint32 *displayBuffer = new Uint32[kNewWidth * kNewHeight];

      for (int i = 0; i < kNewWidth * kNewHeight; ++i)
      {
        uint8_t gray = model_input[i];
        displayBuffer[i] = (255 << 24) | (gray << 16) | (gray << 8) | gray;
      }

      SDL_RenderSetScale(renderer, 2, 2);
      SDL_UpdateTexture(texture, NULL, displayBuffer, kNewWidth * sizeof(Uint32));
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, NULL, NULL);
      SDL_RenderPresent(renderer);
      delete[] displayBuffer;
#endif
    }
  }

#if USE_SDL
  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
#endif

  return 0;
}
