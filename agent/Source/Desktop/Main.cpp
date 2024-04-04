// Precompiler flag to enable or disable SDL
#define USE_SDL 1

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

void downsample(Uint32* originalBuffer, Uint32* newBuffer, int originalWidth, int originalHeight, int newWidth, int newHeight) {
    // Assume new dimensions are half the original dimensions for simplicity
    int blockWidth = originalWidth / newWidth;
    int blockHeight = originalHeight / newHeight;

    for (int newY = 0; newY < newHeight; ++newY) {
        for (int newX = 0; newX < newWidth; ++newX) {
            // Accumulate values from a 2x2 block in the original image
            Uint32 sumR = 0, sumG = 0, sumB = 0;
            for (int dy = 0; dy < blockHeight; ++dy) {
                for (int dx = 0; dx < blockWidth; ++dx) {
                    Uint32 pixel = originalBuffer[(newY * blockHeight + dy) * originalWidth + (newX * blockWidth + dx)];
                    sumR += (pixel >> 16) & 0xFF;
                    sumG += (pixel >> 8) & 0xFF;
                    sumB += pixel & 0xFF;
                }
            }

            // Average the accumulated values
            Uint32 avgR = (sumR / (blockWidth * blockHeight)) & 0xFF;
            Uint32 avgG = (sumG / (blockWidth * blockHeight)) & 0xFF;
            Uint32 avgB = (sumB / (blockWidth * blockHeight)) & 0xFF;

            // Set the averaged pixel in the new buffer
            newBuffer[newY * newWidth + newX] = (avgR << 16) | (avgG << 8) | avgB;
        }
    }
}


#if USE_SDL
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
#else
void handleCustomEvents(MedNES::Controller &controller, bool &is_running)
{
  // Custom event handling logic here
  // Replace SDL event handling with another form of input handling if needed
}
#endif

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

int main(int argc, char **argv)
{
  const int originalWidth = 256;
  const int originalHeight = 240;
  const int newWidth = 32;  // New desired width
  const int newHeight = 30; // New desired height   const int originalWidth = 256;
  std::string romPath = "";
  std::string COMMAND_LINE_ERROR_MESSAGE = "Use -insert <path/to/rom> to start playing.";
  bool fullscreen = false;

  if (argc < 2)
  {
    std::cout << COMMAND_LINE_ERROR_MESSAGE << std::endl;
    return 1;
  }

  std::string option = argv[1];

  if (option == "-insert")
  {
    romPath = argv[2];
  }
  else
  {
    std::cout << "Unknown option '" << option << "'. " << COMMAND_LINE_ERROR_MESSAGE << std::endl;
    return 1;
  }

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

  if (fullscreen)
  {
    SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
  }

  SDL_Renderer *renderer =
      SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | (headlessMode ? 0 : SDL_RENDERER_PRESENTVSYNC));
  SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                           SDL_TEXTUREACCESS_STATIC, newWidth, newHeight);
#endif

  bool is_running = true;

  MedNES::ROM rom;
  rom.open(romPath);
  rom.printHeader();
  MedNES::Mapper *mapper = rom.getMapper();

  if (mapper == NULL)
  {
    std::cout << "Unknown mapper.";
    return 1;
  }

  auto ppu = MedNES::PPU(mapper);
  MedNES::Controller controller;
  auto cpu = MedNES::CPU6502(mapper, &ppu, &controller);
  cpu.reset();

  int nmiCounter = 0;
  float duration = 0;
  auto t1 = std::chrono::high_resolution_clock::now();
  skip_menu(ppu, cpu, controller);
  while (is_running)
  {
    cpu.step();
    if (ppu.generateFrame)
    {
#if USE_SDL
      handleSDLEvents(controller, map, is_running);
#else
      handleCustomEvents(controller, is_running);
#endif

      nmiCounter++;
      auto t2 = std::chrono::high_resolution_clock::now();
      duration += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
      t1 = std::chrono::high_resolution_clock::now();

      if (nmiCounter == 10)
      {
        float avgFps = 1000 / (duration / nmiCounter);
#if USE_SDL
        std::string fpsTitle = window_title + " (FPS: " + std::to_string((int)avgFps) + ")";
        SDL_SetWindowTitle(window, fpsTitle.c_str());
#endif
        nmiCounter = 0;
        duration = 0;
      }

      ppu.generateFrame = false;
#if USE_SDL
    Uint32* originalBuffer = ppu.buffer; // Your original buffer

    // Allocate memory for the new, smaller buffer
    Uint32* newBuffer = new Uint32[newWidth * newHeight];
    downsample(originalBuffer, newBuffer, originalWidth, originalHeight, newWidth, newHeight);

      SDL_RenderSetScale(renderer, 2, 2);
      SDL_UpdateTexture(texture, NULL, newBuffer, newWidth * sizeof(Uint32));
      SDL_RenderClear(renderer);
      SDL_RenderCopy(renderer, texture, NULL, NULL);
      SDL_RenderPresent(renderer);
    delete[] newBuffer;
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
