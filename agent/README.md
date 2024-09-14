# Model agent

## How it works

Loads the model and feeds the image output from the [MedNES](https://github.com/wpmed92/MedNES) emulator running the game Super Mario Bros. The frame is scaled down and converted into grayscale (1 byte per pixel) then fed to
the model as input.

### Evaluation

The program reads necessary fields from the NES memory in order to
provide model evaluation.

| variable name                | memory address |
|------------------------------|----------------|
| horizontal position in level [[^1]](#1) | 0x006D         |
| position on screen           | 0x0086         |
| horizontal speed             | 0x0057         |
| lives: 255 means game over   | 0x075A         |  

<a id="1">[1]</a> The absolute position in the level is calculated as: `(horizontal_position << 8) | screen_position`
