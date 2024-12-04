from cynes.windowed import WindowedNES

with WindowedNES("mario.nes") as nes:
    while not nes.should_close:
        lives = nes[0x75A]
        x_horizontal = nes[0x006D]
        x_on_screen = nes[0x0086]
        horizontal_speed = nes[0x0057]
        y_position_on_screen = nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen
        frame = nes.step()
        print(x_position)
