from time import sleep

from cynes.windowed import WindowedNES

with WindowedNES("mario.nes") as nes:
    while not nes.should_close:
        lives = nes[0x75A]
        level = nes[0x0760]
        x_horizontal = nes[0x006D]
        x_on_screen = nes[0x0086]
        horizontal_speed = nes[0x0057]
        y_position_on_screen = nes[0x00CE]
        x_position = (x_horizontal << 8) | x_on_screen
        frame = nes.step()
        sleep(0.01)

        score_bcd = [
            nes[0x07DD],  # 1000000 and 100000 place
            nes[0x07DE],  # 10000 and 1000 place
            nes[0x07DF],  # 100 and 10 place
            nes[0x07E0],  # 1 place (if applicable)
            nes[0x07E1],  # 1 place (if applicable)
            nes[0x07E2],  # 1 place (if applicable)
        ]

        # Convert BCD to integer score
        score = 0
        for byte in score_bcd:
            score = score * 100 + ((byte >> 4) * 10) + (byte & 0x0F)

        print(
            f"pos: {x_position}, level: {level}, score: {score}, horizontal_speed: {horizontal_speed}"
        )
